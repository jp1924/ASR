from typing import Any, Dict

import torch
from torch._tensor import Tensor
from torch.nn.modules import Module
from transformers import Trainer
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)

if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward


# 이게 acclerate의 기능과 맞물려서 어떤 사이드 이팩트를 만들어 낼지 모르겠다.
# copied_from: examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py
def multiply_grads(params: torch.nn.Parameter, loss: torch.Tensor) -> None:
    """Multiplies grads by a constant *loss*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(loss):
                loss = loss.to(p.grad.device)
            p.grad.data.mul_(loss)


class Wav2Vec2Pretrainer(Trainer):
    global_outputs = None
    global_percent_masked = None
    global_num_losses = None

    def training_step(self, model: Module, inputs: Dict[str, Tensor | Any]) -> Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        sub_attention_mask = inputs.pop("sub_attention_mask", None)
        sub_attention_mask = (
            sub_attention_mask if sub_attention_mask is not None else torch.ones_like(inputs["mask_time_indices"])
        )
        num_losses = inputs["mask_time_indices"].sum()

        percent_masked = num_losses / sub_attention_mask.sum()

        if is_sagemaker_mp_enabled():
            # NOTE: sagemaker에선 outputs가 나오질 않음! 참고
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # NOTE: accelerate에서 gradient accumulation을 자동으로 계산 해줌. 어떻게 하는지는 모르지만....
            self.accelerator.backward(loss)

        # NOTE: https://github.com/huggingface/transformers/pull/13877#discussion_r723197919 참고

        if self.accelerator.state.num_processes > 1:
            num_losses = self.accelerator.gather_for_metrics(num_losses).sum()
            gradient_multiplier = self.accelerator.state.num_processes / num_losses
            multiply_grads(model.module.parameters(), gradient_multiplier)
        else:
            multiply_grads(model.parameters(), 1 / num_losses)

        self.gumbel_temperature = max(
            self.args.max_gumbel_temperature * self.args.gumbel_temperature_decay**self.state.global_step,
            self.args.min_gumbel_temperature,
        )
        if hasattr(model, "module"):
            model.module.set_gumbel_temperature(self.gumbel_temperature)
        else:
            model.set_gumbel_temperature(self.gumbel_temperature)

        # TODO: 다른 loss에도 gradient accumulation이 적용이 되었는지는 모르겠음. 이건 확인 필요.
        loss = loss.detach()

        # for logging
        outputs.loss.detach()
        outputs.contrastive_loss.detach()
        outputs.diversity_loss.detach()
        outputs.codevector_perplexity.detach()

        self.global_outputs = outputs
        self.global_percent_masked = percent_masked.detach()
        self.global_num_losses = num_losses.detach().item()

        return loss / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # TODO: ctc reduction이 sum이냐 mean이냐에 따라 연산하는 방식이 달라질거임. 그거에 맞춰서 계산하는 방법을 구해야 할 듯
            # all_gather + mean() to get average loss over all processes
            tr_contrastive_loss_scalar = (
                self._nested_gather(self.global_outputs.contrastive_loss / self.global_num_losses).sum().item()
            )
            tr_diversity_loss_scalar = (
                self._nested_gather(self.global_outputs.diversity_loss / self.global_num_losses).sum().item()
            )
            tr_percent_masked = (
                self._nested_gather(self.global_percent_masked / self.accelerator.num_processes).sum().item()
            )
            tr_loss_scalar = self._nested_gather(self.global_outputs.loss / self.global_num_losses).sum().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            # self.codevector_perplexity -= self.codevector_perplexity

            logs["loss"] = round((tr_loss_scalar), 4)
            logs["constrast_loss"] = round(tr_contrastive_loss_scalar, 4)
            logs["div_loss"] = round(tr_diversity_loss_scalar, 4)
            logs["%_mask_idx"] = round(tr_percent_masked, 4)
            logs["ppl"] = round(self.global_outputs.codevector_perplexity.item(), 4)
            logs["temp"] = round(self.gumbel_temperature, 4)

            if grad_norm is not None:
                grad_norm = grad_norm if isinstance(grad_norm, float) else grad_norm.detach().item()
                logs["grad_norm"] = round(grad_norm, 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
