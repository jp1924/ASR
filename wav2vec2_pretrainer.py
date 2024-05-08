from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)


if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_nested_concat,
    )

logger = logging.get_logger(__name__)


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
    contrastive_loss = 0.0
    diversity_loss = 0.0
    loss = 0.0
    codevector_perplexity = 0
    percent_masked = 0
    num_losses = 0

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
        self.contrastive_loss += outputs.contrastive_loss.detach() / self.args.gradient_accumulation_steps
        self.diversity_loss += outputs.diversity_loss.detach() / self.args.gradient_accumulation_steps
        self.loss += outputs.loss.detach() / self.args.gradient_accumulation_steps
        self.codevector_perplexity += outputs.codevector_perplexity.detach() / self.args.gradient_accumulation_steps
        self.percent_masked += percent_masked.detach() / self.args.gradient_accumulation_steps
        self.num_losses += num_losses.detach() / self.args.gradient_accumulation_steps

        # 사실상 return하는 loss는 사용하지 않음
        # inner_training_loop는 수정하기에는 리스크가 너무 큼.
        # 최소한의 코드 수정을 요하기 위해 이런 방식을 사용함.
        return loss / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # TODO: ctc reduction이 sum이냐 mean이냐에 따라 연산하는 방식이 달라질거임. 그거에 맞춰서 계산하는 방법을 구해야 할 듯
            # all_gather + mean() to get average loss over all processes
            tr_contrastive_loss_scalar = self._nested_gather(self.contrastive_loss / self.num_losses)
            tr_diversity_loss_scalar = self._nested_gather(self.diversity_loss / self.num_losses)
            tr_percent_masked = self._nested_gather(self.percent_masked / self.accelerator.num_processes)
            tr_perplexity = self._nested_gather(self.codevector_perplexity / self.accelerator.num_processes)
            tr_loss_scalar = self._nested_gather(self.loss / self.num_losses)

            # reset tr_loss to zero
            self.contrastive_loss -= self.contrastive_loss
            self.diversity_loss -= self.diversity_loss
            self.loss -= self.loss
            self.codevector_perplexity -= self.codevector_perplexity
            self.percent_masked -= self.percent_masked
            self.num_losses -= self.num_losses

            logs["loss"] = round(tr_loss_scalar.sum().item(), 4)
            logs["constrast_loss"] = round(tr_contrastive_loss_scalar.sum().item(), 4)
            logs["div_loss"] = round(tr_diversity_loss_scalar.sum().item(), 4)
            logs["%_mask_idx"] = round(tr_percent_masked.sum().item(), 4)
            logs["ppl"] = round(tr_perplexity.sum().item(), 4)
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

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        # NOTE: Wav2Vec2는 Unsupervised이기 때문에 label이 없음.

        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if isinstance(raw_outputs, dict):
                    raw_outputs = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                outputs = smp_nested_concat(raw_outputs)
            else:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    outputs = tuple(v for k, v in outputs.items() if k not in ignore_keys)

                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

            outputs = nested_detach(outputs)

            loss = outputs[0]
            codevector_perplexity = outputs[3]
            contrastive_loss = outputs[4]
            diversity_loss = outputs[5]
            num_loss = inputs["mask_time_indices"].sum()

        return (loss, codevector_perplexity, contrastive_loss, diversity_loss, num_loss)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        codevector_perplexities_host = None
        contrastive_losses_host = None
        diversity_losses_host = None
        num_losses_host = None
        losses_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_codevector_perplexities = None
        all_contrastive_losses = None
        all_diversity_losses = None
        all_num_losses = None
        all_losses = None
        all_inputs = None

        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, codevector_perplexity, contrastive_loss, diversity_loss, num_loss = self.prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if codevector_perplexity is not None:
                codevector_perplexities = self.gather_function((codevector_perplexity.repeat(batch_size)))
                codevector_perplexities_host = (
                    codevector_perplexities
                    if codevector_perplexities_host is None
                    else nested_concat(codevector_perplexities_host, codevector_perplexities, padding_index=-100)
                )
            if contrastive_loss is not None:
                contrastive_losses = self.gather_function((contrastive_loss.repeat(batch_size)))
                contrastive_losses_host = (
                    contrastive_losses
                    if contrastive_losses_host is None
                    else nested_concat(contrastive_losses_host, contrastive_losses, padding_index=-100)
                )
            if diversity_loss is not None:
                diversity_losses = self.gather_function((diversity_loss.repeat(batch_size)))
                diversity_losses_host = (
                    diversity_losses
                    if diversity_losses_host is None
                    else nested_concat(diversity_losses_host, diversity_losses, padding_index=-100)
                )
            if num_loss is not None:
                num_losses = self.gather_function((num_loss.repeat(batch_size)))
                num_losses_host = (
                    num_losses
                    if num_losses_host is None
                    else nested_concat(num_losses_host, num_losses, padding_index=-100)
                )
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if codevector_perplexities_host is not None:
                    codevector_perplexities = nested_numpify(codevector_perplexities_host)
                    all_codevector_perplexities = (
                        codevector_perplexities
                        if all_codevector_perplexities is None
                        else np.concatenate((all_codevector_perplexities, codevector_perplexities), axis=0)
                    )
                if contrastive_losses_host is not None:
                    contrastive_losses = nested_numpify(contrastive_losses_host)
                    all_contrastive_losses = (
                        contrastive_losses
                        if all_contrastive_losses is None
                        else np.concatenate((all_contrastive_losses, contrastive_losses), axis=0)
                    )
                if diversity_losses_host is not None:
                    diversity_losses = nested_numpify(diversity_losses_host)
                    all_diversity_losses = (
                        diversity_losses
                        if all_diversity_losses is None
                        else np.concatenate((all_diversity_losses, diversity_losses), axis=0)
                    )
                if num_losses_host is not None:
                    num_losses = nested_numpify(num_losses_host)
                    all_num_losses = (
                        num_losses if all_num_losses is None else np.concatenate((all_num_losses, num_losses), axis=0)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                # Set back to None to begin a new accumulation
                (
                    losses_host,
                    codevector_perplexities_host,
                    contrastive_losses_host,
                    diversity_losses_host,
                    num_losses_host,
                    inputs_host,
                ) = (None, None, None, None, None, None)

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if codevector_perplexities_host is not None:
            codevector_perplexities = nested_numpify(codevector_perplexities_host)
            all_codevector_perplexities = (
                codevector_perplexities
                if all_codevector_perplexities is None
                else np.concatenate((all_codevector_perplexities, codevector_perplexities), axis=0)
            )
        if contrastive_losses_host is not None:
            contrastive_losses = nested_numpify(contrastive_losses_host)
            all_contrastive_losses = (
                contrastive_losses
                if all_contrastive_losses is None
                else np.concatenate((all_contrastive_losses, contrastive_losses), axis=0)
            )
        if diversity_losses_host is not None:
            diversity_losses = nested_numpify(diversity_losses_host)
            all_diversity_losses = (
                diversity_losses
                if all_diversity_losses is None
                else np.concatenate((all_diversity_losses, diversity_losses), axis=0)
            )
        if num_losses_host is not None:
            num_losses = nested_numpify(num_losses_host)
            all_num_losses = (
                num_losses if all_num_losses is None else np.concatenate((all_num_losses, num_losses), axis=0)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = round(all_losses.sum().item() / all_num_losses.sum().item(), 4)

        if all_codevector_perplexities is not None:
            metrics[f"{metric_key_prefix}_ppl"] = round(all_codevector_perplexities.mean().item(), 4)

        if all_contrastive_losses is not None:
            metrics[f"{metric_key_prefix}_contrastive_loss"] = round(
                all_contrastive_losses.sum().item() / all_num_losses.sum().item(), 4
            )

        if all_diversity_losses is not None:
            metrics[f"{metric_key_prefix}_diversity_loss"] = round(
                all_diversity_losses.sum().item() / all_num_losses.sum().item(), 4
            )

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
