import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.utils.data import DataLoader, RandomSampler, Sampler

from transformers import FeatureExtractionMixin, PreTrainedModel, ProcessorMixin, Trainer
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations.deepspeed import deepspeed_init
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    LengthGroupedSampler,
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
    seed_worker,
)
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
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


# packing을 위한거
def __packing_getitems__(self, keys: List[List[int]]) -> List:
    """Can be used to get a batch using a list of integers indices."""

    return_ls = list()
    for key in keys:
        batch = self.__getitem__(key)
        n_examples = len(batch[next(iter(batch))])

        return_ls.append([{col: array[i] for col, array in batch.items()} for i in range(n_examples)])
    return return_ls


@dataclass
class DataPackingCollatorForWav2Vec2Pretraining(DataCollatorMixin):
    model: PreTrainedModel
    feature_extractor: FeatureExtractionMixin
    pack_max_seq: int = 512
    mask_time_prob: float = 0.65
    mask_time_length: int = 10
    mask_time_min_masks: int = 0
    num_negatives: int = 100
    return_tensors: str = "pt"

    do_old_packing: bool = False

    def _process_packing_list(self, feature_ls):
        input_values_ls, position_ids_ls, mask_time_indices_ls = list(), list(), list()
        for packing_ls in feature_ls:
            for feature in packing_ls:
                input_values = feature["input_values"]
                length = self.model._get_feat_extract_output_lengths(len(input_values)).item()

                mask_indices = _compute_mask_indices(
                    (1, length),
                    self.mask_time_prob,
                    self.mask_time_length,
                    min_masks=self.mask_time_min_masks,
                )

                input_values_ls.append(input_values)
                position_ids_ls.append(torch.arange(length))
                mask_time_indices_ls.append(torch.tensor(mask_indices))

        batch = dict()
        batch["input_values"] = input_values_ls
        batch["position_ids"] = torch.concat(position_ids_ls)
        batch["mask_time_indices"] = torch.concat(mask_time_indices_ls, dim=-1)
        batch["sampled_negative_indices"] = None

        return batch

    def _process_old_packing_list(self, feature_ls):
        batch_size = len(feature_ls)
        input_values = list()
        position_ids = np.zeros((batch_size, self.pack_max_seq)) - 1
        attention_mask = np.zeros((batch_size, 1, self.pack_max_seq, self.pack_max_seq))
        mask_time_indices = np.zeros((batch_size, self.pack_max_seq))
        for batch_idx, packing_ls in enumerate(feature_ls):
            start_idx = 0
            for pack in packing_ls:
                length = int(pack["length"])
                end_idx = start_idx + length
                mask_time_indices[batch_idx, start_idx:end_idx] = _compute_mask_indices(
                    (1, length),
                    self.mask_time_prob,
                    self.mask_time_length,
                    min_masks=self.mask_time_min_masks,
                )
                attention_mask[batch_idx, 0, start_idx:end_idx, start_idx:end_idx] = 1
                position_ids[batch_idx, start_idx:end_idx] = np.arange(length)
                start_idx = end_idx

            input_values.append([pack["input_values"] for pack in packing_ls])

        sampled_negative_indices = _sample_negative_indices(
            mask_time_indices.shape,
            self.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        batch = dict()
        batch["input_values"] = input_values
        batch["position_ids"] = torch.tensor(position_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        batch["mask_time_indices"] = torch.tensor(mask_time_indices)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices)
        batch["sub_attention_mask"] = torch.tensor(position_ids != -1, dtype=torch.long)

        return batch

    def _process_feature_list(self, feature_ls):
        features = [{"input_values": x["input_values"]} for x in feature_ls]

        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=True,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
        )

        device = batch.input_values.device
        batch_size, sample_size = batch.input_values.shape
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(sample_size).item()

        # make sure that no loss is computed on padded inputs
        if hasattr(batch, "attention_mask") is not None and self.model.training:
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length,
                batch.attention_mask,
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            shape=features_shape,
            mask_prob=self.mask_time_prob,
            mask_length=self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
            min_masks=self.mask_time_min_masks,
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape=features_shape,
            num_negatives=self.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch.to(self.model.dtype)

    def torch_call(self, feature_ls):
        if isinstance(feature_ls, list) and isinstance(feature_ls[0], list):
            if self.do_old_packing:
                return self._process_old_packing_list(feature_ls)
            else:
                return self._process_packing_list(feature_ls)
        else:
            return self._process_feature_list(feature_ls)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    processor: ProcessorMixin
    padding: Union[bool, str] = "longest"
    return_tensors: str = "pt"

    def torch_call(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features=input_features,
            labels=label_features,
            padding=self.padding,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
        )

        # replace padding with -100 to ignore loss correctly
        pad_token_id = self.processor.tokenizer.pad_token_type_id
        batch.labels[batch.labels == pad_token_id] = -100
        if hasattr(batch, "attention_mask"):
            batch["attention_mask"] = batch.attention_mask.to(torch.long)

        return batch


class PackingSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
        do_shuffle: bool = False,
    ):
        self.dataset = dataset

        self.packing_strategies = self._get_packing_strategies(
            lengths=lengths,
            max_seq_len=max_seq_len,
            max_seq_per_pack=max_seq_per_pack,
        )

        self.do_shuffle = do_shuffle
        self.lengths = lengths

        self.packing_sample_ls = self._transform_length_to_indices(
            strategies_per_length=self.packing_strategies,
            lengths=self.lengths,
        )

    def _get_packing_strategies(
        self,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
    ) -> dict:
        def add_pack(
            pack: List[int],
            count: int,
            tmp: defaultdict,
            final: defaultdict,
            limit: int,
            offset: int,
        ) -> None:
            if len(pack) == limit or offset == 0:
                final[offset].append((count, pack))
            else:
                tmp[offset].append((count, pack))

        seq_lens, counts = np.unique(lengths, return_counts=True)
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        histogram[seq_lens - 1] = counts

        reversed_histogram = np.flip(histogram)

        tmp_strategies_per_length = defaultdict(list)
        strategies_per_length = defaultdict(list)

        for i in range(max_seq_len):
            n_sequences_to_bin = reversed_histogram[i]
            length_to_bin = max_seq_len - i
            offset = i + 1  # largest possible offset
            while n_sequences_to_bin > 0:
                if (length_to_bin + offset) in tmp_strategies_per_length:
                    # extract shortest pack that will get modified
                    n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                    new_pack = pack + [length_to_bin]
                    count = min(n_sequences_to_pack, n_sequences_to_bin)
                    if n_sequences_to_pack > n_sequences_to_bin:
                        # old pack gets reduced
                        n_sequences_to_pack -= n_sequences_to_bin
                        tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                        n_sequences_to_bin = 0
                    else:
                        n_sequences_to_bin -= n_sequences_to_pack
                    add_pack(
                        new_pack, count, tmp_strategies_per_length, strategies_per_length, max_seq_per_pack, offset
                    )
                    # clean up to speed up main key search
                    if not tmp_strategies_per_length[length_to_bin + offset]:
                        tmp_strategies_per_length.pop(length_to_bin + offset)
                else:
                    offset -= 1
                # Does not fit anywhere. Create new pack.
                if offset < 0:
                    add_pack(
                        [length_to_bin],
                        n_sequences_to_bin,
                        tmp_strategies_per_length,
                        strategies_per_length,
                        max_seq_per_pack,
                        i,
                    )
                    n_sequences_to_bin = 0
        # merge all strategies
        for key in tmp_strategies_per_length:
            strategies_per_length[key].extend(tmp_strategies_per_length[key])

        return strategies_per_length

    def _transform_length_to_indices(self, strategies_per_length: dict, lengths: List[int]) -> List[List[int]]:
        length_to_indices = {}
        length_array = np.array(lengths)
        unique_lengths = np.unique(length_array).tolist()

        for length in unique_lengths:
            dataset_idx_ls = np.where(length_array == length)[0].tolist()
            if self.do_shuffle:
                random.shuffle(dataset_idx_ls)
            length_to_indices[length] = dataset_idx_ls

        pack_strategies_ls = [
            pack
            for strategies in strategies_per_length.values()
            for strategies_num, pack_strategies in strategies
            for pack in ([pack_strategies] * strategies_num)
        ]

        packing_sample_ls = list()
        for pack_strategies in pack_strategies_ls:
            pack_size = len(pack_strategies)
            strategie_position = 0

            dataset_idx_ls = list()
            while strategie_position + 1 <= pack_size:
                length = pack_strategies[strategie_position]
                pack_length_ls = length_to_indices[length]
                dataset_idx_ls.append(pack_length_ls.pop())
                length_to_indices[length] = pack_length_ls
                strategie_position += 1

            packing_sample_ls.append(dataset_idx_ls)

        if self.do_shuffle:
            random.shuffle(packing_sample_ls)

        return packing_sample_ls

    def __iter__(self):
        if self.do_shuffle:
            packing_sample_ls = self._transform_length_to_indices(
                strategies_per_length=self.packing_strategies,
                lengths=self.lengths,
            )
        else:
            packing_sample_ls = self.packing_sample_ls

        return iter(packing_sample_ls)

    def __len__(self):
        return len(self.packing_sample_ls)


class ASRPreTrainer(Trainer):
    contrastive_loss = 0.0
    diversity_loss = 0.0
    loss = 0.0
    codevector_perplexity = 0
    percent_masked = 0
    num_losses = 0

    def training_step(self, model: Module, inputs: Dict[str, Tensor | Any], num_items_in_batch=None) -> Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

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

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # BUG: Accelerate에서 gradient accumulation을 자동 처리하므로 여기서 loss *= gradient_accumulation_steps는
            #      중복 스케일링을 일으킬 수 있습니다.
            loss *= self.args.gradient_accumulation_steps
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
        return loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
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

            self.log(logs, start_time)

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
            self._save_checkpoint(model, trial)
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
            all_losses = all_losses.astype(np.float32)
            metrics[f"{metric_key_prefix}_loss"] = round(all_losses.sum().item() / all_num_losses.sum().item(), 4)

        if all_codevector_perplexities is not None:
            all_codevector_perplexities = all_codevector_perplexities.astype(np.float32)
            metrics[f"{metric_key_prefix}_ppl"] = round(all_codevector_perplexities.mean().item(), 4)

        if all_contrastive_losses is not None:
            all_contrastive_losses = all_contrastive_losses.astype(np.float32)
            metrics[f"{metric_key_prefix}_contrastive_loss"] = round(
                all_contrastive_losses.sum().item() / all_num_losses.sum().item(), 4
            )

        # inf가 발생하는 원인은 overflow 때문임.
        if all_diversity_losses is not None:
            all_diversity_losses = all_diversity_losses.astype(np.float32)
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

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # NOTE: packing을 사용할 경우 packing에 알맞은 getitems를 사용하도록 합니다.
        if self.args.do_packing:
            # 래핑된 함수를 정의하여 self를 전달할 수 있도록 합니다.
            def getitems_wrapper(keys):
                return __packing_getitems__(self.train_dataset, keys)

            setattr(self.train_dataset, "__getitems__", getitems_wrapper)

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            logger.info("length 혹은 train_dataset이 없어서 RandomSampler를 사용합니다.")
            return None

        if self.args.group_by_length and self.args.do_packing:
            raise ValueError("group_by_length and do_packing cannot be used together.")

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.do_packing:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None

            logger.info("packing sampler를 사용합니다.")
            return PackingSampler(
                dataset=self.train_dataset,
                lengths=lengths,
                max_seq_len=self.args.audio_max_seq,
                max_seq_per_pack=self.args.packing_max_elem,
                do_shuffle=self.args.packing_shuffle,
            )

        else:
            return RandomSampler(self.train_dataset)
