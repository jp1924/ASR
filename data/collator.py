import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from transformers import PretrainedConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining, Wav2Vec2Processor
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)


def get_feat_extract_output_lengths(
    input_lengths: int,
    config,
    add_adapter: Optional[bool] = None,
) -> int:
    """
    Computes the output length of the convolutional layers
    """

    add_adapter = config.add_adapter if add_adapter is None else add_adapter

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        return math.floor((input_length - kernel_size) / stride) + 1

    for kernel_size, stride in zip(config.conv_kernel, config.conv_stride):
        input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

    if add_adapter:
        for _ in range(config.num_adapter_layers):
            input_lengths = _conv_out_length(input_lengths, 1, config.adapter_stride)

    return input_lengths


@dataclass
class DataPackingCollatorForWav2Vec2Pretraining(DataCollatorMixin):
    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    pack_max_seq: int = 512
    mask_time_prob: float = 0.65
    mask_time_length: int = 10
    mask_time_min_masks: int = 0
    num_negatives: int = 100
    return_tensors: str = "pt"

    def torch_call(self, features):
        if "feat_split_idx" in features[0]:
            input_values_ls = list()
            feat_split_indices_ls = list()
            mask_time_indices = np.zeros((len(features), self.pack_max_seq))
            expand_attention_mask = np.zeros((len(features), 1, self.pack_max_seq, self.pack_max_seq))
            for batch_idx, feature in enumerate(features):
                input_values_ls.append(feature["input_values"])

                start_idx = 0
                for feat_len in feature["feat_split_idx"]:
                    feat_len = int(feat_len)
                    end_idx = start_idx + feat_len
                    mask_time_indices[batch_idx, start_idx:end_idx] = _compute_mask_indices(
                        (1, feat_len),
                        self.mask_time_prob,
                        self.mask_time_length,
                        min_masks=self.mask_time_min_masks,
                    )
                    expand_attention_mask[batch_idx, 0, start_idx:end_idx, start_idx:end_idx] = 1
                    start_idx += feat_len

                feat_split_indices_ls.append(feature["feat_split_idx"])

            sampled_negative_indices = _sample_negative_indices(
                mask_time_indices.shape,
                self.num_negatives,
                mask_time_indices=mask_time_indices,
            )

            batch = dict()
            batch["input_values"] = input_values_ls
            batch["feat_split_idx"] = feat_split_indices_ls
            batch["attention_mask"] = torch.tensor(expand_attention_mask)
            batch["mask_time_indices"] = torch.tensor(mask_time_indices)
            batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices)
        else:
            features = [{"input_values": x["input_values"]} for x in features]

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
            # evaluate에선 sub_attention_mask 사용을 안함. 그리고 없애는 처리도 하지 않기 때문에 애러가 발생함.
            if hasattr(batch, "attention_mask") is not None and self.model.training:
                # compute real output lengths according to convolution formula
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

            batch = batch.to(self.model.dtype)

        return batch


@dataclass
class DataCollatorForWav2Vec2Pretraining(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    mask_time_min_masks: int = 0
    num_negatives: int = 100
    return_tensors: str = "pt"

    def torch_call(self, features):
        # features가 2차원 리스트로 들어올 떄 feature_extractor에서 padding을 진행하지 못함. 따라서 이걸 1차원 리스트로 변경 함.
        features = [{"input_values": x["input_values"]} for x in features]

        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=self.return_tensors,
        )

        device = batch.input_values.device
        batch_size, sample_size = batch.input_values.shape
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(sample_size).item()

        # make sure that no loss is computed on padded inputs
        # evaluate에선 sub_attention_mask 사용을 안함. 그리고 없애는 처리도 하지 않기 때문에 애러가 발생함.
        if hasattr(batch, "attention_mask") is not None and self.model.training:
            # compute real output lengths according to convolution formula
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

        batch = batch.to(self.model.dtype)

        return batch


@dataclass
class DataPackingCollatorCTCWithPadding(DataCollatorMixin):
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    config: PretrainedConfig = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    return_tensors: str = "pt"
    packing_max_seq_len: int = 512

    def torch_call(self, features):
        if "target_lengths" in features[0]:
            input_values = [feature["input_values"] for feature in features]

            # attention_mask = np.zeros((len(input_values), 1, self.packing_max_seq_len, self.packing_max_seq_len))

            # for batch_idx, input_value in enumerate(input_values):
            #     start_idx = 0
            #     for single_sample in input_value:
            #         feat_length = get_feat_extract_output_lengths(len(single_sample), self.config)
            #         end_idx = start_idx + feat_length
            #         attention_mask[batch_idx, 0, start_idx:end_idx, start_idx:end_idx] = 1

            #         start_idx += feat_length

            target_lengths = [feature["target_lengths"] for feature in features]
            target_lengths = torch.concat(target_lengths)

            labels = [feature["labels"] for feature in features]
            labels = torch.concat(labels, dim=-1)

            attention_mask = [feature["attention_mask"] for feature in features]
            attention_mask = np.concatenate(attention_mask, axis=0)
            attention_mask = torch.tensor(attention_mask)

            # test_arr = torch.tensor(attention_mask.sum(-1)[-1][0])
            # _, idx = np.unique(test_arr, return_index=True)
            # unique_ordered = test_arr[np.sort(idx)]

            # self.processor.tokenizer.batch_decode(torch.split(labels, target_lengths.tolist()))

            batch = {
                "input_values": input_values,
                "target_lengths": target_lengths,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        else:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features=input_features,
                labels=label_features,
                padding=self.padding,
                return_attention_mask=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )

            # replace padding with -100 to ignore loss correctly
            batch.labels[batch.labels == 0] = -100
            if hasattr(batch, "attention_mask"):
                batch["attention_mask"] = batch.attention_mask.to(torch.long)
        return batch


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features=input_features,
            labels=label_features,
            padding=self.padding,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # replace padding with -100 to ignore loss correctly
        batch.labels[batch.labels == 0] = -100
        if hasattr(batch, "attention_mask"):
            batch["attention_mask"] = batch.attention_mask.to(torch.long)

        return batch
