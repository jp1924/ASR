import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    Wav2Vec2Adapter,
    Wav2Vec2BaseModelOutput,
    Wav2Vec2Config,
    Wav2Vec2EncoderLayer,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2PreTrainedModel,
    _compute_mask_indices,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


class PackedWav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None and len(attention_mask.shape) == 2:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )
        elif attention_mask is not None and len(attention_mask.shape) == 4:
            # make sure padded tokens output 0
            # attention_mask는 bool tensor로 되어 있도록 하자, 1 True, 0 False로
            expand_attention_mask = attention_mask[:, 0, 0].unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                raise ValueError(
                    "야 여기 값 확인 안되었으니깐, attention_mask가 원본 wav2vec2하고 동일한지 확인하고 학습해라."
                )
                # extend attention_mask
                attention_mask = 1.0 - attention_mask
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

        raise ValueError("여기 positional encoding 있음. 이거 수정하고 학습 하셈!")
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PackedWav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        hidden_states,
        feat_split_idx=None,
        attention_mask=None,
        feat_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None and len(attention_mask.shape) == 2:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )
        elif feat_attention_mask is not None and len(feat_attention_mask.shape) == 4:
            # make sure padded tokens output 0
            # attention_mask는 bool tensor로 되어 있도록 하자, 1 True, 0 False로
            # expand_attention_mask = (
            #     torch.diagonal(attention_mask, dim1=2, dim2=3).transpose(2, 1).repeat(1, 1, hidden_states.shape[2])
            # )
            # hidden_states = hidden_states * expand_attention_mask.to(dtype=hidden_states.dtype)
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                feat_attention_mask = 1.0 - feat_attention_mask
                feat_attention_mask = feat_attention_mask * torch.finfo(hidden_states.dtype).min

            attention_mask = feat_attention_mask

        # if feat_split_idx:
        #     positional_hidden_states = torch.zeros(
        #         hidden_states.shape,
        #         device=hidden_states.device,
        #         dtype=hidden_states.dtype,
        #     )
        #     for idx, (feat_split, hidden_state) in enumerate(zip(feat_split_idx, hidden_states)):
        #         start_idx = 0
        #         for split_idx in feat_split.tolist():
        #             end_idx = start_idx + split_idx
        #             sample = hidden_state[start_idx:end_idx][None]
        #             sample = self.dropout(sample + self.pos_conv_embed(sample))

        #             positional_hidden_states[idx, start_idx:end_idx, :] = sample[0]
        #             start_idx += split_idx
        #     hidden_state = positional_hidden_states
        # else:
        #     position_embeddings = self.pos_conv_embed(hidden_states)
        #     hidden_states = hidden_states + position_embeddings
        #     hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PackedWav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = PackedWav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = PackedWav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        hidden_states=None,
        extract_features=None,
        pack_extract_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feat_split_idx: Optional[torch.Tensor] = None,
        feat_attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if pack_extract_features is not None and pack_attention_mask is None:
        #     raise ValueError("pack_extract_features는 있는데 attention_mask가 입력되지 않았음. 이거 입력하셈.")

        # if pack_extract_features is None:
        #     extract_features = self.feature_extractor(input_values)
        # else:
        #     extract_features = pack_extract_features

        # extract_features = extract_features.transpose(1, 2)

        # if pack_attention_mask is not None:
        #     attention_mask = pack_attention_mask
        # elif attention_mask is not None:
        #     # compute reduced attention_mask corresponding to feature vectors
        #     attention_mask = self._get_feature_vector_attention_mask(
        #         extract_features.shape[1], attention_mask, add_adapter=False
        #     )

        # hidden_states, extract_features = self.feature_projection(extract_features)
        # hidden_states = self._mask_hidden_states(
        #     hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # )

        encoder_outputs = self.encoder(
            hidden_states,
            feat_attention_mask=feat_attention_mask,
            feat_split_idx=feat_split_idx,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PackedWav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = PackedWav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Union[Optional[torch.Tensor], List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        feat_attention_mask: Optional[torch.Tensor] = None,
        split_idx: Optional[torch.Tensor] = None,
        feat_split_idx: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
        >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        >>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = input_values.shape
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
        >>> mask_time_indices = _compute_mask_indices(
        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        ... )
        >>> sampled_negative_indices = _sample_negative_indices(
        ...     features_shape=(batch_size, sequence_length),
        ...     num_negatives=model.config.num_negatives,
        ...     mask_time_indices=mask_time_indices,
        ... )
        >>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
        >>> sampled_negative_indices = torch.tensor(
        ...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        ... )

        >>> with torch.no_grad():
        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        >>> # show that cosine similarity is much higher than random
        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
        tensor(True)

        >>> # for contrastive loss training model should be put into train mode
        >>> model = model.train()
        >>> loss = model(
        ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
        ... ).loss
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        max_seq_len = 512

        pack_extract_features = None
        if isinstance(input_values, list):
            extract_features_ls = list()
            hidden_states_ls = list()
            for batch_idx, batch in enumerate(input_values):
                start_idx = 0
                extract_features = torch.zeros(
                    (1, max_seq_len, self.config.conv_dim[-1]), device=self.device, dtype=self.dtype
                )
                hidden_states = torch.zeros(
                    (1, max_seq_len, self.config.hidden_size), device=self.device, dtype=self.dtype
                )
                for input_value in batch:
                    input_value = input_value if len(input_value.shape) == 2 else input_value[None]
                    input_value = input_value.to(self.dtype).to(self.device)
                    extract_feature = self.wav2vec2.feature_extractor(input_value)
                    extract_feature = extract_feature.transpose(1, 2)
                    feat_len = extract_feature.shape[1]

                    hidden_state, extract_feature = self.wav2vec2.feature_projection(extract_feature)
                    hidden_state = self.wav2vec2._mask_hidden_states(
                        hidden_state,
                        mask_time_indices=mask_time_indices[batch_idx, start_idx : start_idx + feat_len][None],
                        attention_mask=torch.ones((1, feat_len), device=hidden_state.device, dtype=hidden_state.dtype),
                    )

                    position_embeddings = self.wav2vec2.encoder.pos_conv_embed(hidden_state)
                    hidden_state = hidden_state + position_embeddings
                    hidden_state = self.wav2vec2.encoder.dropout(hidden_state)

                    hidden_states[0, start_idx : start_idx + feat_len, :] = hidden_state
                    extract_features[0, start_idx : start_idx + feat_len, :] = extract_feature
                    start_idx += feat_len

                extract_features_ls.append(extract_features)
                hidden_states_ls.append(hidden_states)
            pack_extract_features = torch.concat(extract_features_ls)
            pack_hidden_states = torch.concat(hidden_states_ls)

        outputs = self.wav2vec2(
            input_values,
            hidden_states=pack_hidden_states,
            extract_features=pack_extract_features,
            attention_mask=attention_mask,
            pack_extract_features=pack_extract_features,
            feat_split_idx=feat_split_idx,
            feat_attention_mask=feat_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
        quantized_features = torch.zeros(
            (extract_features.shape[0], max_seq_len, self.config.codevector_dim),
            device=self.device,
            dtype=self.dtype,
        )
        codevector_perplexity = torch.tensor(0, device=self.device, dtype=self.dtype)
        for batch_idx, feat_idx_ls in enumerate(feat_split_idx):
            start_idx = 0

            for feat_idx in feat_idx_ls:
                feat_idx = int(feat_idx)
                end_idx = start_idx + feat_idx

                quantized_features[batch_idx, start_idx:end_idx], perplexity = self.quantizer(
                    extract_features[batch_idx, start_idx:end_idx][None],
                    mask_time_indices=mask_time_indices[batch_idx, start_idx:end_idx][None],
                )
                codevector_perplexity += perplexity
                start_idx += feat_idx

        codevector_perplexity = codevector_perplexity / extract_features.shape[0]

        ########################################################################################
        ########################################################################################
        ########################################################################################

        # quantized_features, codevector_perplexity = self.quantizer(
        #     extract_features, mask_time_indices=mask_time_indices
        # )

        quantized_features = quantized_features.to(self.project_q.weight.dtype)
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        #######################################################################################
        #######################################################################################
        #######################################################################################

        # total_loss_ls = list()
        # total_con_loss_ls = list()
        # total_div_loss_ls = list()
        # for idx, feat_idx_ls in enumerate(feat_split_idx):
        #     start_idx = 0
        #     con_loss_ls = list()
        #     div_loss_ls = list()
        #     loss_ls = list()
        #     for feat_idx in feat_idx_ls:
        #         end_idx = start_idx + int(feat_idx)

        #         sample_mask_time_indices = mask_time_indices[idx, start_idx:end_idx][None]
        #         sm_sampled_negative_indices = sampled_negative_indices[idx, start_idx:end_idx][None]
        #         quantized_features, codevector_perplexity = self.quantizer(
        #             extract_features[idx, start_idx:end_idx][None],
        #             mask_time_indices=sample_mask_time_indices,
        #         )

        #         quantized_features = quantized_features.to(self.project_q.weight.dtype)
        #         quantized_features = self.project_q(quantized_features)

        #         loss = contrastive_loss = diversity_loss = None
        #         if sampled_negative_indices is not None:
        #             batch_size, sequence_length, hidden_size = quantized_features.shape

        #             # for training, we sample negatives
        #             # 3. sample K negatives (distractors) quantized states for contrastive loss
        #             # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
        #             # sample negative quantized vectors BTC => (BxT)C
        #             negative_quantized_features = quantized_features.view(-1, hidden_size)[
        #                 sm_sampled_negative_indices.long().view(-1)
        #             ]
        #             negative_quantized_features = negative_quantized_features.view(
        #                 batch_size, sequence_length, -1, hidden_size
        #             ).permute(2, 0, 1, 3)

        #             # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
        #             # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
        #             logits = self.compute_contrastive_logits(
        #                 quantized_features[None, :],
        #                 negative_quantized_features,
        #                 transformer_features[idx, start_idx:end_idx][None],
        #                 self.config.contrastive_logits_temperature,
        #             )

        #             # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
        #             # its cosine similarity will be masked
        #             neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

        #             if neg_is_pos.any():
        #                 logits[1:][neg_is_pos] = float("-inf")

        #             # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
        #             # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
        #             logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        #             target = ((1 - sample_mask_time_indices.long()) * -100).transpose(0, 1).flatten()

        #             contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
        #             # 7. compute diversity loss: \mathbf{L}_d
        #             num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
        #             diversity_loss = (
        #                 (num_codevectors - codevector_perplexity) / num_codevectors
        #             ) * sample_mask_time_indices.sum()

        #             # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
        #             loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        #         start_idx += int(feat_idx)

        #         con_loss_ls.append(contrastive_loss)
        #         div_loss_ls.append(diversity_loss)
        #         loss_ls.append(loss)

        #     total_loss_ls.append(torch.stack(loss_ls).sum())
        #     total_con_loss_ls.append(torch.stack(con_loss_ls).sum())
        #     total_div_loss_ls.append(torch.stack(div_loss_ls).sum())

        # loss = torch.stack(total_loss_ls).sum()
        # contrastive_loss = torch.stack(total_con_loss_ls).sum()
        # diversity_loss = torch.stack(total_div_loss_ls).sum()

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )

    def floating_point_ops(self, *args, **kwargs):
        return 0
