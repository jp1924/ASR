import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _CTC_EXPECTED_LOSS,
    _CTC_EXPECTED_OUTPUT,
    _EXPECTED_OUTPUT_SHAPE,
    _HIDDEN_STATES_START_POSITION,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    WAV_2_VEC_2_START_DOCSTRING,
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
    logger,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
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
        hidden_states,
        attention_mask=None,
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

            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.dropout(hidden_states)
        elif attention_mask is not None and len(attention_mask.shape) == 4:
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        else:
            raise ValueError("dnffoiwfewonfimnvownfejdmjkfhneiwjddfhrnewjomfjn ")

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
        attention_mask=None,
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

            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.dropout(hidden_states)
        elif attention_mask is not None and len(attention_mask.shape) == 4:
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        else:
            raise ValueError("dnffoiwfewonfimnvownfejdmjkfhneiwjddfhrnewjomfjn ")

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
        hidden_states: Optional[torch.Tensor] = None,
        extract_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if hidden_states is not None and extract_features is None:
            raise ValueError("aslkdadkasndlasdnkasldn")
        elif hidden_states is None and extract_features is not None:
            raise ValueError("aslkdadkasndlasdnkasldn")
        elif hidden_states is None and extract_features is None and input_values is None:
            raise ValueError("aslkdadkasndlasdnkasldn")

        if input_values is not None:
            extract_features = self.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)

            if attention_mask is not None:
                # compute reduced attention_mask corresponding to feature vectors
                attention_mask = self._get_feature_vector_attention_mask(
                    extract_features.shape[1], attention_mask, add_adapter=False
                )

            hidden_states, extract_features = self.feature_projection(extract_features)
            hidden_states = self._mask_hidden_states(
                hidden_states,
                mask_time_indices=mask_time_indices,
                attention_mask=attention_mask,
            )

        encoder_outputs = self.encoder(
            hidden_states,
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

        logger.warning_once("max_seq_len가 하드코딩 되어 있음.")

        max_seq_len = 512

        if isinstance(input_values, list) and feat_split_idx is not None:
            batch_size = len(input_values)
            extract_features = torch.zeros(
                (batch_size, max_seq_len, self.config.conv_dim[-1]),
                device=self.device,
                dtype=self.dtype,
            )
            hidden_states = torch.zeros(
                (batch_size, max_seq_len, self.config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            for batch_idx, packing_ls in enumerate(input_values):
                start_idx = 0
                for input_value in packing_ls:
                    input_value = input_value[None].to(self.dtype).to(self.device)
                    extract_feature = self.wav2vec2.feature_extractor(input_value)
                    extract_feature = extract_feature.transpose(1, 2)

                    feat_len = extract_feature.shape[1]

                    hidden_state, extract_feature = self.wav2vec2.feature_projection(extract_feature)
                    hidden_state = self.wav2vec2._mask_hidden_states(
                        hidden_state,
                        mask_time_indices=mask_time_indices[batch_idx, start_idx : start_idx + feat_len][None],
                        # attention_mask=torch.ones((1, feat_len), device=hidden_state.device, dtype=hidden_state.dtype),
                    )

                    position_embeddings = self.wav2vec2.encoder.pos_conv_embed(hidden_state)
                    hidden_state = hidden_state + position_embeddings
                    hidden_state = self.wav2vec2.encoder.dropout(hidden_state)

                    hidden_states[batch_idx, start_idx : start_idx + feat_len] = hidden_state
                    extract_features[batch_idx, start_idx : start_idx + feat_len] = extract_feature

                    start_idx += feat_len

            input_values = None

        outputs = self.wav2vec2(
            input_values,
            hidden_states=hidden_states,
            extract_features=extract_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None and feat_split_idx is None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        if feat_split_idx is not None:
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
        else:
            quantized_features, codevector_perplexity = self.quantizer(
                extract_features,
                mask_time_indices=mask_time_indices,
            )

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


@add_start_docstrings(
    """Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
    """
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`Wav2Vec2ForCTC`] with adapters. Uses 'eng' by
            default.
    """,
)
class PackedWav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, target_lang: Optional[str] = None):
        super().__init__(config)

        self.wav2vec2 = PackedWav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2 so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Wav2Vec2 never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
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

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        target_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        logger.warning_once("max_seq_len가 하드코딩 되어 있음.")

        hidden_states = None
        extract_features = None
        max_seq_len = 512
        if isinstance(input_values, list) and target_lengths is not None:
            batch_size = len(input_values)
            extract_features = torch.zeros(
                (batch_size, max_seq_len, self.config.conv_dim[-1]),
                device=self.device,
                dtype=self.dtype,
            )
            hidden_states = torch.zeros(
                (batch_size, max_seq_len, self.config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            for batch_idx, packing_ls in enumerate(input_values):
                start_idx = 0
                for input_value in packing_ls:
                    input_value = input_value[None].to(self.dtype).to(self.device)
                    extract_feature = self.wav2vec2.feature_extractor(input_value)
                    extract_feature = extract_feature.transpose(1, 2)

                    feat_len = extract_feature.shape[1]

                    hidden_state, extract_feature = self.wav2vec2.feature_projection(extract_feature)
                    hidden_state = self.wav2vec2._mask_hidden_states(hidden_state)

                    position_embeddings = self.wav2vec2.encoder.pos_conv_embed(hidden_state)
                    hidden_state = hidden_state + position_embeddings
                    hidden_state = self.wav2vec2.encoder.dropout(hidden_state)

                    hidden_states[batch_idx, start_idx : start_idx + feat_len] = hidden_state
                    extract_features[batch_idx, start_idx : start_idx + feat_len] = extract_feature

                    start_idx += feat_len

            test_input_values = input_values
            input_values = None

        outputs = self.wav2vec2(
            input_values,
            hidden_states=hidden_states,
            extract_features=extract_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask

            if target_lengths is None:
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

                # assuming that padded tokens are filled with -100
                # when not being attended to
                labels_mask = labels >= 0
                target_lengths = labels_mask.sum(-1)
                flattened_targets = labels.masked_select(labels_mask)
                log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            else:
                if len(labels.shape) != 1:
                    raise ValueError()

                audio_max_seq_len = 512

                flattened_targets = labels
                padding_mask_ls = torch.diagonal(attention_mask, dim1=2, dim2=3)[:, 0].bool()
                logits_ls = logits
                spread_logits = torch.zeros(
                    (target_lengths.shape[0], audio_max_seq_len, self.config.vocab_size),
                    device=self.device,
                    dtype=self.dtype,
                )

                split_logits_ls = list()
                for batch_idx, (padding_mask, logits) in enumerate(zip(padding_mask_ls, logits_ls)):
                    # split_idx_ls = attention_mask[batch_idx, 0, padding_mask].to(torch.int32).sum(-1).cpu().numpy()
                    split_idx_ls = self.get_pack_feat_len_from_attention_mask(attention_mask[batch_idx])

                    if padding_mask.sum(-1).item() != sum(split_idx_ls):
                        # [self._get_feat_extract_output_lengths(x.shape[0]) for x in test_input_values[batch_idx]]
                        split_idx_ls[-1] += 1

                    # _, idx = np.unique(split_idx_ls, return_index=True)
                    # split_logits = torch.split(logits, split_idx_ls[np.sort(idx)].tolist())
                    split_logits = torch.split(logits[padding_mask], split_idx_ls)
                    split_logits_ls.extend(split_logits)

                for idx, x in enumerate(split_logits_ls):
                    seq_len, _ = x.shape
                    spread_logits[idx, :seq_len] = x

                log_probs = nn.functional.log_softmax(spread_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
                input_lengths = torch.tensor(
                    [x.shape[0] for x in split_logits_ls], device=self.device, dtype=torch.long
                )

            # ctc_loss doesn't support fp16

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def floating_point_ops(self, *args, **kwargs):
        return 0

    def get_pack_feat_len_from_attention_mask(self, attention_mask):
        # 차이값을 계산하여 시작과 끝 지점 찾기

        # attention_mask.shape: (1, width, height)
        diag_tensor = torch.diagonal(attention_mask[0], -1).cpu()
        diff = torch.diff(diag_tensor)  # 연속된 값의 차이를 계산
        # 시작 지점은 1로 변하는 지점 (diff == 1)
        # 끝 지점은 0으로 변하는 지점 (diff == -1)

        # 처음과 마지막 위치도 고려하기 위해 앞뒤로 패딩
        start_indices = torch.where(diff == 1)[0] + 1
        end_indices = torch.where(diff == -1)[0] + 1

        if diag_tensor[0] == 1:  # 시작이 1인 경우 시작 인덱스에 0 추가
            start_indices = torch.cat([torch.tensor([0], device=diag_tensor.device), start_indices])

        if diag_tensor[-1] == 1:  # 끝이 1인 경우 마지막 인덱스 추가
            end_indices = torch.cat([end_indices, torch.tensor([len(diag_tensor) - 1], device=diag_tensor.device)])

        # 각 구간의 길이 계산
        lengths = end_indices - start_indices + 1
        # lengths[-1] += 1

        return lengths.tolist()
