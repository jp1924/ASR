import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

# from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    Wav2Vec2Adapter,
    Wav2Vec2AttnAdapterLayer,
    Wav2Vec2BaseModelOutput,
    Wav2Vec2Config,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
    Wav2Vec2FeedForward,
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2PreTrainedModel,
    _compute_mask_indices,
    _sample_negative_indices,
    logger,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    layer_head_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, 1)
    value_states = repeat_kv(value, 1)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if layer_head_mask is not None:
        logger.error("이거 아직 구현하기 전임.")
        if layer_head_mask.size() != (module.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(module.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, module.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@dataclass
class PackWav2Vec2ForPreTrainingOutput(Wav2Vec2ForPreTrainingOutput):
    """
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.

    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """

    mask_time_indices: Optional[List[torch.BoolTensor]] = None


class Wav2Vec2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Wav2Vec2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.is_decoder = False

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            position_ids=position_ids,
            layer_head_mask=layer_head_mask,
            dropout=0.0 if not self.training else self.config.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.layer_norm_eps = config.layer_norm_eps

        self.attention = Wav2Vec2Attention(config=config, layer_idx=layer_idx)

        self.dropout = nn.Dropout(self.hidden_dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            position_ids=position_ids,
            layer_head_mask=layer_head_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config: Wav2Vec2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.layer_norm_eps = config.layer_norm_eps

        self.attention = Wav2Vec2Attention(config=config, layer_idx=layer_idx)

        self.dropout = nn.Dropout(self.hidden_dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)

        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            position_ids=position_ids,
            layer_head_mask=layer_head_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
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
                        attention_mask=attention_mask,
                        layer_head_mask=layer_head_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=layer_head_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
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


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor]], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states = hidden_states * expand_attention_mask.to(dtype=hidden_states.dtype)
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
                        attention_mask=attention_mask,
                        layer_head_mask=layer_head_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        layer_head_mask=layer_head_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
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
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

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
        input_values: Union[List[torch.Tensor], torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
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

        if isinstance(input_values, torch.Tensor):
            extract_features = self.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)

            if attention_mask is not None:
                # compute reduced attention_mask corresponding to feature vectors
                attention_mask = self._get_feature_vector_attention_mask(
                    extract_features.shape[1], attention_mask, add_adapter=False
                )

            hidden_states, extract_features = self.feature_projection(extract_features)
            hidden_states = self._mask_hidden_states(
                hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
            )
            mask_time_indices, sampled_negative_indices = None, None
        elif isinstance(input_values, list):
            hidden_states, extract_features = [], []
            mask_time_indices, sampled_negative_indices = [], []
            for input_value in input_values:
                extract_feature = self.feature_extractor(input_value[None])
                extract_feature = extract_feature.transpose(1, 2)
                mask_time = _compute_mask_indices(
                    extract_feature.shape[:-1],
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    min_masks=self.config.mask_time_min_masks,
                )
                negative_indices = _sample_negative_indices(
                    mask_time.shape,
                    self.config.num_negatives,
                    mask_time_indices=mask_time,
                )
                mask_time = torch.tensor(mask_time)
                negative_indices = torch.tensor(negative_indices)
                hidden_state, extract_feature = self.feature_projection(extract_feature)
                hidden_state = self._mask_hidden_states(hidden_state, mask_time)

                # positional encoding
                position_embeddings = self.encoder.pos_conv_embed(hidden_state)
                hidden_state = hidden_state + position_embeddings
                hidden_state = self.encoder.dropout(hidden_state)

                hidden_states.append(hidden_state)
                extract_features.append(extract_feature)

                mask_time_indices.append(mask_time)
                sampled_negative_indices.append(negative_indices)

            hidden_states = torch.cat(hidden_states, dim=1)
            extract_features = torch.cat(extract_features, dim=1)

        encoder_outputs = self.encoder(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if isinstance(input_values, list):
            if not return_dict:
                return (
                    (hidden_states, extract_features) + encoder_outputs[1:],
                    mask_time_indices,
                    sampled_negative_indices,
                )

            return (
                Wav2Vec2BaseModelOutput(
                    last_hidden_state=hidden_states,
                    extract_features=extract_features,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                ),
                mask_time_indices,
                sampled_negative_indices,
            )
        else:
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
        position_ids: Optional[torch.Tensor] = None,
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

        # import copy

        # input_values_ls = [
        #     input_value[mask.bool()] for input_value, mask in zip(copy.deepcopy(input_values), attention_mask)
        # ]
        # position_ids = torch.concat(
        #     [torch.arange(self._get_feat_extract_output_lengths(x.shape[0])) for x in input_values_ls]
        # ).to(self.device)

        if isinstance(input_values, list):
            outputs, pack_mask_time_indices, pack_sampled_negative_indices = self.wav2vec2(
                input_values,
                # input_values_ls,
                position_ids=position_ids,
                # attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                # mask_time_indices=mask_time_indices,
                return_dict=return_dict,
            )
        else:
            outputs = self.wav2vec2(
                input_values,
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

        if attention_mask is not None and position_ids is None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        if isinstance(input_values, list):
            split_idx_ls = [x.shape[1] for x in pack_mask_time_indices]
            split_transformer = transformer_features.split(split_idx_ls, dim=1)
            quantized_features_ls = list()
            loss_ls, diversity_loss_ls, contrastive_loss_ls, codevector_perplexity_ls = list(), list(), list(), list()
            logits_ls, targets_ls, codevector_perplexity_ls = list(), list(), list()
            import time

            start_time = time.time()
            for idx, split_features in enumerate(extract_features.split(split_idx_ls, dim=1)):
                mask_time = pack_mask_time_indices[idx].to(self.device)
                quantized_feature, codevector_perplexity = self.quantizer(
                    split_features,
                    mask_time_indices=mask_time,
                )
                quantized_feature = quantized_feature.to(self.project_q.weight.dtype)
                quantized_feature = self.project_q(quantized_feature)

                codevector_perplexity_ls.append(codevector_perplexity)

                loss, contrastive_loss, diversity_loss = None, None, None
                if pack_sampled_negative_indices is not None:
                    batch_size, sequence_length, hidden_size = quantized_feature.shape

                    # for training, we sample negatives
                    # 3. sample K negatives (distractors) quantized states for contrastive loss
                    # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
                    # sample negative quantized vectors BTC => (BxT)C
                    negative_quantized_features = quantized_feature.view(-1, hidden_size)[
                        pack_sampled_negative_indices[idx].long().view(-1)
                    ]
                    negative_quantized_features = negative_quantized_features.view(
                        batch_size, sequence_length, -1, hidden_size
                    ).permute(2, 0, 1, 3)

                    # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
                    # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
                    logits = self.compute_contrastive_logits(
                        quantized_feature[None, :],
                        negative_quantized_features,
                        split_transformer[idx],
                        self.config.contrastive_logits_temperature,
                    )

                    # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
                    # its cosine similarity will be masked
                    neg_is_pos = (quantized_feature == negative_quantized_features).all(-1)

                    if neg_is_pos.any():
                        logits[1:][neg_is_pos] = float("-inf")

                    # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
                    # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
                    logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
                    target = ((1 - mask_time.long()) * -100).transpose(0, 1).flatten()

                    logits_ls.append(logits)
                    targets_ls.append(target)
                    codevector_perplexity_ls.append(codevector_perplexity)

                    # contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
                    # # 7. compute diversity loss: \mathbf{L}_d
                    # num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
                    # diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time.sum()

                    # # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
                    # loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

                    # diversity_loss_ls.append(diversity_loss)
                    # loss_ls.append(loss)
                    # contrastive_loss_ls.append(contrastive_loss)
                    # codevector_perplexity_ls.append(codevector_perplexity)

                quantized_features_ls.append(quantized_feature)
            end_time = time.time()
            final_logits = torch.concat(logits_ls).float()
            final_targets = torch.concat(targets_ls)

            contrastive_loss = nn.functional.cross_entropy(final_logits, final_targets, reduction="sum")
            codevector_perplexity = sum(
                [codevector_perplexity_ls[idx] / x.sum() for idx, x in enumerate(pack_mask_time_indices)]
            )

            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * sum(
                [x.sum() for x in pack_mask_time_indices]
            )

            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

            quantized_features = torch.concat(quantized_features_ls, dim=1)
        else:
            start_time = time.time()
            quantized_features, codevector_perplexity = self.quantizer(
                extract_features,
                mask_time_indices=mask_time_indices,
            )

            quantized_features = quantized_features.to(self.project_q.weight.dtype)
            quantized_features = self.project_q(quantized_features)

            loss, contrastive_loss, diversity_loss = None, None, None
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
                diversity_loss = (
                    (num_codevectors - codevector_perplexity) / num_codevectors
                ) * mask_time_indices.sum()

                # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
                loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss
            end_time = time.time()

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            # no pack, bsz: 10
            # print(loss, contrastive_loss, diversity_loss, codevector_perplexity)
            # tensor(237.2548, device='cuda:1', grad_fn=<AddBackward0>) tensor(232.3723, device='cuda:1', grad_fn=<NllLossBackward0>) tensor(48.8255, device='cuda:1', grad_fn=<MulBackward0>) tensor(15.0337, device='cuda:1', grad_fn=<SumBackward0>)

            # print(contrastive_loss)
            # tensor(232.3723, device='cuda:1', grad_fn=<NllLossBackward0>)

        return PackWav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
            mask_time_indices=pack_mask_time_indices if isinstance(input_values, list) else None,
        )

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        # if not hasattr(self, "warnings_issued"):
        #     self.warnings_issued = {}
        # if self.main_input_name in input_dict and isinstance(input_dict[self.main_input_name], list):
        #     return sum(
        #         [
        #             self._get_feat_extract_output_lengths(x.shape[0])
        #             for y in input_dict[self.main_input_name]
        #             for x in y
        #         ]
        #     )  # noqa
        # elif self.main_input_name in input_dict and isinstance(input_dict[self.main_input_name], torch.Tensor):
        #     return input_dict[self.main_input_name].numel()
        # elif "estimate_tokens" not in self.warnings_issued:
        #     logger.warning(
        #         "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
        #     )
        #     self.warnings_issued["estimate_tokens"] = True
        return 0
