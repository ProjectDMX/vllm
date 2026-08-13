# SPDX-License-Identifier: Apache-2.0
"""LFM2 attention/short-convolution hybrid with bounded DMI hooks."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_CONV_IN,
    HOOK_TYPE_CONV_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .lfm2 import (
    Lfm2Attention,
    Lfm2AttentionDecoderLayer,
    Lfm2ForCausalLM,
    Lfm2MLP,
    Lfm2Model,
    Lfm2ShortConvDecoderLayer,
)
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def _make_mlp(config, quant_config, prefix: str, mlp_type):
    return mlp_type(
        dim=config.block_dim,
        intermediate_size=config.intermediate_size,
        multiple_of=config.block_multiple_of,
        auto_adjust_ff_dim=config.block_auto_adjust_ff_dim,
        ffn_dim_multiplier=config.block_ffn_dim_multiplier,
        quant_config=quant_config,
        prefix=prefix,
    )


class Lfm2PMLP(Lfm2MLP):
    """LFM2 MLP with a post-activation hook."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        gate_up, _ = self.w13(x)
        x = self.act_fn(gate_up)
        if self.hook_post.enabled:
            self.hook_post(x)
        x, _ = self.w2(x)
        return x


class Lfm2PAttention(Lfm2Attention):
    """LFM2 QK-normalized attention with canonical Q/K/V/Z hooks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            self.hook_q.enabled
            or self.hook_k.enabled
            or self.hook_v.enabled
            or self.hook_z.enabled
        ):
            return super().forward(positions, hidden_states)
        n_tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = q.view(n_tokens, self.num_heads, self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        if self.hook_q.enabled:
            self.hook_q(q)
        if self.hook_k.enabled:
            self.hook_k(k)
        if self.hook_v.enabled:
            self.hook_v(
                v.view(n_tokens, self.num_kv_heads, self.head_dim)
            )
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(n_tokens, self.num_heads * self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads * self.head_dim)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.out_proj(attn_output)
        return output


class Lfm2PAttentionDecoderLayer(Lfm2AttentionDecoderLayer):
    """One full-attention LFM2 layer with canonical residual hooks."""

    def __init__(
        self,
        config,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        *,
        attention_type=Lfm2PAttention,
        mlp_type=Lfm2PMLP,
    ) -> None:
        nn.Module.__init__(self)
        self.prefix = prefix
        self.config = config
        self.layer_idx = layer_idx
        self.self_attn = attention_type(
            config=config,
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 8192
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.feed_forward = _make_mlp(
            config,
            quant_config,
            f"{prefix}.feed_forward",
            mlp_type,
        )
        self.operator_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.hook_resid_pre = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self.hook_resid_pre.enabled
            or self.hook_ln1.enabled
            or self.hook_attn_out.enabled
            or self.hook_resid_mid.enabled
            or self.hook_ln2.enabled
            or self.hook_mlp_in.enabled
            or self.hook_mlp_out.enabled
        ):
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **kwargs,
            )
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(
                hidden_states, residual
            )
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return hidden_states, residual


class Lfm2PShortConvDecoderLayer(Lfm2ShortConvDecoderLayer):
    """One stateful short-convolution LFM2 layer with typed boundaries."""

    def __init__(
        self,
        config,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        *,
        mlp_type=Lfm2PMLP,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.short_conv = ShortConv(
            config=config,
            dim=config.conv_dim,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.conv",
        )
        self.feed_forward = _make_mlp(
            config,
            quant_config,
            f"{prefix}.feed_forward",
            mlp_type,
        )
        self.operator_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.hook_resid_pre = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_conv_in = HookPoint()
        self.hook_conv_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self.hook_resid_pre.enabled
            or self.hook_ln1.enabled
            or self.hook_conv_in.enabled
            or self.hook_conv_out.enabled
            or self.hook_resid_mid.enabled
            or self.hook_ln2.enabled
            or self.hook_mlp_in.enabled
            or self.hook_mlp_out.enabled
        ):
            return super().forward(
                hidden_states=hidden_states,
                residual=residual,
                **kwargs,
            )
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(
                hidden_states, residual
            )
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        if self.hook_conv_in.enabled:
            self.hook_conv_in(hidden_states)
        output = torch.empty_like(hidden_states)
        self.short_conv(hidden_states, output)
        if self.hook_conv_out.enabled:
            self.hook_conv_out(output)
        hidden_states, residual = self.ffn_norm(output, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {-1: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    }
)
class Lfm2PModel(Lfm2Model):
    """LFM2 backbone selecting a truthful hook manifest per layer kind."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        attention_layer_type=Lfm2PAttentionDecoderLayer,
        conv_layer_type=Lfm2PShortConvDecoderLayer,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str) -> nn.Module:
            layer_idx = extract_layer_index(prefix)
            layer_type = (
                attention_layer_type
                if config.layer_types[layer_idx] == "full_attention"
                else conv_layer_type
            )
            return layer_type(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(
                config.hidden_size, eps=config.norm_eps
            )
        else:
            self.embedding_norm = PPMissingLayer()
        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if not (
            self.hook_embed.enabled
            or self.hook_resid_final.enabled
            or self.hook_final_ln.enabled
        ):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(
            self.layers, self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, resid_final = self.embedding_norm(
            hidden_states, residual
        )
        if self.hook_resid_final.enabled:
            self.hook_resid_final(resid_final)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
        return hidden_states


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _instrument_upstream_lfm2_model(model: Lfm2Model) -> Lfm2PModel:
    """Attach DMI behavior to the exact module tree built by upstream."""

    model.__class__ = Lfm2PModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.feed_forward.__class__ = Lfm2PMLP
        _add_hook_points(layer.feed_forward, ("post",))
        if model.config.layer_types[layer_no] == "full_attention":
            layer.__class__ = Lfm2PAttentionDecoderLayer
            layer.self_attn.__class__ = Lfm2PAttention
            _add_hook_points(layer.self_attn, ("q", "k", "v", "z"))
            _add_hook_points(
                layer,
                (
                    "resid_pre",
                    "ln1",
                    "attn_out",
                    "resid_mid",
                    "ln2",
                    "mlp_in",
                    "mlp_out",
                ),
            )
        else:
            layer.__class__ = Lfm2PShortConvDecoderLayer
            _add_hook_points(
                layer,
                (
                    "resid_pre",
                    "ln1",
                    "conv_in",
                    "conv_out",
                    "resid_mid",
                    "ln2",
                    "mlp_in",
                    "mlp_out",
                ),
            )
    return model


class Lfm2PForCausalLM(Lfm2ForCausalLM):
    """LFM2 causal LM with layer-kind-specific DMI capabilities."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type=Lfm2PModel,
    ) -> None:
        if model_type is Lfm2PModel:
            # Keep parameter creation and registration order byte-for-byte
            # upstream; attach observational modules only after construction.
            Lfm2ForCausalLM.__init__(
                self,
                vllm_config=vllm_config,
                prefix=prefix,
            )
            self.model = _instrument_upstream_lfm2_model(self.model)
        else:
            # The compare oracle injects model/layer subclasses containing
            # independent `.copy_()` buffers in the same execution graph.
            config = vllm_config.model_config.hf_config
            quant_config = vllm_config.quant_config
            cache_config = vllm_config.cache_config
            if cache_config.mamba_cache_mode == "all":
                raise NotImplementedError(
                    "Lfm2 currently does not support 'all' prefix caching, "
                    "please use '--mamba-cache-mode=align' instead"
                )
            nn.Module.__init__(self)
            self.config = config
            self.model = model_type(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )
            if get_pp_group().is_last_rank:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens
                )
            else:
                self.lm_head = PPMissingLayer()
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors
            )
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().compute_logits(hidden_states)
        if self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _common_layer_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        mlp = None if layer is None else layer.feed_forward

        def hook(module, name: str):
            return None if module is None else getattr(module, name)

        return [
            HookSpec(HOOK_TYPE_RESID_PRE, hook(layer, "hook_resid_pre"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN1, hook(layer, "hook_ln1"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_RESID_MID, hook(layer, "hook_resid_mid"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN2, hook(layer, "hook_ln2"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_IN, hook(layer, "hook_mlp_in"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_POST, hook(mlp, "hook_post"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_OUT, hook(layer, "hook_mlp_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
        ]

    @staticmethod
    def _operator_specs(
        layer_no: int,
        layer: nn.Module | None,
        is_attention: bool,
    ) -> list[HookSpec]:
        def hook(module, name: str):
            return None if module is None else getattr(module, name)

        if not is_attention:
            return [
                HookSpec(HOOK_TYPE_CONV_IN, hook(layer, "hook_conv_in"), layer_no=layer_no, dim0_is_actual_tokens=True),
                HookSpec(HOOK_TYPE_CONV_OUT, hook(layer, "hook_conv_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
            ]
        attention = None if layer is None else layer.self_attn
        return [
            HookSpec(HOOK_TYPE_Q, hook(attention, "hook_q"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_K, hook(attention, "hook_k"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_V, hook(attention, "hook_v"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_Z, hook(attention, "hook_z"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_ATTN_OUT, hook(layer, "hook_attn_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
        ]

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
        model = self.model
        specs = [
            HookSpec(
                HOOK_TYPE_TOKEN_IDS,
                None if model_wide else self.hook_token_ids,
                dtype=torch.int32,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_EMBED,
                None if model_wide else model.hook_embed,
                dim0_is_actual_tokens=True,
            ),
        ]
        for layer_no, layer_kind in enumerate(self.config.layer_types):
            layer = None
            if (
                not model_wide
                and model.start_layer <= layer_no < model.end_layer
            ):
                candidate = model.layers[layer_no]
                if not isinstance(candidate, PPMissingLayer):
                    layer = candidate
            common_specs = self._common_layer_specs(layer_no, layer)
            operator_specs = self._operator_specs(
                layer_no,
                layer,
                layer_kind == "full_attention",
            )
            # Hook IDs are assigned in manifest order and consumed in forward
            # firing order.  Both LFM2 layer kinds enter their operator after
            # resid_pre/ln1 and before resid_mid/ln2/MLP.
            specs.extend(common_specs[:2])
            specs.extend(operator_specs)
            specs.extend(common_specs[2:])
        specs.extend(
            [
                HookSpec(
                    HOOK_TYPE_RESID_FINAL,
                    None if model_wide else model.hook_resid_final,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LN,
                    None if model_wide else model.hook_final_ln,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LOGITS,
                    None if model_wide else self.hook_final_logits,
                ),
            ]
        )
        return specs


__all__ = [
    "Lfm2PAttention",
    "Lfm2PAttentionDecoderLayer",
    "Lfm2PForCausalLM",
    "Lfm2PMLP",
    "Lfm2PModel",
    "Lfm2PShortConvDecoderLayer",
    "_instrument_upstream_lfm2_model",
]
