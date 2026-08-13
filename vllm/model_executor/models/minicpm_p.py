# SPDX-License-Identifier: Apache-2.0
"""Dense MiniCPM decoder with bounded DMI observation hooks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import islice
from numbers import Real

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
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

from .minicpm import (
    MiniCPMAttention,
    MiniCPMDecoderLayer,
    MiniCPMForCausalLM,
    MiniCPMMLP,
    MiniCPMModel,
)
from .utils import PPMissingLayer


def _positive_real(config, name: str) -> float:
    value = getattr(config, name, None)
    if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
        raise NotImplementedError(
            f"DMI MiniCPM support requires a positive numeric {name}"
        )
    return float(value)


def _require_supported_minicpm_config(config) -> None:
    """Fail closed outside the audited dense MiniCPM 4.1 contract."""

    if getattr(config, "model_type", None) != "minicpm":
        raise NotImplementedError("DMI MiniCPM support requires model_type=minicpm")
    if getattr(config, "hidden_act", None) != "silu":
        raise NotImplementedError("DMI MiniCPM support requires hidden_act=silu")
    if getattr(config, "num_experts", 0) != 0:
        raise NotImplementedError("DMI MiniCPM support excludes the MoE branch")
    if getattr(config, "sparse_config", None) is not None:
        raise NotImplementedError(
            "DMI MiniCPM support excludes sparse-attention configurations"
        )
    for name in ("scale_emb", "scale_depth", "dim_model_base"):
        _positive_real(config, name)
    if not isinstance(getattr(config, "tie_word_embeddings", None), bool):
        raise NotImplementedError(
            "DMI MiniCPM support requires explicit tied/untied embeddings"
        )

    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if (
        not isinstance(hidden_size, int)
        or isinstance(hidden_size, bool)
        or hidden_size <= 0
        or not isinstance(num_heads, int)
        or isinstance(num_heads, bool)
        or num_heads <= 0
        or hidden_size % num_heads
        or not isinstance(num_kv_heads, int)
        or isinstance(num_kv_heads, bool)
        or num_kv_heads <= 0
    ):
        raise NotImplementedError(
            "DMI MiniCPM support requires a valid explicit attention head layout"
        )

    if getattr(config, "max_position_embeddings", None) != 65_536:
        raise NotImplementedError(
            "DMI MiniCPM support requires max_position_embeddings=65536"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping):
        raise NotImplementedError(
            "DMI MiniCPM support requires explicit rope_parameters"
        )
    if rope.get("rope_type") != "longrope" or rope.get("rope_theta") != 10_000:
        raise NotImplementedError(
            "DMI MiniCPM support requires LongRoPE at theta 10000"
        )
    if rope.get("original_max_position_embeddings") != 65_536:
        raise NotImplementedError(
            "DMI MiniCPM support requires LongRoPE original context 65536"
        )
    expected_factors = hidden_size // num_heads // 2
    short = rope.get("short_factor")
    long = rope.get("long_factor")
    if (
        not isinstance(short, Sequence)
        or isinstance(short, (str, bytes))
        or not isinstance(long, Sequence)
        or isinstance(long, (str, bytes))
        or len(short) != expected_factors
        or len(long) != expected_factors
        or list(short) != list(long)
        or any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or value <= 0
            for value in (*short, *long)
        )
    ):
        raise NotImplementedError(
            "DMI MiniCPM support requires equal positive LongRoPE factor vectors "
            "matching half the attention head dimension"
        )


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


class MiniCPMPMLP(MiniCPMMLP):
    """Dense MiniCPM MLP with its post-SiLU boundary exposed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class MiniCPMPAttention(MiniCPMAttention):
    """MiniCPM attention with pre-RoPE Q/K/V and pre-o-proj Z hooks."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not any(
            hook.enabled
            for hook in (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        ):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.hook_q.enabled:
            self.hook_q(q.unflatten(-1, (self.num_heads, self.head_dim)))
        if self.hook_k.enabled:
            self.hook_k(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        if self.hook_v.enabled:
            self.hook_v(v.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class MiniCPMPDecoderLayer(MiniCPMDecoderLayer):
    """One dense MiniCPM block preserving depth-scaled residuals."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        layer_hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in layer_hooks):
            return super().forward(positions, hidden_states, residual)

        residual = hidden_states
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        hidden_states = self.input_layernorm(hidden_states)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
        residual_multiplier = self.config.scale_depth / math.sqrt(
            self.config.num_hidden_layers
        )
        hidden_states = residual + hidden_states * residual_multiplier
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        hidden_states = residual + hidden_states * residual_multiplier
        return hidden_states, None


def _instrument_minicpm_model(model: MiniCPMModel) -> "MiniCPMPModel":
    """Attach hooks to the exact module tree created by upstream MiniCPM."""

    _require_supported_minicpm_config(model.config)
    model.__class__ = MiniCPMPModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = MiniCPMPDecoderLayer
        layer.self_attn.__class__ = MiniCPMPAttention
        layer.mlp.__class__ = MiniCPMPMLP
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
        _add_hook_points(layer.self_attn, ("q", "k", "v", "z"))
        _add_hook_points(layer.mlp, ("post",))
    return model


class MiniCPMPModel(MiniCPMModel):
    """Concrete MiniCPM backbone with scaled-embedding and final hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_minicpm_model(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if not any(
            hook.enabled
            for hook in (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        ):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states,
                idx + 1,
                hidden_states,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states)
        hidden_states = self.norm(hidden_states)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)

        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class MiniCPMPForCausalLM(MiniCPMForCausalLM):
    """Dense MiniCPM causal LM with a truthful DMI hook manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_minicpm_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def _init_model(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> MiniCPMPModel:
        return MiniCPMPModel(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.mlp

        def hook(module, name: str):
            return None if module is None else getattr(module, f"hook_{name}")

        def spec(hook_type: int, module, name: str) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(module, name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            )

        return [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
            spec(HOOK_TYPE_Q, attention, "q"),
            spec(HOOK_TYPE_K, attention, "k"),
            spec(HOOK_TYPE_V, attention, "v"),
            spec(HOOK_TYPE_Z, attention, "z"),
            spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
            spec(HOOK_TYPE_LN2, layer, "ln2"),
            spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
            spec(HOOK_TYPE_MLP_POST, mlp, "post"),
            spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"),
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
        for layer_no in range(self.config.num_hidden_layers):
            layer = None
            if not model_wide and model.start_layer <= layer_no < model.end_layer:
                candidate = model.layers[layer_no]
                if not isinstance(candidate, PPMissingLayer):
                    layer = candidate
            specs.extend(self._layer_hook_specs(layer_no, layer))
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
    "MiniCPMPAttention",
    "MiniCPMPDecoderLayer",
    "MiniCPMPForCausalLM",
    "MiniCPMPMLP",
    "MiniCPMPModel",
    "_instrument_minicpm_model",
    "_require_supported_minicpm_config",
]
