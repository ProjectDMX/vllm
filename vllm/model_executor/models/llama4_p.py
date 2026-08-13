# SPDX-License-Identifier: Apache-2.0
"""Llama 4 Scout language decoder with bounded DMI observation hooks."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice

import torch
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
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .llama4 import (
    Llama4Attention,
    Llama4DecoderLayer,
    Llama4ForCausalLM,
    Llama4Model,
    Llama4MoE,
)
from .utils import PPMissingLayer


def _require_supported_llama4_scout_text_config(
    config,
    parallel_config=None,
    quant_config=None,
) -> None:
    """Fail closed outside the audited Llama 4 Scout text decoder."""

    exact_fields = {
        "model_type": "llama4_text",
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "intermediate_size_mlp": 16384,
        "num_hidden_layers": 48,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_local_experts": 16,
        "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 1,
        "attention_chunk_size": 8192,
        "max_position_embeddings": 10_485_760,
        "vocab_size": 202_048,
    }
    for field, expected in exact_fields.items():
        if getattr(config, field, None) != expected:
            raise NotImplementedError(
                f"DMI Llama 4 Scout support requires {field}={expected}"
            )
    if getattr(config, "attention_bias", False):
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires bias-free attention"
        )
    if getattr(config, "tie_word_embeddings", None) is not False:
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires untied embeddings"
        )
    if getattr(config, "output_router_logits", False):
        raise NotImplementedError(
            "DMI Llama 4 Scout support excludes public router-logit outputs"
        )
    if getattr(config, "use_qk_norm", None) is not True:
        raise NotImplementedError("DMI Llama 4 Scout support requires Q/K norm")

    expected_no_rope = [1, 1, 1, 0] * 12
    if list(getattr(config, "no_rope_layers", ()) or ()) != expected_no_rope:
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires the audited 3-local/1-NoPE schedule"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping):
        rope = getattr(config, "rope_scaling", None)
    if (
        not isinstance(rope, Mapping)
        or rope.get("rope_type") != "llama3"
        or rope.get("factor") != 16
        or rope.get("original_max_position_embeddings") != 8192
        or rope.get("high_freq_factor") != 1
        or rope.get("low_freq_factor") != 1
    ):
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires the audited Llama-3 RoPE scaling"
        )
    rope_theta = rope.get("rope_theta", getattr(config, "rope_theta", None))
    if rope_theta != 500_000:
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires RoPE theta 500000"
        )
    if quant_config is not None or getattr(config, "quantization_config", None):
        raise NotImplementedError(
            "DMI Llama 4 Scout lite support is limited to the BF16 checkpoint"
        )

    if parallel_config is not None:
        if (
            getattr(parallel_config, "tensor_parallel_size", 1) != 4
            or getattr(parallel_config, "pipeline_parallel_size", 1) != 1
            or getattr(parallel_config, "data_parallel_size", 1) != 1
            or getattr(parallel_config, "enable_expert_parallel", False)
        ):
            raise NotImplementedError(
                "DMI Llama 4 Scout lite support requires TP4/PP1/DP1 without EP"
            )
        if getattr(parallel_config, "use_sequence_parallel_moe", False):
            raise NotImplementedError(
                "DMI Llama 4 Scout lite support excludes sequence-parallel MoE"
            )
        if getattr(parallel_config, "enable_eplb", False):
            raise NotImplementedError("DMI Llama 4 Scout lite support excludes EPLB")


def _require_supported_llama4_scout_config(
    config,
    parallel_config=None,
    quant_config=None,
) -> None:
    """Validate the public multimodal wrapper and its monitored text tier."""

    if getattr(config, "model_type", None) != "llama4":
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires model_type=llama4"
        )
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    if text_config is None or vision_config is None:
        raise NotImplementedError(
            "DMI Llama 4 Scout support requires text and vision configs"
        )
    _require_supported_llama4_scout_text_config(
        text_config,
        parallel_config,
        quant_config,
    )
    expected_vision = {
        "model_type": "llama4_vision_model",
        "hidden_size": 1408,
        "num_hidden_layers": 34,
        "image_size": 336,
        "patch_size": 14,
    }
    for field, expected in expected_vision.items():
        if getattr(vision_config, field, None) != expected:
            raise NotImplementedError(
                f"DMI Llama 4 Scout support requires vision {field}={expected}"
            )


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _capture_compare_buffer(
    module: nn.Module,
    name: str,
    value: torch.Tensor,
) -> None:
    buffer = getattr(module, f"_buf_{name}", None)
    if buffer is not None:
        buffer[: value.shape[0]].copy_(value)


class Llama4PAttention(Llama4Attention):
    """Llama 4 attention with raw pre-RoPE Q/K/V and pre-o-proj Z hooks."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hooks = (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        if not any(hook.enabled for hook in hooks):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.hook_q.enabled:
            q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
            self.hook_q(q_by_head)
            _capture_compare_buffer(self, "q", q_by_head)
        if self.hook_k.enabled:
            k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            self.hook_k(k_by_head)
            _capture_compare_buffer(self, "k", k_by_head)
        if self.hook_v.enabled:
            v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            self.hook_v(v_by_head)
            _capture_compare_buffer(self, "v", v_by_head)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        if self.qk_norm is not None:
            q = q.reshape(-1, self.head_dim)
            q = self.qk_norm(q.float()).reshape(-1, self.q_size).to(q.dtype)
            k = k.reshape(-1, self.head_dim)
            k = self.qk_norm(k.float()).reshape(-1, self.kv_size).to(k.dtype)
        if self.attn_temperature_tuning and self.nope:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
            _capture_compare_buffer(self, "z", attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4PMoE(Llama4MoE):
    """Llama 4 MoE with token-major routing observation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hooks = (
            self.hook_router_logits,
            self.hook_topk_ids,
            self.hook_topk_weights,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(hidden_states)

        num_tokens = hidden_states.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        router_logits, _ = self.router(hidden_states)
        if self.hook_router_logits.enabled:
            self.hook_router_logits(router_logits)
            _capture_compare_buffer(self, "router_logits", router_logits)
        if self.hook_topk_ids.enabled or self.hook_topk_weights.enabled:
            topk_weights, topk_ids = self.experts.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            if self.hook_topk_ids.enabled:
                self.hook_topk_ids(topk_ids)
                _capture_compare_buffer(self, "topk_ids", topk_ids)
            if self.hook_topk_weights.enabled:
                self.hook_topk_weights(topk_weights)
                _capture_compare_buffer(self, "topk_weights", topk_weights)
        experts_out = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self.is_sequence_parallel:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]
        return experts_out


class Llama4PDecoderLayer(Llama4DecoderLayer):
    """Llama 4 decoder layer with fused-residual boundaries exposed."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(positions, hidden_states, residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
            _capture_compare_buffer(self, "resid_pre", residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
            _capture_compare_buffer(self, "ln1", hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
            _capture_compare_buffer(self, "attn_out", hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
            _capture_compare_buffer(self, "resid_mid", residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
            _capture_compare_buffer(self, "ln2", hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
            _capture_compare_buffer(self, "mlp_in", hidden_states)

        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
            _capture_compare_buffer(self, "mlp_out", hidden_states)
        return hidden_states, residual


def _instrument_llama4_language_model(
    language_model: Llama4ForCausalLM,
    parallel_config=None,
    quant_config=None,
) -> Llama4PForCausalLM:
    """Attach decoder hooks to the exact tree built by upstream Llama 4."""

    _require_supported_llama4_scout_text_config(
        language_model.config,
        parallel_config,
        quant_config,
    )
    language_model.__class__ = Llama4PForCausalLM
    _add_hook_points(language_model, ("token_ids", "final_logits"))
    model = language_model.model
    model.__class__ = Llama4PModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        if not isinstance(layer.feed_forward, Llama4MoE):
            raise NotImplementedError(
                "DMI Llama 4 Scout support requires MoE in every decoder layer"
            )
        layer.__class__ = Llama4PDecoderLayer
        layer.self_attn.__class__ = Llama4PAttention
        layer.feed_forward.__class__ = Llama4PMoE
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
        _add_hook_points(
            layer.feed_forward,
            ("router_logits", "topk_ids", "topk_weights"),
        )
    return language_model


class Llama4PModel(Llama4Model):
    """Concrete Llama 4 text backbone with embedding and final hooks."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                **extra_layer_kwargs,
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
                _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for index, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **extra_layer_kwargs,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states,
                index + 1,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, final_residual = self.norm(hidden_states, residual)
        if self.hook_resid_final.enabled:
            self.hook_resid_final(final_residual)
            _capture_compare_buffer(self, "resid_final", final_residual)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
            _capture_compare_buffer(self, "final_ln", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Llama4PForCausalLM(Llama4ForCausalLM):
    """Llama 4 Scout text model with a truthful DMI hook manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_llama4_scout_text_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_llama4_language_model(
            self,
            vllm_config.parallel_config,
            vllm_config.quant_config,
        )

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
            _capture_compare_buffer(self, "token_ids", input_ids)
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
            _capture_compare_buffer(self, "final_logits", logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        moe = None if layer is None else layer.feed_forward

        def hook(module, name: str):
            return None if module is None else getattr(module, f"hook_{name}")

        def spec(hook_type: int, module, name: str, **kwargs) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(module, name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
                **kwargs,
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
            spec(HOOK_TYPE_ROUTER_LOGITS, moe, "router_logits"),
            spec(HOOK_TYPE_TOPK_IDS, moe, "topk_ids", dtype=torch.int32),
            spec(
                HOOK_TYPE_TOPK_WEIGHTS,
                moe,
                "topk_weights",
                dtype=torch.float32,
            ),
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
    "Llama4PAttention",
    "Llama4PDecoderLayer",
    "Llama4PForCausalLM",
    "Llama4PModel",
    "Llama4PMoE",
    "_instrument_llama4_language_model",
    "_require_supported_llama4_scout_config",
    "_require_supported_llama4_scout_text_config",
]
