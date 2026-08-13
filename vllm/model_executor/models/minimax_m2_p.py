# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M2.7 model with bounded DMI decoder hooks."""

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
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.minimax_rms_norm import MiniMaxText01RMSNormTP
from vllm.sequence import IntermediateTensors

from .minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2MoE,
)
from .utils import PPMissingLayer


def _require_supported_minimax_m27_config(
    config,
    parallel_config=None,
    quant_config=None,
    dtype=None,
    *,
    speculative_config=None,
) -> None:
    """Fail closed outside the exact MiniMax-M2.7 FP8 TP4 lite cell."""

    exact_fields = {
        "model_type": "minimax_m2",
        "hidden_act": "silu",
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rotary_dim": 64,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "shared_intermediate_size": 0,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "output_router_logits": False,
        "use_qk_norm": True,
        "qk_norm_type": "per_layer",
        "max_position_embeddings": 204_800,
        "vocab_size": 200_064,
        "rms_norm_eps": 1e-6,
        "attention_dropout": 0.0,
        "router_jitter_noise": 0.0,
        "num_mtp_modules": 3,
        "mtp_transformer_layers": 1,
        "use_mtp": True,
    }
    for field, expected in exact_fields.items():
        if getattr(config, field, None) != expected:
            raise NotImplementedError(
                f"DMI MiniMax-M2.7 support requires {field}={expected}"
            )
    if getattr(config, "attention_bias", False):
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires bias-free attention"
        )
    if getattr(config, "tie_word_embeddings", False):
        raise NotImplementedError("DMI MiniMax-M2.7 support requires untied embeddings")
    if list(getattr(config, "attn_type_list", ()) or ()) != [1] * 62:
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires full attention in all 62 layers"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping):
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires the audited partial RoPE"
        )
    if (
        rope.get("rope_type") != "default"
        or rope.get("rope_theta") != 5_000_000
        or rope.get("partial_rotary_factor", 0.5) != 0.5
    ):
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires default half-head RoPE"
        )
    quantization = getattr(config, "quantization_config", None)
    expected_quantization = {
        "activation_scheme": "dynamic",
        "fmt": "float8_e4m3fn",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "gate",
            "e_score_correction_bias",
            "lm_head",
        ],
    }
    if not isinstance(quantization, Mapping) or dict(quantization) != (
        expected_quantization
    ):
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires the native block-FP8 checkpoint"
        )
    if quant_config is not None and quant_config.get_name() != "fp8":
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires the vLLM FP8 quantizer"
        )
    if dtype is not None and dtype is not torch.bfloat16:
        raise NotImplementedError(
            "DMI MiniMax-M2.7 support requires runtime dtype=torch.bfloat16"
        )
    if speculative_config is not None:
        raise NotImplementedError(
            "DMI MiniMax-M2.7 lite support excludes speculative/MTP execution"
        )

    if parallel_config is not None:
        if (
            getattr(parallel_config, "tensor_parallel_size", 1) != 4
            or getattr(parallel_config, "pipeline_parallel_size", 1) != 1
            or getattr(parallel_config, "data_parallel_size", 1) != 1
            or getattr(parallel_config, "enable_expert_parallel", False)
        ):
            raise NotImplementedError(
                "DMI MiniMax-M2.7 lite support requires TP4/PP1/DP1 without EP"
            )
        if getattr(parallel_config, "use_sequence_parallel_moe", False):
            raise NotImplementedError(
                "DMI MiniMax-M2.7 lite support excludes sequence-parallel MoE"
            )
        if getattr(parallel_config, "enable_eplb", False):
            raise NotImplementedError("DMI MiniMax-M2.7 lite support excludes EPLB")
        if any(
            getattr(parallel_config, field, 1) != 1
            for field in (
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise NotImplementedError(
                "DMI MiniMax-M2.7 lite support excludes context parallelism"
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


class MiniMaxM2PAttention(MiniMaxM2Attention):
    """Attention exposing post-QK-norm/pre-RoPE Q/K/V and pre-o-proj Z."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hooks = (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        if not any(hook.enabled for hook in hooks):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = MiniMaxText01RMSNormTP.forward_qkv(
            self.q_norm,
            self.k_norm,
            qkv,
            self.q_size,
            self.kv_size,
        )
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
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
            _capture_compare_buffer(self, "z", attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM2PMoE(MiniMaxM2MoE):
    """MiniMax MoE exposing replicated FP32 routing decisions."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hooks = (
            self.hook_router_logits,
            self.hook_topk_ids,
            self.hook_topk_weights,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(hidden_states)

        num_tokens, hidden_dim = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden_dim)
        router_logits, _ = self.gate(flat_states)
        if self.hook_router_logits.enabled:
            self.hook_router_logits(router_logits)
            _capture_compare_buffer(self, "router_logits", router_logits)
        if self.hook_topk_ids.enabled or self.hook_topk_weights.enabled:
            topk_weights, topk_ids = self.experts.router.select_experts(
                hidden_states=flat_states,
                router_logits=router_logits,
            )
            if self.hook_topk_ids.enabled:
                self.hook_topk_ids(topk_ids)
                _capture_compare_buffer(self, "topk_ids", topk_ids)
            if self.hook_topk_weights.enabled:
                self.hook_topk_weights(topk_weights)
                _capture_compare_buffer(self, "topk_weights", topk_weights)
        output = self.experts(
            hidden_states=flat_states,
            router_logits=router_logits,
        )
        return output.view(num_tokens, hidden_dim)


class MiniMaxM2PDecoderLayer(MiniMaxM2DecoderLayer):
    """One MiniMax layer with fused-residual boundaries exposed."""

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

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
            _capture_compare_buffer(self, "resid_mid", residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
            _capture_compare_buffer(self, "ln2", hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
            _capture_compare_buffer(self, "mlp_in", hidden_states)

        hidden_states = self.block_sparse_moe(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
            _capture_compare_buffer(self, "mlp_out", hidden_states)
        return hidden_states, residual


def _instrument_minimax_m27_model(
    model: MiniMaxM2Model,
    parallel_config=None,
    quant_config=None,
    dtype=None,
    *,
    speculative_config=None,
) -> MiniMaxM2PModel:
    """Attach hooks to the native MiniMax model constructed by upstream."""

    _require_supported_minimax_m27_config(
        model.config,
        parallel_config,
        quant_config,
        dtype,
        speculative_config=speculative_config,
    )
    model.__class__ = MiniMaxM2PModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = MiniMaxM2PDecoderLayer
        layer.self_attn.__class__ = MiniMaxM2PAttention
        layer.block_sparse_moe.__class__ = MiniMaxM2PMoE
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
            layer.block_sparse_moe,
            ("router_logits", "topk_ids", "topk_weights"),
        )
    return model


class MiniMaxM2PModel(MiniMaxM2Model):
    """MiniMax backbone with embedding and final residual/norm hooks."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in hooks):
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
                _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_no, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_no + 1,
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


class MiniMaxM2PForCausalLM(MiniMaxM2ForCausalLM):
    """Exact MiniMax-M2.7 public class with a complete decoder manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_minimax_m27_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _add_hook_points(self, ("token_ids", "final_logits"))
        self.model = _instrument_minimax_m27_model(
            self.model,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
        )

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
            _capture_compare_buffer(self, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
            _capture_compare_buffer(self, "final_logits", logits)
        return logits

    def _layer_hook_specs(
        self,
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        moe = None if layer is None else layer.block_sparse_moe

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
    "MiniMaxM2PAttention",
    "MiniMaxM2PDecoderLayer",
    "MiniMaxM2PForCausalLM",
    "MiniMaxM2PModel",
    "MiniMaxM2PMoE",
    "_instrument_minimax_m27_model",
    "_require_supported_minimax_m27_config",
]
