# SPDX-License-Identifier: Apache-2.0
"""Qwen3-MoE decoder with bounded DMI observation hooks."""

from __future__ import annotations

from collections.abc import Mapping

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
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from .utils import PPMissingLayer


def _require_supported_qwen3_moe_config(config, parallel_config=None) -> None:
    """Fail closed outside the audited Qwen3-30B-A3B decoder contract."""

    if getattr(config, "model_type", None) != "qwen3_moe":
        raise NotImplementedError("DMI Qwen3-MoE support requires model_type=qwen3_moe")
    if getattr(config, "hidden_act", None) != "silu":
        raise NotImplementedError("DMI Qwen3-MoE support requires hidden_act=silu")
    if getattr(config, "attention_bias", False):
        raise NotImplementedError("DMI Qwen3-MoE support requires bias-free attention")
    if getattr(config, "tie_word_embeddings", None) is not False:
        raise NotImplementedError("DMI Qwen3-MoE support requires untied embeddings")
    if getattr(config, "output_router_logits", False):
        raise NotImplementedError(
            "DMI Qwen3-MoE support excludes public router-logit outputs"
        )

    num_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if (
        not isinstance(num_experts, int)
        or isinstance(num_experts, bool)
        or num_experts <= 0
        or not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 0 < top_k <= num_experts
    ):
        raise NotImplementedError(
            "DMI Qwen3-MoE support requires a valid expert/top-k layout"
        )
    if getattr(config, "decoder_sparse_step", None) != 1 or list(
        getattr(config, "mlp_only_layers", ()) or ()
    ):
        raise NotImplementedError(
            "DMI Qwen3-MoE lite support requires every decoder layer to be MoE"
        )
    if getattr(config, "shared_expert_intermediate_size", None) not in (None, 0):
        raise NotImplementedError("DMI Qwen3-MoE lite support excludes shared experts")
    if getattr(config, "norm_topk_prob", None) is not True:
        raise NotImplementedError(
            "DMI Qwen3-MoE support requires normalized top-k probabilities"
        )

    num_layers = getattr(config, "num_hidden_layers", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if (
        not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or num_layers <= 0
        or not isinstance(num_heads, int)
        or isinstance(num_heads, bool)
        or num_heads <= 0
        or not isinstance(num_kv_heads, int)
        or isinstance(num_kv_heads, bool)
        or num_kv_heads <= 0
        or num_heads % num_kv_heads
        or not isinstance(head_dim, int)
        or isinstance(head_dim, bool)
        or head_dim <= 0
    ):
        raise NotImplementedError(
            "DMI Qwen3-MoE support requires a valid explicit GQA head layout"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping):
        rope = getattr(config, "rope_scaling", None)
    if rope is not None and (
        not isinstance(rope, Mapping)
        or rope.get("rope_type") not in (None, "default")
        or any(
            key in rope
            for key in (
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
            )
        )
    ):
        raise NotImplementedError(
            "DMI Qwen3-MoE lite support excludes scaled or extended RoPE"
        )
    rope_theta = (
        rope.get("rope_theta", getattr(config, "rope_theta", None))
        if isinstance(rope, Mapping)
        else getattr(config, "rope_theta", None)
    )
    if rope_theta != 1_000_000:
        raise NotImplementedError("DMI Qwen3-MoE support requires RoPE theta 1000000")
    if getattr(config, "quantization_config", None) is not None:
        raise NotImplementedError(
            "DMI Qwen3-MoE lite support is limited to unquantized BF16 weights"
        )

    if parallel_config is not None:
        if any(
            getattr(parallel_config, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            )
        ) or getattr(parallel_config, "enable_expert_parallel", False):
            raise NotImplementedError(
                "DMI Qwen3-MoE lite support is limited to TP1/PP1/DP1 without EP"
            )
        if getattr(parallel_config, "use_sequence_parallel_moe", False):
            raise NotImplementedError(
                "DMI Qwen3-MoE lite support excludes sequence-parallel MoE routing"
            )
        if getattr(parallel_config, "enable_eplb", False):
            raise NotImplementedError("DMI Qwen3-MoE lite support excludes EPLB")


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


class Qwen3MoePAttention(Qwen3MoeAttention):
    """Qwen3 attention with post-QK-norm/pre-RoPE Q/K/V and Z hooks."""

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

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        if self.hook_q.enabled:
            self.hook_q(q_by_head)
            _capture_compare_buffer(self, "q", q_by_head)
        if self.hook_k.enabled:
            self.hook_k(k_by_head)
            _capture_compare_buffer(self, "k", k_by_head)
        if self.hook_v.enabled:
            v_by_head = v.view(
                *v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim
            )
            self.hook_v(v_by_head)
            _capture_compare_buffer(self, "v", v_by_head)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
            _capture_compare_buffer(self, "z", attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoePSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Qwen3 sparse block with token-major routing observation."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hooks = (
            self.hook_router_logits,
            self.hook_topk_ids,
            self.hook_topk_weights,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(hidden_states)

        if hidden_states.dim() not in (1, 2):
            raise AssertionError("Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs")
        is_input_1d = hidden_states.dim() == 1
        num_tokens = 1 if is_input_1d else hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)
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

        experts_router_input = (
            hidden_states if self.experts.is_internal_router else router_logits
        )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=experts_router_input,
        )
        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoePDecoderLayer(Qwen3MoeDecoderLayer):
    """One Qwen3-MoE decoder layer with fused-residual boundaries exposed."""

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

        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
            _capture_compare_buffer(self, "mlp_out", hidden_states)
        return hidden_states, residual


def _instrument_qwen3_moe_model(
    model: Qwen3MoeModel,
    parallel_config=None,
) -> Qwen3MoePModel:
    """Attach hooks to the exact module tree constructed by upstream."""

    _require_supported_qwen3_moe_config(model.config, parallel_config)
    model.__class__ = Qwen3MoePModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            raise NotImplementedError(
                "DMI Qwen3-MoE lite support requires sparse MoE in every layer"
            )
        layer.__class__ = Qwen3MoePDecoderLayer
        layer.self_attn.__class__ = Qwen3MoePAttention
        layer.mlp.__class__ = Qwen3MoePSparseMoeBlock
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
            layer.mlp,
            ("router_logits", "topk_ids", "topk_weights"),
        )
    return model


class Qwen3MoePModel(Qwen3MoeModel):
    """Concrete Qwen3-MoE backbone with embedding and final hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_qwen3_moe_model(self, vllm_config.parallel_config)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
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

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_no in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_no](
                positions, hidden_states, residual
            )
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


class Qwen3MoePForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3-MoE causal LM with a truthful DMI hook manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        _require_supported_qwen3_moe_config(config, vllm_config.parallel_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model = _instrument_qwen3_moe_model(
            self.model, vllm_config.parallel_config
        )
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

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
        mlp = None if layer is None else layer.mlp

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
            spec(HOOK_TYPE_ROUTER_LOGITS, mlp, "router_logits"),
            spec(HOOK_TYPE_TOPK_IDS, mlp, "topk_ids", dtype=torch.int32),
            spec(
                HOOK_TYPE_TOPK_WEIGHTS,
                mlp,
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
    "Qwen3MoePAttention",
    "Qwen3MoePDecoderLayer",
    "Qwen3MoePForCausalLM",
    "Qwen3MoePModel",
    "Qwen3MoePSparseMoeBlock",
    "_instrument_qwen3_moe_model",
    "_require_supported_qwen3_moe_config",
]
