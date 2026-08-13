# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decoder boundaries and MoE routing hooks for DMI."""

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
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HookSpec,
)
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
    GlmMoeDsaForCausalLM,
)
from .utils import PPMissingLayer


def _glm52_indexer_types() -> list[str]:
    return [
        "full" if layer_no < 3 or (layer_no - 2) % 4 == 0 else "shared"
        for layer_no in range(78)
    ]


def _require_supported_glm52_config(
    config,
    parallel_config=None,
    quant_config=None,
    dtype=None,
    *,
    use_mla: bool = True,
    speculative_config=None,
) -> None:
    """Fail closed outside the exact GLM-5.2 BF16 TP32 lite cell."""

    exact_fields = {
        "model_type": "glm_moe_dsa",
        "hidden_act": "silu",
        "hidden_size": 6144,
        "intermediate_size": 12_288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "moe_router_dtype": "float32",
        "ep_size": 1,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "index_topk_pattern": None,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
        "max_position_embeddings": 1_048_576,
        "vocab_size": 154_880,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "num_nextn_predict_layers": 1,
    }
    for field, expected in exact_fields.items():
        if getattr(config, field, None) != expected:
            raise NotImplementedError(
                f"DMI GLM-5.2 support requires {field}={expected}"
            )
    if getattr(config, "attention_bias", False):
        raise NotImplementedError("DMI GLM-5.2 support requires bias-free attention")
    if getattr(config, "mlp_bias", False):
        raise NotImplementedError("DMI GLM-5.2 support requires bias-free MLPs")
    if getattr(config, "tie_word_embeddings", None) is not False:
        raise NotImplementedError("DMI GLM-5.2 support requires untied embeddings")
    if getattr(config, "llama_4_scaling", None) is not None:
        raise NotImplementedError("DMI GLM-5.2 lite support excludes Llama-4 scaling")
    if (
        list(getattr(config, "layer_types", ()) or ())
        != ["deepseek_sparse_attention"] * 78
    ):
        raise NotImplementedError("DMI GLM-5.2 support requires DSA in all 78 layers")
    if list(getattr(config, "mlp_layer_types", ()) or ()) != [
        *(["dense"] * 3),
        *(["sparse"] * 75),
    ]:
        raise NotImplementedError(
            "DMI GLM-5.2 support requires 3 dense then 75 sparse MLP layers"
        )
    if list(getattr(config, "indexer_types", ()) or ()) != _glm52_indexer_types():
        raise NotImplementedError(
            "DMI GLM-5.2 support requires the audited DSA indexer-sharing schedule"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping) or dict(rope) != {
        "rope_type": "default",
        "rope_theta": 8_000_000,
    }:
        raise NotImplementedError(
            "DMI GLM-5.2 support requires default interleaved RoPE at theta 8000000"
        )
    if getattr(config, "rope_interleave", None) is not True:
        raise NotImplementedError("DMI GLM-5.2 support requires interleaved RoPE")
    if quant_config is not None or getattr(config, "quantization_config", None):
        raise NotImplementedError(
            "DMI GLM-5.2 lite support is limited to unquantized BF16 weights"
        )
    if dtype is not None and dtype is not torch.bfloat16:
        raise NotImplementedError(
            "DMI GLM-5.2 lite support requires runtime dtype=torch.bfloat16"
        )
    if use_mla is not True:
        raise NotImplementedError("DMI GLM-5.2 support requires vLLM MLA")
    if speculative_config is not None:
        raise NotImplementedError(
            "DMI GLM-5.2 lite support excludes speculative/MTP execution"
        )

    if parallel_config is not None:
        if (
            getattr(parallel_config, "tensor_parallel_size", 1) != 32
            or getattr(parallel_config, "pipeline_parallel_size", 1) != 1
            or getattr(parallel_config, "data_parallel_size", 1) != 1
            or getattr(parallel_config, "enable_expert_parallel", False)
        ):
            raise NotImplementedError(
                "DMI GLM-5.2 lite support requires TP32/PP1/DP1 without EP"
            )
        if getattr(parallel_config, "use_sequence_parallel_moe", False):
            raise NotImplementedError(
                "DMI GLM-5.2 lite support excludes sequence-parallel MoE"
            )
        if getattr(parallel_config, "enable_eplb", False):
            raise NotImplementedError("DMI GLM-5.2 lite support excludes EPLB")
        if any(
            getattr(parallel_config, field, 1) != 1
            for field in (
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise NotImplementedError(
                "DMI GLM-5.2 lite support excludes context parallelism"
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


class GlmMoeDsaPMLP(DeepseekV2MLP):
    """The three dense GLM MLPs with post-activation observation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        gate_up, _ = self.gate_up_proj(x)
        post = self.act_fn(gate_up)
        self.hook_post(post)
        _capture_compare_buffer(self, "mlp_post", post)
        output, _ = self.down_proj(post)
        return output


class GlmMoeDsaPMoE(DeepseekV2MoE):
    """GLM sparse MLP exposing replicated routing decisions only."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        hooks = (
            self.hook_router_logits,
            self.hook_topk_ids,
            self.hook_topk_weights,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(hidden_states, already_sequence_parallel)
        if self.is_sequence_parallel or already_sequence_parallel:
            raise RuntimeError("GLM-5.2 lite routing hooks require non-SP execution")

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

        experts_router_input = (
            flat_states if self.experts.is_internal_router else router_logits
        )
        output = self.experts(
            hidden_states=flat_states,
            router_logits=experts_router_input,
        )
        return output.view(num_tokens, hidden_dim)


class GlmMoeDsaPDecoderLayer(DeepseekV2DecoderLayer):
    """One GLM layer exposing boundaries around opaque MLA/DSA attention."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
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
            return super().forward(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
            )
        if self.use_sequence_parallel_moe:
            raise RuntimeError("GLM-5.2 lite decoder hooks require non-SP execution")
        if llama_4_scaling is not None:
            raise RuntimeError("GLM-5.2 lite decoder excludes Llama-4 scaling")

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

        hidden_states = self.self_attn(positions, hidden_states, None)
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

        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
            _capture_compare_buffer(self, "mlp_out", hidden_states)
        return hidden_states, residual


def _instrument_glm52_model(
    model: DeepseekV2Model,
    parallel_config=None,
    quant_config=None,
    dtype=None,
    *,
    use_mla: bool = True,
    speculative_config=None,
) -> GlmMoeDsaPModel:
    """Attach hooks to the exact GLM tree constructed by upstream."""

    _require_supported_glm52_config(
        model.config,
        parallel_config,
        quant_config,
        dtype,
        use_mla=use_mla,
        speculative_config=speculative_config,
    )
    model.__class__ = GlmMoeDsaPModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        if not isinstance(layer.self_attn, DeepseekV2MLAAttention):
            raise NotImplementedError("DMI GLM-5.2 support requires MLA attention")
        layer.__class__ = GlmMoeDsaPDecoderLayer
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
        if layer_no < model.config.first_k_dense_replace:
            if not isinstance(layer.mlp, DeepseekV2MLP):
                raise NotImplementedError(
                    "DMI GLM-5.2 support requires dense MLPs in layers 0-2"
                )
            layer.mlp.__class__ = GlmMoeDsaPMLP
            _add_hook_points(layer.mlp, ("post",))
        else:
            if not isinstance(layer.mlp, DeepseekV2MoE):
                raise NotImplementedError(
                    "DMI GLM-5.2 support requires MoE MLPs after layer 2"
                )
            layer.mlp.__class__ = GlmMoeDsaPMoE
            _add_hook_points(
                layer.mlp,
                ("router_logits", "topk_ids", "topk_weights"),
            )
    return model


class GlmMoeDsaPModel(DeepseekV2Model):
    """GLM backbone with decoder-global input and output observations."""

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
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
                _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_no, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if layer_no in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                None,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

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


class GlmMoeDsaPForCausalLM(GlmMoeDsaForCausalLM):
    """Exact GLM-5.2 public class with a truthful reduced manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_glm52_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            use_mla=vllm_config.model_config.use_mla,
            speculative_config=vllm_config.speculative_config,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _add_hook_points(self, ("token_ids", "final_logits"))
        self.model = _instrument_glm52_model(
            self.model,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            use_mla=vllm_config.model_config.use_mla,
            speculative_config=vllm_config.speculative_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
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

        specs = [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
            spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
            spec(HOOK_TYPE_LN2, layer, "ln2"),
            spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
        ]
        if layer_no < self.config.first_k_dense_replace:
            specs.append(spec(HOOK_TYPE_MLP_POST, mlp, "post"))
        else:
            specs.extend(
                [
                    spec(HOOK_TYPE_ROUTER_LOGITS, mlp, "router_logits"),
                    spec(HOOK_TYPE_TOPK_IDS, mlp, "topk_ids", dtype=torch.int32),
                    spec(
                        HOOK_TYPE_TOPK_WEIGHTS,
                        mlp,
                        "topk_weights",
                        dtype=torch.float32,
                    ),
                ]
            )
        specs.append(spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"))
        return specs

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
    "GlmMoeDsaPDecoderLayer",
    "GlmMoeDsaPForCausalLM",
    "GlmMoeDsaPMLP",
    "GlmMoeDsaPModel",
    "GlmMoeDsaPMoE",
    "_glm52_indexer_types",
    "_instrument_glm52_model",
    "_require_supported_glm52_config",
]
