# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 multimodal wrapper exporting reduced language-decoder hooks."""

from __future__ import annotations

from typing import Any

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
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_TOKEN_IDS,
    HookSpec,
)
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.sequence import IntermediateTensors

_FULL_ATTN_LAYERS = [*range(4, 93, 4), 93]
_KDA_LAYERS = [
    layer_no for layer_no in range(1, 94) if layer_no not in _FULL_ATTN_LAYERS
]


def _require_value(config: Any, name: str, expected: Any) -> None:
    actual = vars(config)[name] if name in vars(config) else getattr(config, name, None)
    if actual != expected:
        raise NotImplementedError(
            f"DMI Kimi K3 lite support requires {name}={expected!r}; got {actual!r}"
        )


def _require_supported_kimi_k3_config(
    config: Any,
    parallel_config: Any,
    quant_config: Any = None,
    dtype: torch.dtype | None = None,
    *,
    speculative_config: Any = None,
    moe_backend: str | None = None,
) -> None:
    """Fail closed outside the pinned NVIDIA BF16/TP32 K3 cell."""

    for name, expected in {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "bos_token_id": 163_584,
        "eos_token_id": 163_586,
        "pad_token_id": 163_839,
        "image_placeholder": "<|kimi_image_placeholder|>",
        "media_placeholder_token_id": 163_605,
        "tie_word_embeddings": False,
    }.items():
        _require_value(config, name, expected)

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise NotImplementedError("DMI Kimi K3 lite support requires text_config")
    expected_text = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "first_k_dense_replace": 1,
        "hidden_act": "situ",
        "hidden_size": 7168,
        "intermediate_size": 33_792,
        "kv_lora_rank": 512,
        "latent_moe_use_norm": True,
        "max_position_embeddings": 1_048_576,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "moe_intermediate_size": 3072,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_hidden_layers": 93,
        "num_key_value_heads": 96,
        "num_nextn_predict_layers": 0,
        "num_shared_experts": 2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-5,
        "routed_expert_hidden_size": 3584,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "v_head_dim": 128,
        "vocab_size": 163_840,
    }
    for name, expected in expected_text.items():
        _require_value(text_config, name, expected)

    linear_attn = getattr(text_config, "linear_attn_config", None)
    if linear_attn != {
        "full_attn_layers": _FULL_ATTN_LAYERS,
        "gate_lower_bound": -5.0,
        "head_dim": 128,
        "kda_layers": _KDA_LAYERS,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    }:
        raise NotImplementedError(
            "DMI Kimi K3 lite support requires the official 69-KDA/24-MLA "
            f"schedule; got {linear_attn!r}"
        )

    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise NotImplementedError("DMI Kimi K3 lite support requires vision_config")
    for name, expected in {
        "patch_size": 14,
        "merge_kernel_size": (2, 2),
        "merge_type": "sd2_tpool",
        "mm_projector_type": "patchmergerv2",
        "mm_hidden_size": 1024,
        "qkv_hidden_size": 1536,
        "text_hidden_size": 7168,
        "vt_hidden_size": 1024,
        "vt_intermediate_size": 4096,
        "vt_num_attention_heads": 12,
        "vt_num_hidden_layers": 27,
    }.items():
        _require_value(vision_config, name, expected)

    quantization = getattr(text_config, "quantization_config", None) or {}
    groups = quantization.get("config_groups", {})
    group = groups.get("group_0", {})
    weights = group.get("weights", {})
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("format") != "mxfp4-pack-quantized"
        or quantization.get("quantization_status") != "compressed"
        or group.get("format") != "mxfp4-pack-quantized"
        or weights.get("group_size") != 32
        or weights.get("num_bits") != 4
        or weights.get("scale_dtype") != "torch.uint8"
    ):
        raise NotImplementedError(
            "DMI Kimi K3 lite support requires the official compressed-tensors "
            "MXFP4 checkpoint"
        )
    if quant_config is not None:
        get_name = getattr(quant_config, "get_name", None)
        if not callable(get_name) or get_name() != "compressed-tensors":
            raise NotImplementedError(
                "DMI Kimi K3 lite support requires vLLM compressed-tensors"
            )
    if dtype is not None and dtype != torch.bfloat16:
        raise NotImplementedError("DMI Kimi K3 lite support requires BF16 dtype")
    if speculative_config is not None:
        raise NotImplementedError(
            "DMI Kimi K3 lite support excludes speculative/MTP execution"
        )
    if moe_backend == "deep_gemm_mega_moe":
        raise NotImplementedError(
            "DMI Kimi K3 H100 support excludes the SM100 MegaMoE backend"
        )

    for name, expected in {
        "tensor_parallel_size": 32,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }.items():
        actual = getattr(parallel_config, name, expected)
        if actual != expected:
            raise NotImplementedError(
                f"DMI Kimi K3 lite support requires TP32/PP1/DP1; got {name}={actual!r}"
            )
    for name in (
        "enable_expert_parallel",
        "use_sequence_parallel_moe",
        "enable_eplb",
        "use_ubatching",
    ):
        if getattr(parallel_config, name, False):
            raise NotImplementedError(f"DMI Kimi K3 lite support excludes {name}")
    for name in (
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel_config, name, 1) != 1:
            raise NotImplementedError(
                "DMI Kimi K3 lite support excludes context parallelism"
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


class KimiK3PDecoderLayer(KimiDecoderLayer):
    """Expose uniform 2D boundaries around KDA/MLA and dense/MoE blocks."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        hooks = (
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                positions,
                hidden_states,
                residual,
                prefix_sum,
                **kwargs,
            )

        hidden_states, prefix_sum, residual = self._pre_attn_norm(
            hidden_states, residual, prefix_sum
        )
        assert hidden_states is not None
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
            _capture_compare_buffer(self, "ln1", hidden_states)

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        hidden_states = self._run_self_attn(positions, hidden_states)
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
            _capture_compare_buffer(self, "attn_out", hidden_states)

        hidden_states, prefix_sum, residual = self._post_attn_norm(
            hidden_states, residual, prefix_sum
        )
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
        return hidden_states, prefix_sum, residual


class KimiK3PModel(KimiLinearModel):
    """Kimi text model with embedding and final pre-norm state hooks."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final)
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                **kwargs,
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
        assert hidden_states is not None

        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            if self.use_attn_res or residual is None:
                aux_hidden_states.append(hidden_states)
            else:
                aux_hidden_states.append(hidden_states + residual)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None

        prefix_sum = None
        if self.use_attn_res:
            block_residual = hidden_states.new_empty(
                hidden_states.size(0),
                self.num_attn_res_blocks,
                hidden_states.size(1),
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                prefix_sum=prefix_sum,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                if self.use_attn_res:
                    assert prefix_sum is not None
                    aux_hidden_state = prefix_sum + hidden_states
                else:
                    assert residual is not None
                    aux_hidden_state = hidden_states + residual
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)

        assert hidden_states is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.use_attn_res:
            assert prefix_sum is not None
            hidden_states = attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
            )
        else:
            hidden_states = hidden_states + residual
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states)
            _capture_compare_buffer(self, "resid_final", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class KimiK3PLinearForCausalLM(KimiLinearForCausalLM):
    """Instrumented native language model owned by the K3 wrapper."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids is not None and self.hook_token_ids.enabled:
            self.hook_token_ids(input_ids)
            _capture_compare_buffer(self, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        hidden_states = self.model.norm(hidden_states, None)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
            _capture_compare_buffer(self, "final_ln", hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
            _capture_compare_buffer(self, "final_logits", logits)
        return logits

    def _layer_hook_specs(
        self,
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        def hook(name: str):
            return None if layer is None else getattr(layer, f"hook_{name}")

        def spec(hook_type: int, name: str) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            )

        return [
            spec(HOOK_TYPE_LN1, "ln1"),
            spec(HOOK_TYPE_ATTN_OUT, "attn_out"),
            spec(HOOK_TYPE_LN2, "ln2"),
            spec(HOOK_TYPE_MLP_IN, "mlp_in"),
            spec(HOOK_TYPE_MLP_OUT, "mlp_out"),
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
                    None if model_wide else self.hook_final_ln,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LOGITS,
                    None if model_wide else self.hook_final_logits,
                ),
            ]
        )
        return specs


def _instrument_kimi_k3_language_model(
    language_model: KimiLinearForCausalLM,
) -> KimiK3PLinearForCausalLM:
    language_model.__class__ = KimiK3PLinearForCausalLM
    _add_hook_points(language_model, ("token_ids", "final_ln", "final_logits"))
    model = language_model.model
    model.__class__ = KimiK3PModel
    _add_hook_points(model, ("embed", "resid_final"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = KimiK3PDecoderLayer
        _add_hook_points(layer, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))
    return language_model


class KimiK3PForConditionalGeneration(KimiK3ForConditionalGeneration):
    """Public Kimi K3 wrapper exporting only reduced decoder boundaries."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_kimi_k3_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
            moe_backend=vllm_config.kernel_config.moe_backend,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.language_model = _instrument_kimi_k3_language_model(self.language_model)

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = [
    "KimiK3PDecoderLayer",
    "KimiK3PForConditionalGeneration",
    "KimiK3PLinearForCausalLM",
    "KimiK3PModel",
    "_instrument_kimi_k3_language_model",
    "_require_supported_kimi_k3_config",
]
