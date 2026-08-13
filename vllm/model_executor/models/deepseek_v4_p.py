# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash plugin model with bounded decoder monitoring hooks."""

from __future__ import annotations

from itertools import islice
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
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from vllm.sequence import IntermediateTensors

_EXPECTED_COMPRESS_RATIOS = [
    0,
    0,
    *[ratio for _ in range(20) for ratio in (4, 128)],
    4,
    0,
]


def _require_value(config: Any, name: str, expected: Any) -> None:
    actual = vars(config).get(name, getattr(config, name, None))
    if actual != expected:
        raise NotImplementedError(
            f"DMI DeepSeek V4 Flash lite support requires {name}={expected!r}; "
            f"got {actual!r}"
        )


def _require_supported_deepseek_v4_flash_config(
    config: Any,
    parallel_config: Any,
    quant_config: Any = None,
    dtype: torch.dtype | None = None,
    *,
    speculative_config: Any = None,
    moe_backend: str | None = None,
) -> None:
    """Fail closed outside the pinned NVIDIA BF16/TP4 Flash cell."""

    expected = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "expert_dtype": "fp4",
        "hc_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 512,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "max_position_embeddings": 1_048_576,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 6,
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 1,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
        "vocab_size": 129_280,
        "compress_rope_theta": 160_000,
        "compress_ratios": _EXPECTED_COMPRESS_RATIOS,
    }
    for name, value in expected.items():
        _require_value(config, name, value)

    rope_scaling = dict(getattr(config, "rope_scaling", {}) or {})
    rope_type = rope_scaling.pop("rope_type", None)
    legacy_rope_type = rope_scaling.pop("type", None)
    rope_theta = rope_scaling.pop("rope_theta", 10_000)
    if rope_type is None:
        rope_type = legacy_rope_type
    if (
        rope_type != "yarn"
        or rope_scaling
        != {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65_536,
        }
        or rope_theta != 10_000
    ):
        raise NotImplementedError(
            "DMI DeepSeek V4 Flash lite support requires the official YaRN "
            f"contract; got {getattr(config, 'rope_scaling', None)!r}"
        )

    expected_quant = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    if getattr(config, "quantization_config", None) != expected_quant:
        raise NotImplementedError(
            "DMI DeepSeek V4 Flash lite support requires native FP8/FP4 "
            "checkpoint quantization"
        )
    if quant_config is not None:
        get_name = getattr(quant_config, "get_name", None)
        if not callable(get_name) or get_name() != "deepseek_v4_fp8":
            raise NotImplementedError(
                "DMI DeepSeek V4 Flash lite support requires vLLM "
                "deepseek_v4_fp8 quantization"
            )
        if getattr(quant_config, "weight_block_size", [128, 128]) != [128, 128]:
            raise NotImplementedError(
                "DMI DeepSeek V4 Flash lite support requires 128x128 FP8 blocks"
            )
    if dtype is not None and dtype != torch.bfloat16:
        raise NotImplementedError(
            "DMI DeepSeek V4 Flash lite support requires BF16 runtime dtype"
        )
    if speculative_config is not None:
        raise NotImplementedError(
            "DMI DeepSeek V4 Flash lite support excludes speculative/MTP execution"
        )
    if moe_backend == "deep_gemm_mega_moe":
        raise NotImplementedError(
            "DMI DeepSeek V4 Flash H100 support excludes the SM100 MegaMoE backend"
        )

    for name, value in {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }.items():
        actual = getattr(parallel_config, name, value)
        if actual != value:
            raise NotImplementedError(
                "DMI DeepSeek V4 Flash lite support requires TP4/PP1/DP1; "
                f"got {name}={actual!r}"
            )
    for name in (
        "enable_expert_parallel",
        "use_sequence_parallel_moe",
        "enable_eplb",
        "use_ubatching",
    ):
        if getattr(parallel_config, name, False):
            raise NotImplementedError(
                f"DMI DeepSeek V4 Flash lite support excludes {name}"
            )
    for name in (
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel_config, name, 1) != 1:
            raise NotImplementedError(
                "DMI DeepSeek V4 Flash lite support excludes context parallelism"
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


class DeepseekV4PDecoderLayer(DeepseekV4DecoderLayer):
    """Expose only uniform two-dimensional MHC decoder boundaries."""

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hooks = (
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                x,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )

        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        if residual is None:
            if x.dim() == 2:
                assert self.hc_attn_fn_broadcast is not None
                residual, post_mix, res_mix, x = mhc_pre_broadcast_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                    fn_broadcast=self.hc_attn_fn_broadcast,
                )
            else:
                residual = x
                post_mix, res_mix, x = mhc_pre_tilelang(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                )
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )

        if self.hook_ln1.enabled:
            self.hook_ln1(x)
            _capture_compare_buffer(self, "ln1", x)
        if self.use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]
        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)
        if self.hook_attn_out.enabled:
            self.hook_attn_out(x)
            _capture_compare_buffer(self, "attn_out", x)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        if self.hook_ln2.enabled:
            self.hook_ln2(x)
            _capture_compare_buffer(self, "ln2", x)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(x)
            _capture_compare_buffer(self, "mlp_in", x)
        x = self.ffn(x, input_ids)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(x)
            _capture_compare_buffer(self, "mlp_out", x)
        return x, residual, post_mix, res_mix


class DeepseekV4PModel(DeepseekV4Model):
    """Native MHC model with embedding and collapsed final-state hooks."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
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
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
                _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None
        layer = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_state = aux_recon.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            hidden_states = (
                final_aux_recon
                if self.end_layer in self.aux_hidden_state_layers
                else mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states)
            _capture_compare_buffer(self, "resid_final", hidden_states)
        hidden_states = self.norm(hidden_states)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
            _capture_compare_buffer(self, "final_ln", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class DeepseekV4PForCausalLM(DeepseekV4ForCausalLM):
    """Pinned NVIDIA DeepSeek V4 Flash with a truthful reduced manifest."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_deepseek_v4_flash_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
            moe_backend=vllm_config.kernel_config.moe_backend,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _add_hook_points(self, ("token_ids", "final_logits"))
        self.model.__class__ = DeepseekV4PModel
        _add_hook_points(self.model, ("embed", "resid_final", "final_ln"))
        for layer_no in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            layer.__class__ = DeepseekV4PDecoderLayer
            _add_hook_points(layer, ("ln1", "attn_out", "ln2", "mlp_in", "mlp_out"))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.hook_token_ids.enabled:
            self.hook_token_ids(input_ids)
            _capture_compare_buffer(self, "token_ids", input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
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
    "DeepseekV4PDecoderLayer",
    "DeepseekV4PForCausalLM",
    "DeepseekV4PModel",
    "_require_supported_deepseek_v4_flash_config",
]
