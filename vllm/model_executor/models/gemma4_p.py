# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 E2B decoder-boundary monitoring behind the public MM wrapper.

The pinned E2B checkpoint has heterogeneous attention head dimensions and
heterogeneous MLP widths.  DMI's current shape contract describes one global
head/intermediate dimension, so this variant intentionally exposes only the
decoder boundaries whose shapes are uniform and exact.
"""

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
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_TOKEN_IDS,
    HookSpec,
)
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

from .gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
    Gemma4Model,
)
from .gemma4_mm import Gemma4ForConditionalGeneration
from .utils import PPMissingLayer

_EXPECTED_LAYER_TYPES = [
    "full_attention" if (layer_no + 1) % 5 == 0 else "sliding_attention"
    for layer_no in range(35)
]


def _require_value(config: Any, name: str, expected: Any) -> None:
    # Transformers' heterogeneous configs deliberately raise when a global
    # per-layer attribute (for example head_dim) is read through getattr.
    # The raw checkpoint value remains in __dict__ and is the value this
    # version-pinned validator needs to freeze.
    actual = vars(config).get(name, None)
    if name not in vars(config):
        actual = getattr(config, name, None)
    if actual != expected:
        raise NotImplementedError(
            f"DMI Gemma 4 E2B lite support requires {name}={expected!r}; got {actual!r}"
        )


def _require_supported_gemma4_e2b_config(
    config: Any,
    parallel_config: Any,
    quant_config: Any = None,
    dtype: torch.dtype | None = None,
    *,
    speculative_config: Any = None,
    kv_sharing_fast_prefill: bool = False,
) -> None:
    """Fail closed outside the exact public E2B BF16/TP1 lite cell."""

    _require_value(config, "model_type", "gemma4")
    _require_value(config, "image_token_id", 258880)
    _require_value(config, "audio_token_id", 258881)
    _require_value(config, "video_token_id", 258884)
    _require_value(config, "vision_soft_tokens_per_image", 280)

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support requires nested text_config"
        )
    expected_text = {
        "model_type": "gemma4_text",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "vocab_size": 262144,
        "max_position_embeddings": 131072,
        "sliding_window": 512,
        "num_kv_shared_layers": 20,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "use_double_wide_mlp": True,
        "use_bidirectional_attention": None,
        "tie_word_embeddings": True,
        "final_logit_softcapping": 30.0,
    }
    for name, expected in expected_text.items():
        _require_value(text_config, name, expected)
    global_head_dim = vars(text_config).get("global_head_dim")
    if global_head_dim is not None:
        _require_value(text_config, "global_head_dim", 512)
    per_layer_config = getattr(text_config, "per_layer_config", None)
    if per_layer_config is not None:
        expected_head_dims = [
            512 if layer_type == "full_attention" else 256
            for layer_type in _EXPECTED_LAYER_TYPES
        ]
        actual_head_dims = [vars(layer).get("head_dim") for layer in per_layer_config]
        if actual_head_dims != expected_head_dims:
            raise NotImplementedError(
                "DMI Gemma 4 E2B lite support requires heterogeneous "
                f"per-layer head_dim={expected_head_dims!r}; "
                f"got {actual_head_dims!r}"
            )
    elif global_head_dim != 512:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support requires global_head_dim=512; "
            f"got {global_head_dim!r}"
        )
    _require_value(text_config, "layer_types", _EXPECTED_LAYER_TYPES)
    _require_value(
        text_config,
        "rope_parameters",
        {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
    )

    vision_config = getattr(config, "vision_config", None)
    audio_config = getattr(config, "audio_config", None)
    if vision_config is None or audio_config is None:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support requires the public image/audio towers"
        )
    for name, expected in {
        "model_type": "gemma4_vision",
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 10240,
        "default_output_length": 280,
    }.items():
        _require_value(vision_config, name, expected)
    for name, expected in {
        "model_type": "gemma4_audio",
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "output_proj_dims": 1536,
        "conv_kernel_size": 5,
    }.items():
        _require_value(audio_config, name, expected)

    if quant_config is not None or getattr(text_config, "quantization_config", None):
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support is frozen to the BF16 checkpoint"
        )
    if dtype is not None and dtype != torch.bfloat16:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support requires BF16 runtime dtype"
        )
    if speculative_config is not None:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support excludes speculative/Eagle execution"
        )
    if kv_sharing_fast_prefill:
        raise NotImplementedError(
            "DMI Gemma 4 E2B lite support excludes KV-sharing fast prefill"
        )

    expected_parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    for name, expected in expected_parallel.items():
        actual = getattr(parallel_config, name, expected)
        if actual != expected:
            raise NotImplementedError(
                "DMI Gemma 4 E2B lite support requires TP1/PP1/DP1; "
                f"got {name}={actual!r}"
            )
    for name in (
        "enable_expert_parallel",
        "use_sequence_parallel_moe",
        "enable_eplb",
    ):
        if getattr(parallel_config, name, False):
            raise NotImplementedError(f"DMI Gemma 4 E2B lite support excludes {name}")
    for name in (
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel_config, name, 1) != 1:
            raise NotImplementedError(
                "DMI Gemma 4 E2B lite support excludes context parallelism"
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


class Gemma4PDecoderLayer(Gemma4DecoderLayer):
    """Gemma 4 layer exposing only uniform hidden-size boundaries."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
                per_layer_input=per_layer_input,
                **kwargs,
            )

        residual = hidden_states
        self.hook_resid_pre(residual)
        _capture_compare_buffer(self, "resid_pre", residual)

        hidden_states = self.input_layernorm(residual)
        self.hook_ln1(hidden_states)
        _capture_compare_buffer(self, "ln1", hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        self.hook_attn_out(hidden_states)
        _capture_compare_buffer(self, "attn_out", hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        self.hook_resid_mid(residual)
        _capture_compare_buffer(self, "resid_mid", residual)

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        self.hook_ln2(hidden_states)
        _capture_compare_buffer(self, "ln2", hidden_states)
        self.hook_mlp_in(hidden_states)
        _capture_compare_buffer(self, "mlp_in", hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self.hook_mlp_out(hidden_states)
        _capture_compare_buffer(self, "mlp_out", hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            contribution = self.per_layer_projection(gate * per_layer_input)
            contribution = self.post_per_layer_input_norm(contribution)
            hidden_states = hidden_states + contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


class Gemma4PModel(Gemma4Model):
    """Gemma 4 decoder backbone with embedding/final hidden boundaries."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        hooks = (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        if not any(hook.enabled for hook in hooks):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                per_layer_inputs,
                **kwargs,
            )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs
                )
            else:
                hidden_states = self.embed_input_ids(input_ids)
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds
                )
            self.hook_embed(hidden_states)
            _capture_compare_buffer(self, "embed", hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if per_layer_inputs is not None:
                per_layer_inputs = intermediate_tensors["per_layer_inputs"]

        residual = None
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            layer_per_input = None
            if per_layer_inputs is not None:
                actual_layer_idx = self.start_layer + layer_idx
                layer_per_input = per_layer_inputs[:, actual_layer_idx, :]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                per_layer_input=layer_per_input,
                **kwargs,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states,
                layer_idx + 1,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            tensors = {"hidden_states": hidden_states}
            if per_layer_inputs is not None:
                tensors["per_layer_inputs"] = per_layer_inputs
            return IntermediateTensors(tensors)

        self.hook_resid_final(hidden_states)
        _capture_compare_buffer(self, "resid_final", hidden_states)
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        _capture_compare_buffer(self, "final_ln", hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Gemma4PForCausalLM(Gemma4ForCausalLM):
    """Instrumented native language model owned by the MM wrapper."""

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
            spec(HOOK_TYPE_RESID_PRE, "resid_pre"),
            spec(HOOK_TYPE_LN1, "ln1"),
            spec(HOOK_TYPE_ATTN_OUT, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, "resid_mid"),
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


def _instrument_gemma4_language_model(
    language_model: Gemma4ForCausalLM,
) -> Gemma4PForCausalLM:
    language_model.__class__ = Gemma4PForCausalLM
    _add_hook_points(language_model, ("token_ids", "final_logits"))
    model = language_model.model
    model.__class__ = Gemma4PModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = Gemma4PDecoderLayer
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
    return language_model


class Gemma4PForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Public Gemma 4 multimodal model exporting decoder boundaries only."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_gemma4_e2b_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
            vllm_config.model_config.dtype,
            speculative_config=vllm_config.speculative_config,
            kv_sharing_fast_prefill=(vllm_config.cache_config.kv_sharing_fast_prefill),
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.language_model = _instrument_gemma4_language_model(self.language_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if self.language_model.hook_token_ids.enabled:
            self.language_model.hook_token_ids(input_ids)
            _capture_compare_buffer(self.language_model, "token_ids", input_ids)
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
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.language_model.hook_final_logits.enabled:
            self.language_model.hook_final_logits(logits)
            _capture_compare_buffer(self.language_model, "final_logits", logits)
        return logits

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = [
    "Gemma4PDecoderLayer",
    "Gemma4PForCausalLM",
    "Gemma4PForConditionalGeneration",
    "Gemma4PModel",
    "_instrument_gemma4_language_model",
    "_require_supported_gemma4_e2b_config",
]
