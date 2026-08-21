# SPDX-License-Identifier: Apache-2.0
"""Bounded ERNIE 4.5 dense variant using DMI's hooked Llama model."""

from __future__ import annotations

from collections.abc import Mapping

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

from .llama_p import LlamaPForCausalLM
from .utils import PPMissingLayer


def _require_supported_ernie45_config(vllm_config: VllmConfig) -> None:
    """Fail closed outside the audited ERNIE 4.5 0.3B text contract."""

    config = vllm_config.model_config.hf_config
    if getattr(config, "model_type", None) != "ernie4_5":
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires model_type=ernie4_5"
        )
    if getattr(config, "hidden_act", None) != "silu":
        raise NotImplementedError("DMI ERNIE 4.5 support requires SiLU")
    if getattr(config, "use_bias", False):
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires bias-free projections"
        )
    if getattr(config, "is_causal", True) is False:
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires causal attention"
        )
    if getattr(config, "layer_types", None):
        raise NotImplementedError(
            "DMI ERNIE 4.5 support does not include mixed/sliding schedules"
        )
    if not getattr(config, "tie_word_embeddings", False):
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires tied input/output embeddings"
        )
    if getattr(config, "logit_scale", 1.0) not in (None, 1.0):
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires unit logit scale"
        )

    head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires an explicit positive head_dim"
        )
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, Mapping):
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires rope_parameters"
        )
    if rope.get("rope_type") != "default" or rope.get("rope_theta") != 500_000:
        raise NotImplementedError(
            "DMI ERNIE 4.5 support requires the audited default RoPE at "
            "theta 500000"
        )


def _apply_ernie45_attention_contract(model) -> None:
    """Replay the exact post-construction adjustments from upstream ERNIE."""

    for layer in model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        layer.self_attn.rotary_emb.is_neox_style = False
        layer.self_attn.o_proj.bias = None
        layer.self_attn.o_proj.skip_bias_add = True


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Ernie4_5PForCausalLM(LlamaPForCausalLM):
    """ERNIE 4.5 dense text model for the audited 0.3B configuration."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_ernie45_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_ernie45_attention_contract(self.model)


__all__ = ["Ernie4_5PForCausalLM"]
