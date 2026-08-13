# SPDX-License-Identifier: Apache-2.0
"""ERNIE 4.5 compare model for same-graph storage verification."""

from __future__ import annotations

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig

from .ernie45_p import (
    _apply_ernie45_attention_contract,
    _require_supported_ernie45_config,
)
from .llama_compare import LlamaCompareForCausalLM


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Ernie4_5CompareForCausalLM(LlamaCompareForCausalLM):
    """ERNIE 4.5 with DMI hooks and independent D2D reference copies."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_ernie45_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _apply_ernie45_attention_contract(self.model)


__all__ = ["Ernie4_5CompareForCausalLM"]
