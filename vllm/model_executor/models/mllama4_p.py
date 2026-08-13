# SPDX-License-Identifier: Apache-2.0
"""Llama 4 multimodal wrapper exporting only language-decoder DMI hooks."""

from __future__ import annotations

from vllm.config import VllmConfig

from .llama4_p import (
    _instrument_llama4_language_model,
    _require_supported_llama4_scout_config,
)
from .mllama4 import Llama4ForConditionalGeneration


class Llama4PForConditionalGeneration(Llama4ForConditionalGeneration):
    """Preserve Llama 4 public multimodal behavior and monitor its decoder."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_supported_llama4_scout_config(
            vllm_config.model_config.hf_config,
            vllm_config.parallel_config,
            vllm_config.quant_config,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.language_model = _instrument_llama4_language_model(
            self.language_model,
            vllm_config.parallel_config,
            vllm_config.quant_config,
        )

    def get_hook_specs(self, model_wide: bool = False):
        return self.language_model.get_hook_specs(model_wide=model_wide)


__all__ = ["Llama4PForConditionalGeneration"]
