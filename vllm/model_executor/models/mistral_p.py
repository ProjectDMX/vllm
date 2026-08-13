# SPDX-License-Identifier: Apache-2.0
"""Bounded Mistral variant using DMI's hooked Llama implementation."""

from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from .llama_p import LlamaPForCausalLM
from .mistral import MistralForCausalLM as _MistralForCausalLM


def _reject_unsupported_mistral_branches(vllm_config: VllmConfig) -> None:
    config = vllm_config.model_config.hf_config
    unsupported = []
    if getattr(config, "llama_4_scaling", None) is not None:
        unsupported.append("llama_4_scaling")
    if getattr(config, "ada_rms_norm_t_cond", False):
        unsupported.append("ada_rms_norm_t_cond")
    if unsupported:
        raise NotImplementedError(
            "DMI's Mistral variant has not audited config branch(es): "
            + ", ".join(unsupported)
        )


class MistralPForCausalLM(LlamaPForCausalLM):
    """Mistral text model for configs whose math matches target Llama."""

    embedding_modules: dict[str, str] = {}
    mistral_mapping = _MistralForCausalLM.mistral_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _reject_unsupported_mistral_branches(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    maybe_remap_mistral = _MistralForCausalLM.maybe_remap_mistral

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights
        )

    def get_quant_config(
        self, vllm_config: VllmConfig
    ) -> QuantizationConfig | None:
        return vllm_config.quant_config


__all__ = ["MistralPForCausalLM"]
