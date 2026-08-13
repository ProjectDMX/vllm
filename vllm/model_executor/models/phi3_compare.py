# SPDX-License-Identifier: Apache-2.0
"""Phi-3 compare model for byte-identical DMI transport tests."""

from .llama_compare import LlamaCompareForCausalLM
from .phi3 import Phi3ForCausalLM as _Phi3ForCausalLM


class Phi3CompareForCausalLM(LlamaCompareForCausalLM):
    """Preserve Phi-3 packing in the independent-reference test model."""

    packed_modules_mapping = _Phi3ForCausalLM.packed_modules_mapping


__all__ = ["Phi3CompareForCausalLM"]
