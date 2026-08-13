# SPDX-License-Identifier: Apache-2.0
"""Phi-3 causal LM using DMI's hooked Llama implementation."""

from .llama_p import LlamaPForCausalLM
from .phi3 import Phi3ForCausalLM as _Phi3ForCausalLM


class Phi3PForCausalLM(LlamaPForCausalLM):
    """Preserve Phi-3 fused weight packing while exposing Llama hooks."""

    packed_modules_mapping = _Phi3ForCausalLM.packed_modules_mapping


__all__ = ["Phi3PForCausalLM"]
