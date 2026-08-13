# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 E2B same-graph references for decoder-boundary storage tests."""

from __future__ import annotations

import torch

from vllm.config import VllmConfig

from .gemma4_p import Gemma4PForConditionalGeneration
from .utils import PPMissingLayer


class Gemma4CompareForConditionalGeneration(Gemma4PForConditionalGeneration):
    """Test-only Gemma 4 wrapper retaining exact D2D decoder references."""

    def allocate_compare_buffers(
        self,
        max_len: int,
        vllm_config: VllmConfig,
    ) -> None:
        language_model = self.language_model
        model = language_model.model
        config = language_model.config
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        hidden_size = config.hidden_size

        for name in ("embed", "resid_final", "final_ln"):
            setattr(
                model,
                f"_buf_{name}",
                torch.empty(max_len, hidden_size, device=device, dtype=dtype),
            )
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                setattr(
                    layer,
                    f"_buf_{name}",
                    torch.empty(max_len, hidden_size, device=device, dtype=dtype),
                )

        max_requests = vllm_config.scheduler_config.max_num_seqs
        language_model._buf_final_logits = torch.empty(
            max_requests,
            config.vocab_size,
            device=device,
            dtype=dtype,
        )
        language_model._buf_token_ids = torch.empty(
            max_len,
            device=device,
            dtype=torch.int32,
        )

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        language_model = self.language_model
        model = language_model.model
        for name in ("embed", "resid_final", "final_ln"):
            buffers[name] = getattr(model, f"_buf_{name}")
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer,
                    f"_buf_{name}",
                )
        buffers["final_logits"] = language_model._buf_final_logits
        buffers["token_ids"] = language_model._buf_token_ids
        return buffers


__all__ = ["Gemma4CompareForConditionalGeneration"]
