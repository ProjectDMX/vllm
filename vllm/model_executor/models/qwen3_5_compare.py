# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 decoder-side compare buffers behind the multimodal wrapper."""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from .qwen3_5_p import Qwen3_5PForConditionalGeneration
from .utils import PPMissingLayer


class Qwen3_5CompareForConditionalGeneration(Qwen3_5PForConditionalGeneration):
    """Test-only Qwen3.6 wrapper retaining decoder D2D references."""

    def allocate_compare_buffers(
        self,
        max_len: int,
        vllm_config: VllmConfig,
    ) -> None:
        language_model = self.language_model
        config = language_model.config
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size // tp_size
        model = language_model.model

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
            layer.mlp._buf_mlp_post = torch.empty(
                max_len,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            if layer.layer_type == "full_attention":
                layer.self_attn._buf_q = torch.empty(
                    max_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                for name in ("k", "v"):
                    setattr(
                        layer.self_attn,
                        f"_buf_{name}",
                        torch.empty(
                            max_len,
                            num_kv_heads,
                            head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                    )
                layer.self_attn._buf_z = torch.empty(
                    max_len,
                    num_heads * head_dim,
                    device=device,
                    dtype=dtype,
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
            buffers[f"mlp_post_L{layer_no}"] = layer.mlp._buf_mlp_post
            if layer.layer_type == "full_attention":
                for name in ("q", "k", "v", "z"):
                    buffers[f"{name}_L{layer_no}"] = getattr(
                        layer.self_attn,
                        f"_buf_{name}",
                    )
        buffers["final_logits"] = language_model._buf_final_logits
        buffers["token_ids"] = language_model._buf_token_ids
        return buffers


__all__ = ["Qwen3_5CompareForConditionalGeneration"]
