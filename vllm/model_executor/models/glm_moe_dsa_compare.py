# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 same-graph compare buffers for DMI storage tests."""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from .glm_moe_dsa_p import GlmMoeDsaPForCausalLM
from .utils import PPMissingLayer


class GlmMoeDsaCompareForCausalLM(GlmMoeDsaPForCausalLM):
    """Test-only GLM-5.2 model retaining decoder D2D references."""

    def allocate_compare_buffers(
        self,
        max_len: int,
        vllm_config: VllmConfig,
    ) -> None:
        config = self.config
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        tp_size = get_tensor_model_parallel_world_size()
        hidden_size = config.hidden_size
        model = self.model

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
            if layer_no < config.first_k_dense_replace:
                layer.mlp._buf_mlp_post = torch.empty(
                    max_len,
                    config.intermediate_size // tp_size,
                    device=device,
                    dtype=dtype,
                )
            else:
                layer.mlp._buf_router_logits = torch.empty(
                    max_len,
                    config.n_routed_experts,
                    device=device,
                    dtype=torch.float32,
                )
                layer.mlp._buf_topk_ids = torch.empty(
                    max_len,
                    config.num_experts_per_tok,
                    device=device,
                    dtype=torch.int32,
                )
                layer.mlp._buf_topk_weights = torch.empty(
                    max_len,
                    config.num_experts_per_tok,
                    device=device,
                    dtype=torch.float32,
                )

        max_requests = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(
            max_requests,
            config.vocab_size,
            device=device,
            dtype=dtype,
        )
        self._buf_token_ids = torch.empty(
            max_len,
            device=device,
            dtype=torch.int32,
        )

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        model = self.model
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
            if layer_no < self.config.first_k_dense_replace:
                buffers[f"mlp_post_L{layer_no}"] = layer.mlp._buf_mlp_post
            else:
                for name in ("router_logits", "topk_ids", "topk_weights"):
                    buffers[f"{name}_L{layer_no}"] = getattr(
                        layer.mlp,
                        f"_buf_{name}",
                    )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["GlmMoeDsaCompareForCausalLM"]
