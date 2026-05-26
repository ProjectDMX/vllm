# SPDX-License-Identifier: Apache-2.0
"""Qwen2MoE compare model: qwen2_moe_p + in-forward buffer copies."""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from .qwen2_moe_p import Qwen2MoePForCausalLM


class Qwen2MoeCompareForCausalLM(Qwen2MoePForCausalLM):
    def allocate_compare_buffers(self, max_len: int, vllm_config: VllmConfig) -> None:
        config = self.config
        H = config.hidden_size
        nh = config.num_attention_heads
        nkv = config.num_key_value_heads
        hd = getattr(config, "head_dim", None) or H // nh
        V = config.vocab_size
        E = config.num_experts
        K = config.num_experts_per_tok
        dtype = vllm_config.model_config.dtype

        tp = get_tensor_model_parallel_world_size()
        nh_tp = nh // tp
        nkv_tp = max(1, nkv // tp)
        I_tp = config.intermediate_size // tp
        device = "cuda"

        m = self.model
        m._buf_embed = torch.empty(max_len, H, device=device, dtype=dtype)
        m._buf_resid_final = torch.empty(max_len, H, device=device, dtype=dtype)
        m._buf_final_ln = torch.empty(max_len, H, device=device, dtype=dtype)

        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            attn = layer.self_attn
            layer._buf_resid_pre = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_ln1 = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_attn_out = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_resid_mid = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_ln2 = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_mlp_in = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_mlp_out = torch.empty(max_len, H, device=device, dtype=dtype)
            attn._buf_q = torch.empty(max_len, nh_tp, hd, device=device, dtype=dtype)
            attn._buf_k = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            attn._buf_v = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            attn._buf_z = torch.empty(max_len, nh_tp * hd, device=device, dtype=dtype)
            if hasattr(layer.mlp, "hook_post"):
                layer.mlp._buf_mlp_post = torch.empty(max_len, I_tp, device=device, dtype=dtype)
            if hasattr(layer.mlp, "hook_router_logits"):
                layer.mlp._buf_router_logits = torch.empty(max_len, E, device=device, dtype=dtype)
                layer.mlp._buf_topk_ids = torch.empty(max_len, K, device=device, dtype=torch.int32)
                layer.mlp._buf_topk_weights = torch.empty(max_len, K, device=device, dtype=dtype)

        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(max_reqs, V, device=device, dtype=dtype)
        self._buf_token_ids = torch.empty(max_len, device=device, dtype=torch.int32)

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        bufs: dict[str, torch.Tensor] = {}
        m = self.model
        for attr in ("_buf_embed", "_buf_resid_final", "_buf_final_ln"):
            if hasattr(m, attr):
                bufs[attr[5:]] = getattr(m, attr)
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            for attr in (
                "_buf_resid_pre",
                "_buf_ln1",
                "_buf_attn_out",
                "_buf_resid_mid",
                "_buf_ln2",
                "_buf_mlp_in",
                "_buf_mlp_out",
            ):
                if hasattr(layer, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(layer, attr)
            if hasattr(layer.mlp, "_buf_mlp_post"):
                bufs[f"mlp_post_L{i}"] = layer.mlp._buf_mlp_post
            if hasattr(layer.mlp, "_buf_router_logits"):
                bufs[f"router_logits_L{i}"] = layer.mlp._buf_router_logits
                bufs[f"topk_ids_L{i}"] = layer.mlp._buf_topk_ids
                bufs[f"topk_weights_L{i}"] = layer.mlp._buf_topk_weights
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                if hasattr(layer.self_attn, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(layer.self_attn, attr)
        bufs["final_logits"] = self._buf_final_logits
        bufs["token_ids"] = self._buf_token_ids
        return bufs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)
