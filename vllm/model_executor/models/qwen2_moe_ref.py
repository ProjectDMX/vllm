# SPDX-License-Identifier: Apache-2.0
"""Inference-only Qwen2MoE ref model."""

import json
import os
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

from .interfaces import SupportsLoRA, SupportsPP
from .qwen2_moe_p import Qwen2MoePForCausalLM


class Qwen2MoeRefForCausalLM(Qwen2MoePForCausalLM, SupportsPP, SupportsLoRA):
    """Reference Qwen2MoE model with buffer capture driven by REF_CONFIG."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_ref_buffers(vllm_config)

    def _init_ref_buffers(self, vllm_config: VllmConfig) -> None:
        cfg_path = os.environ.get("REF_CONFIG")
        if not cfg_path:
            return
        with open(cfg_path) as f:
            rc = json.load(f)

        max_len = rc["max_len"]
        enabled = set(rc["enabled_hooks"])
        config = self.config
        H = config.hidden_size
        nh = config.num_attention_heads
        nkv = config.num_key_value_heads
        hd = getattr(config, "head_dim", None) or H // nh
        V = config.vocab_size
        E = config.num_experts
        K = config.num_experts_per_tok
        tp = get_tensor_model_parallel_world_size()
        nh_tp = nh // tp
        nkv_tp = max(1, nkv // tp)
        I_tp = config.intermediate_size // tp
        device = "cuda"
        dtype = vllm_config.model_config.dtype

        m = self.model
        if "embed" in enabled:
            m._buf_embed = torch.empty(max_len, H, device=device, dtype=dtype)
        if "resid_final" in enabled:
            m._buf_resid_final = torch.empty(max_len, H, device=device, dtype=dtype)
        if "final_ln" in enabled:
            m._buf_final_ln = torch.empty(max_len, H, device=device, dtype=dtype)

        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            attn = layer.self_attn
            if "resid_pre" in enabled:
                layer._buf_resid_pre = torch.empty(max_len, H, device=device, dtype=dtype)
            if "ln1" in enabled:
                layer._buf_ln1 = torch.empty(max_len, H, device=device, dtype=dtype)
            if "attn_out" in enabled:
                layer._buf_attn_out = torch.empty(max_len, H, device=device, dtype=dtype)
            if "resid_mid" in enabled:
                layer._buf_resid_mid = torch.empty(max_len, H, device=device, dtype=dtype)
            if "ln2" in enabled:
                layer._buf_ln2 = torch.empty(max_len, H, device=device, dtype=dtype)
            if "mlp_in" in enabled:
                layer._buf_mlp_in = torch.empty(max_len, H, device=device, dtype=dtype)
            if "mlp_out" in enabled:
                layer._buf_mlp_out = torch.empty(max_len, H, device=device, dtype=dtype)
            if "mlp_post" in enabled and hasattr(layer.mlp, "hook_post"):
                layer.mlp._buf_mlp_post = torch.empty(max_len, I_tp, device=device, dtype=dtype)
            if "router_logits" in enabled and hasattr(layer.mlp, "hook_router_logits"):
                layer.mlp._buf_router_logits = torch.empty(max_len, E, device=device, dtype=dtype)
            if "topk_ids" in enabled and hasattr(layer.mlp, "hook_topk_ids"):
                layer.mlp._buf_topk_ids = torch.empty(max_len, K, device=device, dtype=torch.int32)
            if "topk_weights" in enabled and hasattr(layer.mlp, "hook_topk_weights"):
                layer.mlp._buf_topk_weights = torch.empty(max_len, K, device=device, dtype=torch.float32)
            if "q" in enabled:
                attn._buf_q = torch.empty(max_len, nh_tp, hd, device=device, dtype=dtype)
            if "k" in enabled:
                attn._buf_k = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            if "v" in enabled:
                attn._buf_v = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            if "z" in enabled:
                attn._buf_z = torch.empty(max_len, nh_tp * hd, device=device, dtype=dtype)

        if "final_logits" in enabled:
            self._buf_final_logits = torch.empty(max_len, V, device=device, dtype=dtype)
        if "token_ids" in enabled:
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
            if hasattr(layer.mlp, "_buf_topk_ids"):
                bufs[f"topk_ids_L{i}"] = layer.mlp._buf_topk_ids
            if hasattr(layer.mlp, "_buf_topk_weights"):
                bufs[f"topk_weights_L{i}"] = layer.mlp._buf_topk_weights
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                if hasattr(layer.self_attn, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(layer.self_attn, attr)
        for attr in ("_buf_final_logits", "_buf_token_ids"):
            if hasattr(self, attr):
                bufs[attr[5:]] = getattr(self, attr)
        return bufs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)
