# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 compare model: identical to qwen3_p.py (HookPoints + ring::producer)
but also captures tensors via .copy_() to pre-allocated buffers for bitwise
comparison against ClickHouse output. Used for transport correctness testing
under torch.compile + CUDA graphs where ref and monitored models must have
identical compiled graphs."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .qwen2 import Qwen2MLP as _Qwen2MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HookSpec,
    HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_RESID_MID, HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V,
    HOOK_TYPE_Z, HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN, HOOK_TYPE_MLP_OUT, HOOK_TYPE_MLP_POST, HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_EMBED, HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
)

logger = init_logger(__name__)


class Qwen3MLP(_Qwen2MLP):
    """Qwen3 MLP with hook_post (after activation, before down_proj)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_post = HookPoint()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        self._buf_mlp_post[:x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=per_layer_sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        self.hook_q(q_by_head)
        self._buf_q[:q_by_head.shape[0]].copy_(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        self.hook_k(k_by_head)
        self._buf_k[:k_by_head.shape[0]].copy_(k_by_head)
        k = k_by_head.view(k.shape)
        v_head = v.view(*v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim)
        self.hook_v(v_head)
        self._buf_v[:v_head.shape[0]].copy_(v_head)
        v = v_head.view(v.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:attn_output.shape[0]].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hook_resid_pre = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # resid_pre: read-only capture, preserves fused norm
        if residual is not None:
            rp = hidden_states + residual
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(rp)
            self._buf_resid_pre[:rp.shape[0]].copy_(rp)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(hidden_states)
            self._buf_resid_pre[:hidden_states.shape[0]].copy_(hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        self.hook_ln1(hidden_states)
        self._buf_ln1[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        self.hook_attn_out(hidden_states)
        self._buf_attn_out[:hidden_states.shape[0]].copy_(hidden_states)

        rm = hidden_states + residual
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(rm)
        self._buf_resid_mid[:rm.shape[0]].copy_(rm)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )
        self.hook_embed = HookPoint()
        if get_pp_group().is_last_rank:
            self.hook_resid_final = HookPoint()
            self.hook_final_ln = HookPoint()
        else:
            self.hook_resid_final = PPMissingLayer()
            self.hook_final_ln = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            self.hook_embed(hidden_states)
            self._buf_embed[:hidden_states.shape[0]].copy_(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], 0, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual})

        rf = hidden_states + residual
        if self.hook_resid_final.enabled:
            self.hook_resid_final(rf)
        self._buf_resid_final[:rf.shape[0]].copy_(rf)
        hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[:hidden_states.shape[0]].copy_(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3CompareForCausalLM(
    LocalArgmaxMixin,
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.hook_final_logits = HookPoint()
        self.hook_token_ids = HookPoint()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            self._buf_token_ids[:input_ids.shape[0]].copy_(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        self._buf_final_logits[:logits.shape[0]].copy_(logits)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def allocate_compare_buffers(self, max_len: int, vllm_config: VllmConfig) -> None:
        """Allocate D2D capture buffers for bitwise comparison against ring transport."""
        config = self.config
        H = config.hidden_size
        nh = config.num_attention_heads
        nkv = config.num_key_value_heads
        hd = getattr(config, "head_dim", None) or H // nh
        V = config.vocab_size
        dtype = vllm_config.model_config.dtype

        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
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
            if isinstance(layer, PPMissingLayer):
                continue
            attn = layer.self_attn
            layer._buf_resid_pre = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_ln1 = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_attn_out = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_resid_mid = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_ln2 = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_mlp_in = torch.empty(max_len, H, device=device, dtype=dtype)
            layer._buf_mlp_out = torch.empty(max_len, H, device=device, dtype=dtype)
            layer.mlp._buf_mlp_post = torch.empty(max_len, I_tp, device=device, dtype=dtype)
            attn._buf_q = torch.empty(max_len, nh_tp, hd, device=device, dtype=dtype)
            attn._buf_k = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            attn._buf_v = torch.empty(max_len, nkv_tp, hd, device=device, dtype=dtype)
            attn._buf_z = torch.empty(max_len, nh_tp * hd, device=device, dtype=dtype)

        # final_logits is [num_reqs, vocab], not [total_tokens, vocab].
        # Use max_num_seqs (much smaller than max_len) to avoid OOM.
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(max_reqs, V, device=device, dtype=dtype)
        self._buf_token_ids = torch.empty(max_len, device=device, dtype=torch.int32)

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        """Return {name: buffer} for all allocated compare buffers."""
        bufs: dict[str, torch.Tensor] = {}
        m = self.model
        for attr in ("_buf_embed", "_buf_resid_final", "_buf_final_ln"):
            if hasattr(m, attr):
                bufs[attr[5:]] = getattr(m, attr)
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            attn = layer.self_attn
            for attr in ("_buf_resid_pre", "_buf_ln1", "_buf_attn_out",
                         "_buf_resid_mid", "_buf_ln2", "_buf_mlp_in", "_buf_mlp_out"):
                if hasattr(layer, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(layer, attr)
            v = getattr(layer.mlp, "_buf_mlp_post", None)
            if v is not None:
                bufs[f"mlp_post_L{i}"] = v
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                if hasattr(attn, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(attn, attr)
        for attr in ("_buf_final_logits", "_buf_token_ids"):
            if hasattr(self, attr):
                bufs[attr[5:]] = getattr(self, attr)
        return bufs

    def _get_layer_hook_specs(self, layer_no: int, layer) -> list[HookSpec]:
        attn = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.mlp

        def hook(module, name: str):
            return None if module is None else getattr(module, name)

        return [
            HookSpec(HOOK_TYPE_RESID_PRE, hook(layer, "hook_resid_pre"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN1, hook(layer, "hook_ln1"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_Q, hook(attn, "hook_q"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_K, hook(attn, "hook_k"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_V, hook(attn, "hook_v"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_Z, hook(attn, "hook_z"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_ATTN_OUT, hook(layer, "hook_attn_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_RESID_MID, hook(layer, "hook_resid_mid"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN2, hook(layer, "hook_ln2"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_IN, hook(layer, "hook_mlp_in"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_POST, hook(mlp, "hook_post"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_OUT, hook(layer, "hook_mlp_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
        ]

    def get_hook_specs(
        self, *, model_wide: bool = False
    ) -> list[HookSpec]:
        specs: list[HookSpec] = []
        m = self.model
        # vLLM flat layout: dim-0 of every per-token hook is total_tokens.
        # Mark these specs so the vLLM adapter (when padding_strip=True)
        # can substitute actual_q_len for q_len in shape + reservation.
        # Excluded: FINAL_LOGITS (dim-0 = num_requests, not total_tokens).
        specs.append(HookSpec(HOOK_TYPE_TOKEN_IDS, None if model_wide else self.hook_token_ids, dtype=torch.int32, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_EMBED, None if model_wide else m.hook_embed, dim0_is_actual_tokens=True))
        layer_indices = (
            range(len(m.layers))
            if model_wide
            else range(m.start_layer, m.end_layer)
        )
        for i in layer_indices:
            layer = None if model_wide else m.layers[i]
            if layer is not None and isinstance(layer, PPMissingLayer):
                continue
            specs.extend(self._get_layer_hook_specs(i, layer))
        specs.append(HookSpec(HOOK_TYPE_RESID_FINAL, None if model_wide else m.hook_resid_final, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LN, None if model_wide else m.hook_final_ln, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LOGITS, None if model_wide else self.hook_final_logits))
        return specs
