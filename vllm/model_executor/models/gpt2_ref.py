# SPDX-License-Identifier: Apache-2.0
# Reference GPT-2 model for identical check.
# Copy of gpt2.py with # BENCH_OFF D2D capture lines.
# No HookPoints.  Buffer allocation reads REF_CONFIG env.
"""Inference-only GPT-2 model with ref capture buffers."""

import json
import os
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import GPT2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class GPT2Attention(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // total_num_heads
        self.scale = self.head_dim**-0.5

        self.c_attn = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.c_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scale,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        # BENCH_OFF q: self._buf_q[:q_head.shape[0]].copy_(q_head)
        k_head = k.view(*k.shape[:-1], self.num_heads, self.head_dim)
        # BENCH_OFF k: self._buf_k[:k_head.shape[0]].copy_(k_head)
        v_head = v.view(*v.shape[:-1], self.num_heads, self.head_dim)
        # BENCH_OFF v: self._buf_v[:v_head.shape[0]].copy_(v_head)
        attn_output = self.attn(q, k, v)
        # BENCH_OFF z: self._buf_z[:attn_output.shape[0]].copy_(attn_output)
        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, config: GPT2Config,
                 quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True,
                                         quant_config=quant_config, prefix=f"{prefix}.c_fc")
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=True,
                                         quant_config=quant_config, prefix=f"{prefix}.c_proj")
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        # BENCH_OFF mlp_post: self._buf_mlp_post[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, cache_config: CacheConfig | None = None,
                 quant_config: QuantizationConfig | None = None, prefix: str = ""):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, cache_config, quant_config, prefix=f"{prefix}.attn")
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor):
        # BENCH_OFF resid_pre: self._buf_resid_pre[:hidden_states.shape[0]].copy_(hidden_states)
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # BENCH_OFF ln1: self._buf_ln1[:hidden_states.shape[0]].copy_(hidden_states)
        attn_output = self.attn(hidden_states=hidden_states)
        # BENCH_OFF attn_out: self._buf_attn_out[:attn_output.shape[0]].copy_(attn_output)
        hidden_states = attn_output + residual
        # BENCH_OFF resid_mid: self._buf_resid_mid[:hidden_states.shape[0]].copy_(hidden_states)
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        # BENCH_OFF ln2: self._buf_ln2[:hidden_states.shape[0]].copy_(hidden_states)
        # BENCH_OFF mlp_in: self._buf_mlp_in[:hidden_states.shape[0]].copy_(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # BENCH_OFF mlp_out: self._buf_mlp_out[:feed_forward_hidden_states.shape[0]].copy_(feed_forward_hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


@support_torch_compile
class GPT2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim,
                                           quant_config=quant_config, prefix=f"{prefix}.wte")
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h")
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.n_embd)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(self, input_ids: torch.Tensor | None, position_ids: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None,
                inputs_embeds: torch.Tensor | None) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            # BENCH_OFF embed: self._buf_embed[:inputs_embeds.shape[0]].copy_(inputs_embeds)
            position_embeds = self.wpe(position_ids)
            # BENCH_OFF pos_embed: self._buf_pos_embed[:position_embeds.shape[0]].copy_(position_embeds)
            hidden_states = inputs_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        # BENCH_OFF resid_final: self._buf_resid_final[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        # BENCH_OFF final_ln: self._buf_final_ln[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states

    def _transpose_conv1d(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Transpose Hugging Face Conv1D weights into vLLM layout."""

        for name, loaded_weight in weights:
            if name.endswith(".weight") and any(
                projection in name
                for projection in ("c_attn", "c_proj", "c_fc")
            ):
                loaded_weight = loaded_weight.t()
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self, skip_substrs=[".attn.bias", ".attn.masked_bias"]
        )
        return loader.load_weights(self._transpose_conv1d(weights))


class GPT2RefLMHeadModel(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "transformer"))
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
                                       quant_config=quant_config, prefix=f"{prefix}.lm_head")
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.transformer.wte)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.transformer.make_empty_intermediate_tensors
        self._init_ref_buffers(vllm_config)

    def _init_ref_buffers(self, vllm_config: VllmConfig) -> None:
        """Pre-allocate capture buffers for enabled hooks.

        Buffers are allocated before CUDA graph capture so they live
        outside the graph memory pool and won't be reused during replay.
        """
        cfg_path = os.environ.get("REF_CONFIG")
        if not cfg_path:
            return
        with open(cfg_path) as f:
            rc = json.load(f)
        enabled = set(rc["enabled_hooks"])
        max_len = rc.get("max_len", 8192)
        config = vllm_config.model_config.hf_config
        H = config.hidden_size                          # 768 for GPT-2
        n_heads = config.num_attention_heads             # 12
        head_dim = H // n_heads                          # 64
        inner_dim = config.n_inner if config.n_inner is not None else 4 * H  # 3072
        vocab_size = config.vocab_size                   # 50257
        dt = vllm_config.model_config.dtype or torch.bfloat16

        # TP: per-rank dimensions for sharded hooks
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp = get_tensor_model_parallel_world_size()
        n_heads_tp = n_heads // tp
        inner_dim_tp = inner_dim // tp

        def _alloc(*shape, dtype=dt):
            return torch.empty(*shape, dtype=dtype, device="cuda")

        tr = self.transformer
        if "embed" in enabled:
            tr._buf_embed = _alloc(max_len, H)
        if "pos_embed" in enabled:
            tr._buf_pos_embed = _alloc(max_len, H)
        if "resid_final" in enabled:
            tr._buf_resid_final = _alloc(max_len, H)
        if "final_ln" in enabled:
            tr._buf_final_ln = _alloc(max_len, H)
        for i in range(tr.start_layer, tr.end_layer):
            block, attn = tr.h[i], tr.h[i].attn
            if "resid_pre" in enabled:
                block._buf_resid_pre = _alloc(max_len, H)
            if "ln1" in enabled:
                block._buf_ln1 = _alloc(max_len, H)
            if "attn_out" in enabled:
                block._buf_attn_out = _alloc(max_len, H)
            if "resid_mid" in enabled:
                block._buf_resid_mid = _alloc(max_len, H)
            if "ln2" in enabled:
                block._buf_ln2 = _alloc(max_len, H)
            if "mlp_in" in enabled:
                block._buf_mlp_in = _alloc(max_len, H)
            if "mlp_out" in enabled:
                block._buf_mlp_out = _alloc(max_len, H)
            if "mlp_post" in enabled:
                block.mlp._buf_mlp_post = _alloc(max_len, inner_dim_tp)
            if "q" in enabled:
                attn._buf_q = _alloc(max_len, n_heads_tp, head_dim)
            if "k" in enabled:
                attn._buf_k = _alloc(max_len, n_heads_tp, head_dim)
            if "v" in enabled:
                attn._buf_v = _alloc(max_len, n_heads_tp, head_dim)
            if "z" in enabled:
                attn._buf_z = _alloc(max_len, n_heads_tp * head_dim)
        if "final_logits" in enabled:
            self._buf_final_logits = _alloc(max_len, vocab_size, dtype=torch.float32)
        if "token_ids" in enabled:
            self._buf_token_ids = _alloc(max_len, dtype=torch.int32)

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        """Return {name: tensor} for all captured ref buffers (non-None)."""
        bufs: dict[str, torch.Tensor] = {}
        tr = self.transformer
        for attr in ("_buf_embed", "_buf_pos_embed", "_buf_resid_final", "_buf_final_ln"):
            v = getattr(tr, attr, None)
            if v is not None:
                bufs[attr[5:]] = v
        for i in range(tr.start_layer, tr.end_layer):
            block, attn = tr.h[i], tr.h[i].attn
            for attr in ("_buf_resid_pre", "_buf_ln1", "_buf_attn_out",
                         "_buf_resid_mid", "_buf_ln2", "_buf_mlp_in", "_buf_mlp_out"):
                v = getattr(block, attr, None)
                if v is not None:
                    bufs[f"{attr[5:]}_L{i}"] = v
            # MLP internal hook (on block.mlp, not block)
            v = getattr(block.mlp, "_buf_mlp_post", None)
            if v is not None:
                bufs[f"mlp_post_L{i}"] = v
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                v = getattr(attn, attr, None)
                if v is not None:
                    bufs[f"{attr[5:]}_L{i}"] = v
        for attr in ("_buf_final_logits", "_buf_token_ids"):
            v = getattr(self, attr, None)
            if v is not None:
                bufs[attr[5:]] = v
        return bufs

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            # BENCH_OFF token_ids: self._buf_token_ids[:input_ids.shape[0]].copy_(input_ids)
            pass
        return self.transformer(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        # BENCH_OFF final_logits: self._buf_final_logits[:logits.shape[0]].copy_(logits)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(_add_transformer_prefix(weights))


def _add_transformer_prefix(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if not name.startswith("transformer.") and not name.startswith("lm_head"):
            name = "transformer." + name
        yield name, tensor
