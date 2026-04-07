# SPDX-License-Identifier: Apache-2.0
# GPT-2 compare model for transport correctness testing.
# Copy of gpt2_p.py with .copy_() capture added after each hook.
"""GPT-2 compare model: HookPoints + .copy_() in same compiled graph."""

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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HookSpec,
    HOOK_TYPE_RESID_PRE, HOOK_TYPE_LN1, HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_RESID_MID, HOOK_TYPE_Q, HOOK_TYPE_K, HOOK_TYPE_V,
    HOOK_TYPE_Z, HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN, HOOK_TYPE_MLP_OUT, HOOK_TYPE_MLP_POST, HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_EMBED, HOOK_TYPE_POS_EMBED, HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS, HOOK_TYPE_TOKEN_IDS,
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
            self.hidden_size, self.head_dim, total_num_heads,
            bias=True, quant_config=quant_config, prefix=f"{prefix}.c_attn",
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size, self.hidden_size,
            bias=True, quant_config=quant_config, prefix=f"{prefix}.c_proj",
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, scale=self.scale,
            cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        # Hooks
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        q_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        self.hook_q(q_head)
        self._buf_q[:q_head.shape[0]].copy_(q_head)
        k_head = k.view(*k.shape[:-1], self.num_heads, self.head_dim)
        self.hook_k(k_head)
        self._buf_k[:k_head.shape[0]].copy_(k_head)
        v_head = v.view(*v.shape[:-1], self.num_heads, self.head_dim)
        self.hook_v(v_head)
        self._buf_v[:v_head.shape[0]].copy_(v_head)

        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:attn_output.shape[0]].copy_(attn_output)

        attn_output, _ = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: GPT2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=True, quant_config=quant_config, prefix=f"{prefix}.c_fc",
        )
        self.c_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=True, quant_config=quant_config, prefix=f"{prefix}.c_proj",
        )
        self.act = get_act_fn(config.activation_function)
        self.hook_post = HookPoint()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        self.hook_post(hidden_states)
        self._buf_mlp_post[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(
            config, cache_config, quant_config, prefix=f"{prefix}.attn")
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config, quant_config, prefix=f"{prefix}.mlp")

        # Hooks
        self.hook_resid_pre = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.hook_resid_pre(hidden_states)
        self._buf_resid_pre[:hidden_states.shape[0]].copy_(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        self.hook_ln1(hidden_states)
        self._buf_ln1[:hidden_states.shape[0]].copy_(hidden_states)

        attn_output = self.attn(hidden_states=hidden_states)
        self.hook_attn_out(attn_output)
        self._buf_attn_out[:attn_output.shape[0]].copy_(attn_output)

        hidden_states = attn_output + residual
        self.hook_resid_mid(hidden_states)
        self._buf_resid_mid[:hidden_states.shape[0]].copy_(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:hidden_states.shape[0]].copy_(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(feed_forward_hidden_states)
        self._buf_mlp_out[:feed_forward_hidden_states.shape[0]].copy_(feed_forward_hidden_states)

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
        self.wte = VocabParallelEmbedding(
            config.vocab_size, self.embed_dim,
            quant_config=quant_config, prefix=f"{prefix}.wte",
        )
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.n_embd
        )

        # Hooks
        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            self.hook_embed(inputs_embeds)
            self._buf_embed[:inputs_embeds.shape[0]].copy_(inputs_embeds)
            position_embeds = self.wpe(position_ids)
            self.hook_pos_embed(position_embeds)
            self._buf_pos_embed[:position_embeds.shape[0]].copy_(position_embeds)
            hidden_states = inputs_embeds + position_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.h, self.start_layer, self.end_layer):
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        self.hook_resid_final(hidden_states)
        self._buf_resid_final[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            for conv1d_weight_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_weight_name not in name:
                    continue
                if not name.endswith(".weight"):
                    continue
                loaded_weight = loaded_weight.t()
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GPT2CompareForCausalLM(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.transformer = GPT2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.lm_head",
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.transformer.wte)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

        # Hooks
        self.hook_final_logits = HookPoint()
        self.hook_token_ids = HookPoint()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_input_ids(input_ids)

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
        return self.transformer(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        self._buf_final_logits[:logits.shape[0]].copy_(logits)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        weights = _add_transformer_prefix(weights)
        return loader.load_weights(weights)

    def allocate_compare_buffers(self, max_len: int, vllm_config: VllmConfig) -> None:
        config = vllm_config.model_config.hf_config
        H = config.hidden_size
        n_heads = config.num_attention_heads
        head_dim = H // n_heads
        inner_dim = config.n_inner if config.n_inner is not None else 4 * H
        V = config.vocab_size
        dtype = vllm_config.model_config.dtype

        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp = get_tensor_model_parallel_world_size()
        n_heads_tp = n_heads // tp
        inner_dim_tp = inner_dim // tp

        device = "cuda"
        tr = self.transformer
        tr._buf_embed = torch.empty(max_len, H, device=device, dtype=dtype)
        tr._buf_pos_embed = torch.empty(max_len, H, device=device, dtype=dtype)
        tr._buf_resid_final = torch.empty(max_len, H, device=device, dtype=dtype)
        tr._buf_final_ln = torch.empty(max_len, H, device=device, dtype=dtype)

        for i in range(tr.start_layer, tr.end_layer):
            block = tr.h[i]
            attn = block.attn
            block._buf_resid_pre = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_ln1 = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_attn_out = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_resid_mid = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_ln2 = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_mlp_in = torch.empty(max_len, H, device=device, dtype=dtype)
            block._buf_mlp_out = torch.empty(max_len, H, device=device, dtype=dtype)
            block.mlp._buf_mlp_post = torch.empty(max_len, inner_dim_tp, device=device, dtype=dtype)
            attn._buf_q = torch.empty(max_len, n_heads_tp, head_dim, device=device, dtype=dtype)
            attn._buf_k = torch.empty(max_len, n_heads_tp, head_dim, device=device, dtype=dtype)
            attn._buf_v = torch.empty(max_len, n_heads_tp, head_dim, device=device, dtype=dtype)
            attn._buf_z = torch.empty(max_len, n_heads_tp * head_dim, device=device, dtype=dtype)

        max_reqs = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(max_reqs, V, device=device, dtype=dtype)
        self._buf_token_ids = torch.empty(max_len, device=device, dtype=torch.int32)

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        bufs: dict[str, torch.Tensor] = {}
        tr = self.transformer
        for attr in ("_buf_embed", "_buf_pos_embed", "_buf_resid_final", "_buf_final_ln"):
            if hasattr(tr, attr):
                bufs[attr[5:]] = getattr(tr, attr)
        for i in range(tr.start_layer, tr.end_layer):
            block = tr.h[i]
            attn = block.attn
            for attr in ("_buf_resid_pre", "_buf_ln1", "_buf_attn_out",
                         "_buf_resid_mid", "_buf_ln2", "_buf_mlp_in", "_buf_mlp_out"):
                if hasattr(block, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(block, attr)
            v = getattr(block.mlp, "_buf_mlp_post", None)
            if v is not None:
                bufs[f"mlp_post_L{i}"] = v
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                if hasattr(attn, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(attn, attr)
        for attr in ("_buf_final_logits", "_buf_token_ids"):
            if hasattr(self, attr):
                bufs[attr[5:]] = getattr(self, attr)
        return bufs

    def get_hook_specs(self) -> list[HookSpec]:
        specs: list[HookSpec] = []
        tr = self.transformer

        specs.append(HookSpec(HOOK_TYPE_TOKEN_IDS, self.hook_token_ids))
        specs.append(HookSpec(HOOK_TYPE_EMBED, tr.hook_embed))
        specs.append(HookSpec(HOOK_TYPE_POS_EMBED, tr.hook_pos_embed))

        for i in range(tr.start_layer, tr.end_layer):
            block = tr.h[i]
            attn = block.attn
            specs.append(HookSpec(HOOK_TYPE_RESID_PRE, block.hook_resid_pre, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_LN1, block.hook_ln1, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_Q, attn.hook_q, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_K, attn.hook_k, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_V, attn.hook_v, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_Z, attn.hook_z, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_ATTN_OUT, block.hook_attn_out, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_RESID_MID, block.hook_resid_mid, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_LN2, block.hook_ln2, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_MLP_IN, block.hook_mlp_in, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_MLP_POST, block.mlp.hook_post, layer_no=i))
            specs.append(HookSpec(HOOK_TYPE_MLP_OUT, block.hook_mlp_out, layer_no=i))

        specs.append(HookSpec(HOOK_TYPE_RESID_FINAL, tr.hook_resid_final))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LN, tr.hook_final_ln))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LOGITS, self.hook_final_logits))

        return specs


def _add_transformer_prefix(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if not name.startswith("transformer.") and not name.startswith("lm_head"):
            name = "transformer." + name
        yield name, tensor
