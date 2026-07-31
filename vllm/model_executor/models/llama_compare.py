# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Llama compare model for transport correctness testing.
# Copy of llama_p.py with .copy_() capture into pre-allocated buffers
# added after each HookPoint, so a single compiled graph fires both the
# ring transport and the bitwise-comparison reference.
"""LLaMA compare model: HookPoints + .copy_() in same compiled graph."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
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
    HOOK_TYPE_MLP_IN, HOOK_TYPE_MLP_OUT, HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_EMBED, HOOK_TYPE_FINAL_LN, HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        self.head_dim = head_dim or self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            if hasattr(config, "target_layer_count"):
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types)
            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

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
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

        # Hook points
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
        # Hook Q, K, V in head-view shape; .copy_() into compare buffer
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        self.hook_q(q_by_head)
        self._buf_q[:q_by_head.shape[0]].copy_(q_by_head)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        self.hook_k(k_by_head)
        self._buf_k[:k_by_head.shape[0]].copy_(k_by_head)
        v_by_head = v.view(*v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim)
        self.hook_v(v_by_head)
        self._buf_v[:v_by_head.shape[0]].copy_(v_by_head)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:attn_output.shape[0]].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Hook points
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

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        rm = residual
        self.hook_resid_mid(rm)
        self._buf_resid_mid[:rm.shape[0]].copy_(rm)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    shape_invariants=lambda input_ids, positions, intermediate_tensors=None, inputs_embeds=None: (
        torch._check(positions.size()[0] == input_ids.size()[0]) if input_ids is not None else None
    ),
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Hook points
        self.hook_embed = HookPoint()
        if get_pp_group().is_last_rank:
            self.hook_resid_final = HookPoint()
            self.hook_final_ln = HookPoint()
        else:
            self.hook_resid_final = PPMissingLayer()
            self.hook_final_ln = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

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
        return hidden_states


class LlamaCompareForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
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

        self.model = LlamaModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

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

    def allocate_compare_buffers(self, max_len: int, vllm_config: VllmConfig) -> None:
        """Allocate D2D capture buffers for bitwise comparison against ring transport."""
        config = self.config
        H = config.hidden_size
        nh = config.num_attention_heads
        nkv = getattr(config, "num_key_value_heads", nh)
        hd = getattr(config, "head_dim", None) or H // nh
        V = config.vocab_size
        dtype = vllm_config.model_config.dtype

        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        tp = get_tensor_model_parallel_world_size()
        nh_tp = nh // tp
        nkv_tp = max(1, nkv // tp)

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
            if isinstance(layer, PPMissingLayer):
                continue
            attn = layer.self_attn
            for attr in ("_buf_resid_pre", "_buf_ln1", "_buf_attn_out",
                         "_buf_resid_mid", "_buf_ln2", "_buf_mlp_in", "_buf_mlp_out"):
                if hasattr(layer, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(layer, attr)
            for attr in ("_buf_q", "_buf_k", "_buf_v", "_buf_z"):
                if hasattr(attn, attr):
                    bufs[f"{attr[5:]}_L{i}"] = getattr(attn, attr)
        for attr in ("_buf_final_logits", "_buf_token_ids"):
            if hasattr(self, attr):
                bufs[attr[5:]] = getattr(self, attr)
        return bufs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.model.quant_config is not None and (
                scale_name := self.model.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name or "zero_point" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _get_layer_hook_specs(self, layer_no: int, layer) -> list[HookSpec]:
        attn = None if layer is None else layer.self_attn

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
