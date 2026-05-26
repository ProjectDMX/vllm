# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen2MoE model with monitoring hooks.

V1 scope:
- ordinary dense hooks on the MoE model
- token-major MoE routing-side hooks:
  - router_logits
  - topk_ids
  - topk_weights

Deliberately not included in this wrapper:
- expert-local post-dispatch / pre-combine hooks
"""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors

from monitoring.hook_points import HookPoint
from monitoring.ring_transport import (
    HOOK_TYPE_ATTN_OUT,
    HOOK_TYPE_EMBED,
    HOOK_TYPE_FINAL_LN,
    HOOK_TYPE_FINAL_LOGITS,
    HOOK_TYPE_K,
    HOOK_TYPE_LN1,
    HOOK_TYPE_LN2,
    HOOK_TYPE_MLP_IN,
    HOOK_TYPE_MLP_OUT,
    HOOK_TYPE_MLP_POST,
    HOOK_TYPE_Q,
    HOOK_TYPE_RESID_FINAL,
    HOOK_TYPE_RESID_MID,
    HOOK_TYPE_RESID_PRE,
    HOOK_TYPE_ROUTER_LOGITS,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_TOPK_IDS,
    HOOK_TYPE_TOPK_WEIGHTS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .interfaces import SupportsLoRA, SupportsPP
from .qwen2_moe import (
    Qwen2MoeAttention as _Qwen2MoeAttention,
    Qwen2MoeDecoderLayer as _Qwen2MoeDecoderLayer,
    Qwen2MoeForCausalLM as _Qwen2MoeForCausalLM,
    Qwen2MoeMLP as _Qwen2MoeMLP,
    Qwen2MoeModel as _Qwen2MoeModel,
    Qwen2MoeSparseMoeBlock as _Qwen2MoeSparseMoeBlock,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)


class Qwen2MoeMLP(_Qwen2MoeMLP):
    """Dense MLP variant with hook_post."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_post = HookPoint()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        if hasattr(self, "_buf_mlp_post"):
            self._buf_mlp_post[:x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        if self.expert_gate is not None:
            x = F.sigmoid(self.expert_gate(x)[0]) * x
        return x


class Qwen2MoeSparseMoeBlock(_Qwen2MoeSparseMoeBlock):
    """Sparse MoE block with routing-side hook points."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_router_logits = HookPoint()
        self.hook_topk_ids = HookPoint()
        self.hook_topk_weights = HookPoint()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states)
        self.hook_router_logits(router_logits)
        if hasattr(self, "_buf_router_logits"):
            self._buf_router_logits[:router_logits.shape[0]].copy_(router_logits)

        # V1 routing hooks: observe token-major routing tensors directly.
        # We intentionally do not wire expert-local post-dispatch tensors here.
        topk_weights, topk_ids = self.experts.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        self.hook_topk_ids(topk_ids)
        self.hook_topk_weights(topk_weights)
        if hasattr(self, "_buf_topk_ids"):
            self._buf_topk_ids[:topk_ids.shape[0]].copy_(topk_ids)
        if hasattr(self, "_buf_topk_weights"):
            self._buf_topk_weights[:topk_weights.shape[0]].copy_(topk_weights)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(_Qwen2MoeAttention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
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

        q_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        v_head = v.view(*v.shape[:-1], v.shape[-1] // self.head_dim, self.head_dim)
        self.hook_q(q_head)
        self.hook_k(k_head)
        self.hook_v(v_head)
        if hasattr(self, "_buf_q"):
            self._buf_q[:q_head.shape[0]].copy_(q_head)
        if hasattr(self, "_buf_k"):
            self._buf_k[:k_head.shape[0]].copy_(k_head)
        if hasattr(self, "_buf_v"):
            self._buf_v[:v_head.shape[0]].copy_(v_head)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        if hasattr(self, "_buf_z"):
            self._buf_z[:attn_output.shape[0]].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(_Qwen2MoeDecoderLayer):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
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
        if residual is None:
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(hidden_states)
            if hasattr(self, "_buf_resid_pre"):
                self._buf_resid_pre[:hidden_states.shape[0]].copy_(hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(hidden_states + residual)
            if hasattr(self, "_buf_resid_pre"):
                self._buf_resid_pre[:hidden_states.shape[0]].copy_(hidden_states + residual)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self.hook_ln1(hidden_states)
        if hasattr(self, "_buf_ln1"):
            self._buf_ln1[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        self.hook_attn_out(hidden_states)
        if hasattr(self, "_buf_attn_out"):
            self._buf_attn_out[:hidden_states.shape[0]].copy_(hidden_states)

        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states + residual)
        if hasattr(self, "_buf_resid_mid"):
            self._buf_resid_mid[:hidden_states.shape[0]].copy_(hidden_states + residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        self.hook_ln2(hidden_states)
        if hasattr(self, "_buf_ln2"):
            self._buf_ln2[:hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        if hasattr(self, "_buf_mlp_in"):
            self._buf_mlp_in[:hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(hidden_states)
        if hasattr(self, "_buf_mlp_out"):
            self._buf_mlp_out[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(_Qwen2MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from .qwen2_moe import make_empty_intermediate_tensors_factory

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
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
            if hasattr(self, "_buf_embed"):
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
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states + residual)
        if hasattr(self, "_buf_resid_final"):
            self._buf_resid_final[:hidden_states.shape[0]].copy_(hidden_states + residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        if hasattr(self, "_buf_final_ln"):
            self._buf_final_ln[:hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states


class Qwen2MoePForCausalLM(_Qwen2MoeForCausalLM, SupportsPP, SupportsLoRA):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.hook_final_logits = HookPoint()
        self.hook_token_ids = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            if hasattr(self, "_buf_token_ids"):
                self._buf_token_ids[:input_ids.shape[0]].copy_(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        if hasattr(self, "_buf_final_logits"):
            self._buf_final_logits[:logits.shape[0]].copy_(logits)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def get_hook_specs(self) -> list[HookSpec]:
        specs: list[HookSpec] = []
        m = self.model
        specs.append(
            HookSpec(
                HOOK_TYPE_TOKEN_IDS,
                self.hook_token_ids,
                dtype=torch.int32,
                dim0_is_actual_tokens=True,
            )
        )
        specs.append(HookSpec(HOOK_TYPE_EMBED, m.hook_embed, dim0_is_actual_tokens=True))
        for i in range(m.start_layer, m.end_layer):
            layer = m.layers[i]
            if isinstance(layer, PPMissingLayer):
                continue
            attn = layer.self_attn
            specs.append(HookSpec(HOOK_TYPE_RESID_PRE, layer.hook_resid_pre, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_LN1, layer.hook_ln1, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_Q, attn.hook_q, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_K, attn.hook_k, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_V, attn.hook_v, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_Z, attn.hook_z, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_ATTN_OUT, layer.hook_attn_out, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_RESID_MID, layer.hook_resid_mid, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_LN2, layer.hook_ln2, layer_no=i, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_MLP_IN, layer.hook_mlp_in, layer_no=i, dim0_is_actual_tokens=True))
            if hasattr(layer.mlp, "hook_post"):
                specs.append(HookSpec(HOOK_TYPE_MLP_POST, layer.mlp.hook_post, layer_no=i, dim0_is_actual_tokens=True))
            if hasattr(layer.mlp, "hook_router_logits"):
                specs.append(HookSpec(HOOK_TYPE_ROUTER_LOGITS, layer.mlp.hook_router_logits, layer_no=i, dim0_is_actual_tokens=True))
                specs.append(HookSpec(HOOK_TYPE_TOPK_IDS, layer.mlp.hook_topk_ids, layer_no=i, dtype=torch.int32, dim0_is_actual_tokens=True))
                specs.append(HookSpec(HOOK_TYPE_TOPK_WEIGHTS, layer.mlp.hook_topk_weights, layer_no=i, dtype=torch.float32, dim0_is_actual_tokens=True))
            specs.append(HookSpec(HOOK_TYPE_MLP_OUT, layer.hook_mlp_out, layer_no=i, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_RESID_FINAL, m.hook_resid_final, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LN, m.hook_final_ln, dim0_is_actual_tokens=True))
        specs.append(HookSpec(HOOK_TYPE_FINAL_LOGITS, self.hook_final_logits))
        return specs
