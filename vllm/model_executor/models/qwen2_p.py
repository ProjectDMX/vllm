# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen2 model with DMI monitoring hooks."""

from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

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
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .qwen2 import (
    Qwen2Attention as _Qwen2Attention,
    Qwen2ForCausalLM as _Qwen2ForCausalLM,
    Qwen2MLP as _Qwen2MLP,
    Qwen2Model as _Qwen2Model,
)
from .utils import PPMissingLayer, maybe_prefix


class Qwen2MLP(_Qwen2MLP):
    """Qwen2 MLP with a post-activation, pre-down-projection hook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(_Qwen2Attention):
    """Qwen2 attention with per-head Q/K/V and attention-output hooks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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

        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            q_by_head = self.q_norm(q_by_head)
            k_by_head = self.k_norm(k_by_head)

        self.hook_q(q_by_head)
        self.hook_k(k_by_head)
        v_by_head = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)
        self.hook_v(v_by_head)

        q = q_by_head.reshape(q.shape)
        k = k_by_head.reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 decoder layer with DMI residual, norm, attention, and MLP hooks."""

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=getattr(config, "qk_norm", False),
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = Qwen2MLP(
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
        if residual is None:
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(hidden_states)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(hidden_states + residual)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self.hook_ln1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        self.hook_attn_out(hidden_states)

        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states + residual)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        self.hook_ln2(hidden_states)
        self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": -1,
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class Qwen2Model(_Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen2DecoderLayer,
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
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            self.hook_embed(hidden_states)
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
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states + residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen2PForCausalLM(_Qwen2ForCausalLM):
    """Qwen2/Qwen2.5 causal LM exposing DMI's canonical hook inventory."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        return logits

    def _get_layer_hook_specs(
        self, layer_no: int, layer: Qwen2DecoderLayer | None
    ) -> list[HookSpec]:
        attn = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.mlp

        def hook(module: Any, name: str):
            return None if module is None else getattr(module, name)

        common = {"layer_no": layer_no, "dim0_is_actual_tokens": True}
        return [
            HookSpec(HOOK_TYPE_RESID_PRE, hook(layer, "hook_resid_pre"), **common),
            HookSpec(HOOK_TYPE_LN1, hook(layer, "hook_ln1"), **common),
            HookSpec(HOOK_TYPE_Q, hook(attn, "hook_q"), **common),
            HookSpec(HOOK_TYPE_K, hook(attn, "hook_k"), **common),
            HookSpec(HOOK_TYPE_V, hook(attn, "hook_v"), **common),
            HookSpec(HOOK_TYPE_Z, hook(attn, "hook_z"), **common),
            HookSpec(HOOK_TYPE_ATTN_OUT, hook(layer, "hook_attn_out"), **common),
            HookSpec(HOOK_TYPE_RESID_MID, hook(layer, "hook_resid_mid"), **common),
            HookSpec(HOOK_TYPE_LN2, hook(layer, "hook_ln2"), **common),
            HookSpec(HOOK_TYPE_MLP_IN, hook(layer, "hook_mlp_in"), **common),
            HookSpec(HOOK_TYPE_MLP_POST, hook(mlp, "hook_post"), **common),
            HookSpec(HOOK_TYPE_MLP_OUT, hook(layer, "hook_mlp_out"), **common),
        ]

    def get_hook_specs(self, *, model_wide: bool = False) -> list[HookSpec]:
        model = self.model
        specs = [
            HookSpec(
                HOOK_TYPE_TOKEN_IDS,
                None if model_wide else self.hook_token_ids,
                dtype=torch.int32,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_EMBED,
                None if model_wide else model.hook_embed,
                dim0_is_actual_tokens=True,
            ),
        ]
        layer_indices = (
            range(len(model.layers))
            if model_wide
            else range(model.start_layer, model.end_layer)
        )
        for idx in layer_indices:
            layer = None if model_wide else model.layers[idx]
            if layer is not None and isinstance(layer, PPMissingLayer):
                continue
            specs.extend(self._get_layer_hook_specs(idx, layer))
        specs.extend(
            [
                HookSpec(
                    HOOK_TYPE_RESID_FINAL,
                    None if model_wide else model.hook_resid_final,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LN,
                    None if model_wide else model.hook_final_ln,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_FINAL_LOGITS,
                    None if model_wide else self.hook_final_logits,
                ),
            ]
        )
        return specs
