# SPDX-License-Identifier: Apache-2.0
"""Falcon-H1 hybrid attention/Mamba model with bounded DMI hooks."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
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
    HOOK_TYPE_SSM_IN,
    HOOK_TYPE_SSM_OUT,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .falcon_h1 import (
    FalconH1AttentionDecoderLayer,
    FalconH1ForCausalLM,
    FalconH1MLP,
    FalconH1Model,
    FalconH1SSMDecoderLayer,
)
from .utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class FalconH1PMLP(FalconH1MLP):
    """Falcon-H1 MLP with a post-activation branch hook."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hook_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x[:, : self.intermediate_size // self.tp_size] *= (
            self.gate_multiplier
        )
        x = self.act_fn(x)
        if self.hook_post.enabled:
            self.hook_post(x)
        x, _ = self.down_proj(x)
        x = x * self.down_multiplier
        return x


class FalconH1PAttentionDecoderLayer(FalconH1AttentionDecoderLayer):
    """Falcon-H1 attention branch with canonical Q/K/V/Z hooks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        k = k * self.key_multiplier

        if self.hook_q.enabled:
            self.hook_q(q.unflatten(-1, (self.num_heads, self.head_dim)))
        if self.hook_k.enabled:
            self.hook_k(
                k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            )
        if self.hook_v.enabled:
            self.hook_v(
                v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            )

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class FalconH1PParallelHybrid(nn.Module):
    """One Falcon-H1 parallel attention/Mamba block with typed hooks."""

    def __init__(
        self,
        config,
        layer_idx: int,
        model_config=None,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        *,
        attention_type: type[nn.Module] = FalconH1PAttentionDecoderLayer,
        mlp_type: type[nn.Module] = FalconH1PMLP,
    ) -> None:
        super().__init__()
        self.self_attn = attention_type(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

        ssm_layer_idx = config.num_hidden_layers + layer_idx
        ssm_prefix = prefix.split(".")[0] + f".{ssm_layer_idx}"
        self.mamba = FalconH1SSMDecoderLayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=ssm_prefix,
        )
        self.ssm_out_multiplier = config.ssm_out_multiplier
        self.ssm_in_multiplier = config.ssm_in_multiplier
        self.attention_in_multiplier = config.attention_in_multiplier
        self.attn_out_multiplier = config.attention_out_multiplier

        self.feed_forward = mlp_type(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_ff_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hook_resid_pre = HookPoint()
        self.hook_ln1 = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_ssm_in = HookPoint()
        self.hook_ssm_out = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_ln2 = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        hidden_states = self.input_layernorm(hidden_states)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)

        attn_hidden, _ = self.self_attn(
            positions=positions,
            hidden_states=(
                hidden_states * self.attention_in_multiplier
            ),
            residual=residual,
            **kwargs,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(
                attn_hidden * self.attn_out_multiplier
            )

        if self.hook_ssm_in.enabled:
            self.hook_ssm_in(
                hidden_states * self.ssm_in_multiplier
            )
        ssm_hidden, _ = self.mamba(
            hidden_states=hidden_states * self.ssm_in_multiplier,
            residual=residual,
            **kwargs,
        )
        if self.hook_ssm_out.enabled:
            self.hook_ssm_out(
                ssm_hidden * self.ssm_out_multiplier
            )

        hidden_states = (attn_hidden * self.attn_out_multiplier) + (
            ssm_hidden * self.ssm_out_multiplier
        )
        hidden_states = hidden_states + residual
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return residual + hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    }
)
class FalconH1PModel(FalconH1Model):
    """Falcon-H1 backbone using the DMI hybrid block factory."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = FalconH1PParallelHybrid,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
            self.embedding_multiplier = config.embedding_multiplier
        else:
            self.embed_tokens = PPMissingLayer()
            self.embedding_multiplier = 1.0

        def get_layer(prefix: str) -> nn.Module:
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return layer_type(
                config,
                layer_idx,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        if get_pp_group().is_last_rank:
            self.final_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.final_layernorm = PPMissingLayer()

        self.hook_embed = HookPoint()
        self.hook_resid_final = HookPoint()
        self.hook_final_ln = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds * self.embedding_multiplier
            else:
                hidden_states = (
                    self.embed_input_ids(input_ids)
                    * self.embedding_multiplier
                )
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(
            self.layers, self.start_layer, self.end_layer
        ):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
        return hidden_states


class FalconH1PForCausalLM(FalconH1ForCausalLM):
    """Falcon-H1 causal LM with a hybrid hook capability manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type: type[nn.Module] = FalconH1PModel,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.tie_word_embeddings = config.tie_word_embeddings

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.lm_head_multiplier = config.lm_head_multiplier
            if self.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens
                )
            from vllm.model_executor.layers.logits_processor import (
                LogitsProcessor,
            )

            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                config.vocab_size,
                scale=config.lm_head_multiplier,
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.feed_forward

        def hook(module, name: str):
            return None if module is None else getattr(module, name)

        specs = [
            HookSpec(HOOK_TYPE_RESID_PRE, hook(layer, "hook_resid_pre"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN1, hook(layer, "hook_ln1"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_Q, hook(attention, "hook_q"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_K, hook(attention, "hook_k"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_V, hook(attention, "hook_v"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_Z, hook(attention, "hook_z"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_ATTN_OUT, hook(layer, "hook_attn_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_SSM_IN, hook(layer, "hook_ssm_in"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_SSM_OUT, hook(layer, "hook_ssm_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_RESID_MID, hook(layer, "hook_resid_mid"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_LN2, hook(layer, "hook_ln2"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_IN, hook(layer, "hook_mlp_in"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_POST, hook(mlp, "hook_post"), layer_no=layer_no, dim0_is_actual_tokens=True),
            HookSpec(HOOK_TYPE_MLP_OUT, hook(layer, "hook_mlp_out"), layer_no=layer_no, dim0_is_actual_tokens=True),
        ]
        return specs

    def get_hook_specs(self, model_wide: bool = False) -> list[HookSpec]:
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
        for layer_no in range(self.config.num_hidden_layers):
            layer = None
            if not model_wide and model.start_layer <= layer_no < model.end_layer:
                candidate = model.layers[layer_no]
                if not isinstance(candidate, PPMissingLayer):
                    layer = candidate
            specs.extend(self._layer_hook_specs(layer_no, layer))
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


__all__ = [
    "FalconH1PAttentionDecoderLayer",
    "FalconH1PForCausalLM",
    "FalconH1PMLP",
    "FalconH1PModel",
    "FalconH1PParallelHybrid",
]
