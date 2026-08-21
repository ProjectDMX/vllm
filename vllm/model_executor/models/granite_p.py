# SPDX-License-Identifier: Apache-2.0
"""IBM Granite dense decoder with bounded DMI observation hooks."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
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
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteMLP,
    GraniteModel,
)
from .utils import PPMissingLayer, maybe_prefix


def _require_supported_granite_config(config) -> None:
    """Require the dense Granite scalar-multiplier contract we audited."""

    if getattr(config, "model_type", "granite") != "granite":
        raise NotImplementedError("DMI Granite support requires model_type=granite")
    if getattr(config, "hidden_act", None) != "silu":
        raise NotImplementedError("DMI Granite support requires hidden_act=silu")
    missing = [
        name
        for name in (
            "attention_multiplier",
            "embedding_multiplier",
            "residual_multiplier",
            "logits_scaling",
        )
        if getattr(config, name, None) is None
    ]
    if missing:
        raise NotImplementedError(
            "DMI Granite support requires explicit scalar fields: "
            + ", ".join(missing)
        )


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


class GranitePMLP(GraniteMLP):
    """Granite MLP with the post-activation boundary exposed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class GranitePAttention(GraniteAttention):
    """Granite attention with pre-RoPE Q/K/V and pre-o-proj Z hooks."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not any(
            hook.enabled
            for hook in (self.hook_q, self.hook_k, self.hook_v, self.hook_z)
        ):
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.hook_q.enabled:
            self.hook_q(q.unflatten(-1, (self.num_heads, self.head_dim)))
        if self.hook_k.enabled:
            self.hook_k(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        if self.hook_v.enabled:
            self.hook_v(v.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class GranitePDecoderLayer(GraniteDecoderLayer):
    """One Granite block preserving its scaled residual arithmetic."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer_hooks = (
            self.hook_resid_pre,
            self.hook_ln1,
            self.hook_attn_out,
            self.hook_resid_mid,
            self.hook_ln2,
            self.hook_mlp_in,
            self.hook_mlp_out,
        )
        if not any(hook.enabled for hook in layer_hooks):
            return super().forward(positions, hidden_states)

        residual = hidden_states
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        hidden_states = self.input_layernorm(hidden_states)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


def _instrument_granite_model(model: GraniteModel) -> "GranitePModel":
    """Attach hooks to the exact module tree created by upstream Granite."""

    _require_supported_granite_config(model.config)
    model.__class__ = GranitePModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = GranitePDecoderLayer
        layer.self_attn.__class__ = GranitePAttention
        layer.mlp.__class__ = GranitePMLP
        _add_hook_points(
            layer,
            (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ),
        )
        _add_hook_points(layer.self_attn, ("q", "k", "v", "z"))
        _add_hook_points(layer.mlp, ("post",))
    return model


class GranitePModel(GraniteModel):
    """Concrete compiled Granite backbone with model-wide hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_granite_model(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if not any(
            hook.enabled
            for hook in (self.hook_embed, self.hook_resid_final, self.hook_final_ln)
        ):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            hidden_states *= self.config.embedding_multiplier
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.hook_resid_final.enabled:
            self.hook_resid_final(hidden_states)
        hidden_states = self.norm(hidden_states)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
        return hidden_states


class GranitePForCausalLM(GraniteForCausalLM):
    """Dense Granite causal LM with a truthful DMI hook manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type=GranitePModel,
    ) -> None:
        config = vllm_config.model_config.hf_config
        _require_supported_granite_config(config)
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
            logit_scale = getattr(config, "logit_scale", 1.0)
            logit_scale /= config.logits_scaling
            self.logits_processor = LogitsProcessor(
                config.vocab_size,
                scale=logit_scale,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.hook_token_ids = HookPoint()
        self.hook_final_logits = HookPoint()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _layer_hook_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        attention = None if layer is None else layer.self_attn
        mlp = None if layer is None else layer.mlp

        def hook(module, name: str):
            return None if module is None else getattr(module, f"hook_{name}")

        def spec(hook_type: int, module, name: str) -> HookSpec:
            return HookSpec(
                hook_type,
                hook(module, name),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            )

        return [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
            spec(HOOK_TYPE_Q, attention, "q"),
            spec(HOOK_TYPE_K, attention, "k"),
            spec(HOOK_TYPE_V, attention, "v"),
            spec(HOOK_TYPE_Z, attention, "z"),
            spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
            spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
            spec(HOOK_TYPE_LN2, layer, "ln2"),
            spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
            spec(HOOK_TYPE_MLP_POST, mlp, "post"),
            spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"),
        ]

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
    "GranitePAttention",
    "GranitePDecoderLayer",
    "GranitePForCausalLM",
    "GranitePMLP",
    "GranitePModel",
    "_instrument_granite_model",
    "_require_supported_granite_config",
]
