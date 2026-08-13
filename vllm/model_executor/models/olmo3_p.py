# SPDX-License-Identifier: Apache-2.0
"""OLMo 3 dense decoder with bounded DMI observation hooks."""

from __future__ import annotations

from collections.abc import Mapping
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

from .olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3Model,
)
from .utils import PPMissingLayer, maybe_prefix


def _require_supported_olmo3_config(config) -> None:
    """Fail closed outside the audited dense sliding/full-attention path."""

    if getattr(config, "model_type", None) != "olmo3":
        raise NotImplementedError("DMI OLMo 3 support requires model_type=olmo3")
    if getattr(config, "hidden_act", None) != "silu":
        raise NotImplementedError("DMI OLMo 3 support requires hidden_act=silu")
    if getattr(config, "attention_bias", False):
        raise NotImplementedError("DMI OLMo 3 support requires attention_bias=false")

    layer_types = tuple(getattr(config, "layer_types", ()) or ())
    if len(layer_types) != int(config.num_hidden_layers):
        raise ValueError("OLMo 3 layer_types does not cover every decoder layer")
    unknown = sorted(set(layer_types) - {"sliding_attention", "full_attention"})
    if unknown:
        raise NotImplementedError(f"DMI does not support OLMo 3 layer types {unknown}")
    if "sliding_attention" in layer_types:
        sliding_window = getattr(config, "sliding_window", None)
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise NotImplementedError(
                "DMI OLMo 3 sliding attention requires a positive sliding_window"
            )

    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, Mapping):
        raise NotImplementedError("DMI OLMo 3 support requires rope_parameters")
    missing_rope = [
        layer_type
        for layer_type in sorted(set(layer_types))
        if not isinstance(rope_parameters.get(layer_type), Mapping)
    ]
    if missing_rope:
        raise NotImplementedError(
            "DMI OLMo 3 support requires per-attention-type RoPE parameters: "
            + ", ".join(missing_rope)
        )


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


class Olmo3PMLP(Olmo3MLP):
    """OLMo 3 MLP with the post-activation boundary exposed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class Olmo3PAttention(Olmo3Attention):
    """OLMo 3 attention with post-QK-norm, pre-RoPE Q/K hooks."""

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
        q, k = self._apply_qk_norm(q, k)
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


class Olmo3PDecoderLayer(Olmo3DecoderLayer):
    """OLMo 3 post-norm block with hooks in exact execution order."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer_hooks = (
            self.hook_resid_pre,
            self.hook_attn_out,
            self.hook_ln1,
            self.hook_resid_mid,
            self.hook_mlp_in,
            self.hook_mlp_out,
            self.hook_ln2,
        )
        if not any(hook.enabled for hook in layer_hooks):
            return super().forward(positions, hidden_states)

        residual = hidden_states
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        hidden_states = hidden_states + residual
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(hidden_states)

        residual = hidden_states
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _instrument_olmo3_model(model: Olmo3Model) -> "Olmo3PModel":
    """Attach DMI behavior to the exact module tree built by upstream."""

    _require_supported_olmo3_config(model.config)
    model.__class__ = Olmo3PModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.__class__ = Olmo3PDecoderLayer
        layer.self_attn.__class__ = Olmo3PAttention
        layer.mlp.__class__ = Olmo3PMLP
        _add_hook_points(
            layer,
            (
                "resid_pre",
                "attn_out",
                "ln1",
                "resid_mid",
                "mlp_in",
                "mlp_out",
                "ln2",
            ),
        )
        _add_hook_points(layer.self_attn, ("q", "k", "v", "z"))
        _add_hook_points(layer.mlp, ("post",))
    return model


class Olmo3PModel(Olmo3Model):
    """Concrete compiled OLMo 3 backbone with model-wide hooks."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_olmo3_model(self)

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
                else self.embed_tokens(input_ids)
            )
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

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


class Olmo3PForCausalLM(Olmo3ForCausalLM):
    """Dense OLMo 3 causal LM with a truthful DMI hook manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type=Olmo3PModel,
    ) -> None:
        config = vllm_config.model_config.hf_config
        _require_supported_olmo3_config(config)
        nn.Module.__init__(self)
        self.config = config
        self.model = model_type(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
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
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

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

        # OLMo 3 is post-norm. This order deliberately differs from Llama:
        # the first norm fires after attention and the second after the MLP.
        return [
            spec(HOOK_TYPE_RESID_PRE, layer, "resid_pre"),
            spec(HOOK_TYPE_Q, attention, "q"),
            spec(HOOK_TYPE_K, attention, "k"),
            spec(HOOK_TYPE_V, attention, "v"),
            spec(HOOK_TYPE_Z, attention, "z"),
            spec(HOOK_TYPE_ATTN_OUT, layer, "attn_out"),
            spec(HOOK_TYPE_LN1, layer, "ln1"),
            spec(HOOK_TYPE_RESID_MID, layer, "resid_mid"),
            spec(HOOK_TYPE_MLP_IN, layer, "mlp_in"),
            spec(HOOK_TYPE_MLP_POST, mlp, "post"),
            spec(HOOK_TYPE_MLP_OUT, layer, "mlp_out"),
            spec(HOOK_TYPE_LN2, layer, "ln2"),
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
    "Olmo3PAttention",
    "Olmo3PDecoderLayer",
    "Olmo3PForCausalLM",
    "Olmo3PMLP",
    "Olmo3PModel",
    "_instrument_olmo3_model",
    "_require_supported_olmo3_config",
]
