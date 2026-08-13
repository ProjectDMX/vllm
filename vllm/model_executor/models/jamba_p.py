# SPDX-License-Identifier: Apache-2.0
"""Dense Jamba attention/Mamba hybrid with bounded DMI hooks."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
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
    HOOK_TYPE_SSM_IN,
    HOOK_TYPE_SSM_OUT,
    HOOK_TYPE_TOKEN_IDS,
    HOOK_TYPE_V,
    HOOK_TYPE_Z,
    HookSpec,
)

from .jamba import (
    JambaAttentionDecoderLayer,
    JambaForCausalLM,
    JambaMLP,
    JambaMambaDecoderLayer,
    JambaModel,
)
from .utils import PPMissingLayer, maybe_prefix


def _require_dense_jamba_config(config) -> None:
    """Fail closed rather than applying a dense verdict to Jamba MoE."""

    layer_kinds = tuple(config.layers_block_type)
    if len(layer_kinds) != int(config.num_hidden_layers):
        raise ValueError("Jamba layers_block_type does not cover every layer")
    unknown_kinds = sorted(set(layer_kinds) - {"attention", "mamba"})
    if unknown_kinds:
        raise NotImplementedError(
            f"DMI does not support Jamba layer kinds {unknown_kinds}"
        )

    layer_experts = tuple(int(value) for value in config.layers_num_experts)
    if len(layer_experts) != int(config.num_hidden_layers):
        raise ValueError("Jamba layers_num_experts does not cover every layer")
    moe_layers = [
        layer_no
        for layer_no, num_experts in enumerate(layer_experts)
        if num_experts != 1
    ]
    if moe_layers:
        raise NotImplementedError(
            "DMI dense Jamba support does not include MoE layers; "
            f"expert layers are present at {moe_layers}"
        )


class JambaPMLP(JambaMLP):
    """Jamba dense MLP with an observational post-activation hook."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.hook_post.enabled:
            return super().forward(x)
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        self.hook_post(x)
        x, _ = self.down_proj(x)
        return x


class JambaPAttentionDecoderLayer(JambaAttentionDecoderLayer):
    """One Jamba attention layer with canonical and common hooks."""

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if not (
            self.hook_q.enabled
            or self.hook_k.enabled
            or self.hook_v.enabled
            or self.hook_z.enabled
        ):
            return super().self_attention(
                positions=positions,
                hidden_states=hidden_states,
                **kwargs,
            )

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.hook_q.enabled:
            self.hook_q(q.unflatten(-1, (self.num_heads, self.head_dim)))
        if self.hook_k.enabled:
            self.hook_k(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        if self.hook_v.enabled:
            self.hook_v(v.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        attn_output = self.attn(q, k, v)
        if self.hook_z.enabled:
            self.hook_z(attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self.hook_resid_pre.enabled
            or self.hook_ln1.enabled
            or self.hook_attn_out.enabled
            or self.hook_resid_mid.enabled
            or self.hook_ln2.enabled
            or self.hook_mlp_in.enabled
            or self.hook_mlp_out.enabled
        ):
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **kwargs,
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)

        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        if self.hook_attn_out.enabled:
            self.hook_attn_out(hidden_states)
        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return hidden_states, residual


class JambaPMambaDecoderLayer(JambaMambaDecoderLayer):
    """One Jamba Mamba1 layer with typed branch boundaries."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self.hook_resid_pre.enabled
            or self.hook_ln1.enabled
            or self.hook_ssm_in.enabled
            or self.hook_ssm_out.enabled
            or self.hook_resid_mid.enabled
            or self.hook_ln2.enabled
            or self.hook_mlp_in.enabled
            or self.hook_mlp_out.enabled
        ):
            return super().forward(
                hidden_states=hidden_states,
                residual=residual,
                **kwargs,
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.hook_resid_pre.enabled:
            self.hook_resid_pre(residual)
        if self.hook_ln1.enabled:
            self.hook_ln1(hidden_states)
        if self.hook_ssm_in.enabled:
            self.hook_ssm_in(hidden_states)

        output = torch.empty_like(hidden_states)
        self.mamba(hidden_states, output)
        if self.hook_ssm_out.enabled:
            self.hook_ssm_out(output)
        hidden_states, residual = self.pre_ff_layernorm(output, residual)
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(residual)
        if self.hook_ln2.enabled:
            self.hook_ln2(hidden_states)
        if self.hook_mlp_in.enabled:
            self.hook_mlp_in(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.hook_mlp_out.enabled:
            self.hook_mlp_out(hidden_states)
        return hidden_states, residual


class JambaPModel(JambaModel):
    """Jamba backbone with exact layer-kind-specific observation."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if not (
            self.hook_embed.enabled
            or self.hook_resid_final.enabled
            or self.hook_final_ln.enabled
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
            if self.hook_embed.enabled:
                self.hook_embed(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, resid_final = self.final_layernorm(hidden_states, residual)
        if self.hook_resid_final.enabled:
            self.hook_resid_final(resid_final)
        if self.hook_final_ln.enabled:
            self.hook_final_ln(hidden_states)
        return hidden_states


def _add_hook_points(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        setattr(module, f"hook_{name}", HookPoint())


def _instrument_upstream_jamba_model(model: JambaModel) -> JambaPModel:
    """Attach DMI behavior to the exact tree built by upstream Jamba."""

    _require_dense_jamba_config(model.config)
    model.__class__ = JambaPModel
    _add_hook_points(model, ("embed", "resid_final", "final_ln"))
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.feed_forward.__class__ = JambaPMLP
        _add_hook_points(layer.feed_forward, ("post",))
        common = (
            "resid_pre",
            "ln1",
            "resid_mid",
            "ln2",
            "mlp_in",
            "mlp_out",
        )
        if model.config.layers_block_type[layer_no] == "attention":
            layer.__class__ = JambaPAttentionDecoderLayer
            _add_hook_points(
                layer,
                common + ("q", "k", "v", "z", "attn_out"),
            )
        else:
            layer.__class__ = JambaPMambaDecoderLayer
            _add_hook_points(layer, common + ("ssm_in", "ssm_out"))
    return model


class JambaPForCausalLM(JambaForCausalLM):
    """Dense Jamba causal LM with a heterogeneous capability manifest."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model_type=JambaPModel,
    ) -> None:
        _require_dense_jamba_config(vllm_config.model_config.hf_config)
        if model_type is JambaPModel:
            # Preserve upstream parameter creation and registration exactly;
            # attach only observational modules after construction.
            JambaForCausalLM.__init__(
                self,
                vllm_config=vllm_config,
                prefix=prefix,
            )
            self.model = _instrument_upstream_jamba_model(self.model)
        else:
            # The compare oracle must construct its backbone subclass
            # directly.  JambaModel's compilation decorator binds the
            # concrete forward method during __init__; changing __class__
            # afterward would leave model-wide hooks outside that method.
            config = vllm_config.model_config.hf_config
            nn.Module.__init__(self)
            self.config = config
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config
            self.scheduler_config = vllm_config.scheduler_config
            self.model = model_type(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
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
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if (
            input_ids is not None
            and get_pp_group().is_first_rank
            and self.hook_token_ids.enabled
        ):
            self.hook_token_ids(input_ids)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None and self.hook_final_logits.enabled:
            self.hook_final_logits(logits)
        return logits

    @staticmethod
    def _common_layer_specs(
        layer_no: int,
        layer: nn.Module | None,
    ) -> list[HookSpec]:
        mlp = None if layer is None else layer.feed_forward

        def hook(module, name: str):
            return None if module is None else getattr(module, name)

        return [
            HookSpec(
                HOOK_TYPE_RESID_PRE,
                hook(layer, "hook_resid_pre"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_LN1,
                hook(layer, "hook_ln1"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_RESID_MID,
                hook(layer, "hook_resid_mid"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_LN2,
                hook(layer, "hook_ln2"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_MLP_IN,
                hook(layer, "hook_mlp_in"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_MLP_POST,
                hook(mlp, "hook_post"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_MLP_OUT,
                hook(layer, "hook_mlp_out"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
        ]

    @staticmethod
    def _operator_specs(
        layer_no: int,
        layer: nn.Module | None,
        is_attention: bool,
    ) -> list[HookSpec]:
        def hook(name: str):
            return None if layer is None else getattr(layer, name)

        if not is_attention:
            return [
                HookSpec(
                    HOOK_TYPE_SSM_IN,
                    hook("hook_ssm_in"),
                    layer_no=layer_no,
                    dim0_is_actual_tokens=True,
                ),
                HookSpec(
                    HOOK_TYPE_SSM_OUT,
                    hook("hook_ssm_out"),
                    layer_no=layer_no,
                    dim0_is_actual_tokens=True,
                ),
            ]
        return [
            HookSpec(
                HOOK_TYPE_Q,
                hook("hook_q"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_K,
                hook("hook_k"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_V,
                hook("hook_v"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_Z,
                hook("hook_z"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
            HookSpec(
                HOOK_TYPE_ATTN_OUT,
                hook("hook_attn_out"),
                layer_no=layer_no,
                dim0_is_actual_tokens=True,
            ),
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
        for layer_no, layer_kind in enumerate(self.config.layers_block_type):
            layer = None
            if not model_wide and model.start_layer <= layer_no < model.end_layer:
                candidate = model.layers[layer_no]
                if not isinstance(candidate, PPMissingLayer):
                    layer = candidate
            common_specs = self._common_layer_specs(layer_no, layer)
            operator_specs = self._operator_specs(
                layer_no,
                layer,
                layer_kind == "attention",
            )
            specs.extend(common_specs[:2])
            specs.extend(operator_specs)
            specs.extend(common_specs[2:])
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
    "JambaPAttentionDecoderLayer",
    "JambaPForCausalLM",
    "JambaPMLP",
    "JambaPMambaDecoderLayer",
    "JambaPModel",
    "_instrument_upstream_jamba_model",
    "_require_dense_jamba_config",
]
