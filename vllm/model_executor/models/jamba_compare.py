# SPDX-License-Identifier: Apache-2.0
"""Dense Jamba DMI model with independent transport-reference buffers."""

from __future__ import annotations

from itertools import islice

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.sequence import IntermediateTensors

from .jamba_p import (
    JambaPAttentionDecoderLayer,
    JambaPForCausalLM,
    JambaPMLP,
    JambaPMambaDecoderLayer,
    JambaPModel,
    _instrument_upstream_jamba_model,
)
from .utils import PPMissingLayer


class JambaCompareMLP(JambaPMLP):
    """Capture the exact post-activation input to the down projection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        self.hook_post(x)
        self._buf_mlp_post[: x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        return x


class JambaCompareAttentionDecoderLayer(JambaPAttentionDecoderLayer):
    """Jamba attention layer with independent D2D reference copies."""

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del positions, kwargs
        n_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
        k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        for name, value in (
            ("q", q_by_head),
            ("k", k_by_head),
            ("v", v_by_head),
        ):
            getattr(self, f"hook_{name}")(value)
            getattr(self, f"_buf_{name}")[:n_tokens].copy_(value)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:n_tokens].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        n_tokens = hidden_states.shape[0]
        self.hook_resid_pre(residual)
        self._buf_resid_pre[:n_tokens].copy_(residual)
        self.hook_ln1(hidden_states)
        self._buf_ln1[:n_tokens].copy_(hidden_states)
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
        )
        self.hook_attn_out(hidden_states)
        self._buf_attn_out[:n_tokens].copy_(hidden_states)
        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)
        self.hook_resid_mid(residual)
        self._buf_resid_mid[:n_tokens].copy_(residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:n_tokens].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:n_tokens].copy_(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[:n_tokens].copy_(hidden_states)
        return hidden_states, residual


class JambaCompareMambaDecoderLayer(JambaPMambaDecoderLayer):
    """Jamba Mamba layer with independent D2D reference copies."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        n_tokens = hidden_states.shape[0]
        self.hook_resid_pre(residual)
        self._buf_resid_pre[:n_tokens].copy_(residual)
        self.hook_ln1(hidden_states)
        self._buf_ln1[:n_tokens].copy_(hidden_states)
        self.hook_ssm_in(hidden_states)
        self._buf_ssm_in[:n_tokens].copy_(hidden_states)
        output = torch.empty_like(hidden_states)
        self.mamba(hidden_states, output)
        self.hook_ssm_out(output)
        self._buf_ssm_out[:n_tokens].copy_(output)
        hidden_states, residual = self.pre_ff_layernorm(output, residual)
        self.hook_resid_mid(residual)
        self._buf_resid_mid[:n_tokens].copy_(residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:n_tokens].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:n_tokens].copy_(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[:n_tokens].copy_(hidden_states)
        return hidden_states, residual


class JambaCompareModel(JambaPModel):
    """Jamba backbone with model-wide D2D reference copies."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _instrument_upstream_jamba_model(self)
        _upgrade_to_compare_model(self)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            self.hook_embed(hidden_states)
            self._buf_embed[: hidden_states.shape[0]].copy_(hidden_states)
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
        self.hook_resid_final(resid_final)
        self._buf_resid_final[: resid_final.shape[0]].copy_(resid_final)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states


def _upgrade_to_compare_model(model: JambaPModel) -> JambaCompareModel:
    """Upgrade the already-instrumented upstream tree without rebuilding it."""

    model.__class__ = JambaCompareModel
    for layer_no in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_no]
        if isinstance(layer, PPMissingLayer):
            continue
        layer.feed_forward.__class__ = JambaCompareMLP
        if model.config.layers_block_type[layer_no] == "attention":
            layer.__class__ = JambaCompareAttentionDecoderLayer
        else:
            layer.__class__ = JambaCompareMambaDecoderLayer
    return model


class JambaCompareForCausalLM(JambaPForCausalLM):
    """Dense Jamba compare model used only by transport-value tests."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=JambaCompareModel,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            self._buf_token_ids[: input_ids.shape[0]].copy_(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        self._buf_final_logits[: logits.shape[0]].copy_(logits)
        return logits

    def allocate_compare_buffers(
        self,
        max_len: int,
        vllm_config: VllmConfig,
    ) -> None:
        config = self.config
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        head_dim = config.hidden_size // config.num_attention_heads
        model = self.model
        for name in ("embed", "resid_final", "final_ln"):
            setattr(
                model,
                f"_buf_{name}",
                torch.empty(
                    max_len,
                    config.hidden_size,
                    device=device,
                    dtype=dtype,
                ),
            )
        common_names = (
            "resid_pre",
            "ln1",
            "resid_mid",
            "ln2",
            "mlp_in",
            "mlp_out",
        )
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in common_names:
                setattr(
                    layer,
                    f"_buf_{name}",
                    torch.empty(
                        max_len,
                        config.hidden_size,
                        device=device,
                        dtype=dtype,
                    ),
                )
            layer.feed_forward._buf_mlp_post = torch.empty(
                max_len,
                layer.feed_forward.down_proj.input_size_per_partition,
                device=device,
                dtype=dtype,
            )
            if config.layers_block_type[layer_no] == "attention":
                layer._buf_attn_out = torch.empty(
                    max_len,
                    config.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                layer._buf_q = torch.empty(
                    max_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
                for name in ("k", "v"):
                    setattr(
                        layer,
                        f"_buf_{name}",
                        torch.empty(
                            max_len,
                            num_kv_heads,
                            head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                    )
                layer._buf_z = torch.empty(
                    max_len,
                    num_heads * head_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                for name in ("ssm_in", "ssm_out"):
                    setattr(
                        layer,
                        f"_buf_{name}",
                        torch.empty(
                            max_len,
                            config.hidden_size,
                            device=device,
                            dtype=dtype,
                        ),
                    )
        max_requests = vllm_config.scheduler_config.max_num_seqs
        self._buf_final_logits = torch.empty(
            max_requests,
            config.vocab_size,
            device=device,
            dtype=dtype,
        )
        self._buf_token_ids = torch.empty(
            max_len,
            device=device,
            dtype=torch.int32,
        )

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        model = self.model
        for name in ("embed", "resid_final", "final_ln"):
            buffers[name] = getattr(model, f"_buf_{name}")
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                buffers[f"{name}_L{layer_no}"] = getattr(layer, f"_buf_{name}")
            buffers[f"mlp_post_L{layer_no}"] = layer.feed_forward._buf_mlp_post
            if self.config.layers_block_type[layer_no] == "attention":
                buffers[f"attn_out_L{layer_no}"] = layer._buf_attn_out
                for name in ("q", "k", "v", "z"):
                    buffers[f"{name}_L{layer_no}"] = getattr(layer, f"_buf_{name}")
            else:
                for name in ("ssm_in", "ssm_out"):
                    buffers[f"{name}_L{layer_no}"] = getattr(layer, f"_buf_{name}")
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["JambaCompareForCausalLM"]
