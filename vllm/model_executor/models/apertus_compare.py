# SPDX-License-Identifier: Apache-2.0
"""Apertus DMI model with independent transport-reference buffers."""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.sequence import IntermediateTensors

from .apertus_p import (
    ApertusPAttention,
    ApertusPDecoderLayer,
    ApertusPForCausalLM,
    ApertusPMLP,
    ApertusPModel,
)
from .utils import PPMissingLayer


class ApertusCompareMLP(ApertusPMLP):
    """Capture Apertus xIELU output before the down projection."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        self.hook_post(x)
        self._buf_mlp_post[: x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        return x


class ApertusCompareAttention(ApertusPAttention):
    """Apertus attention with same-graph D2D reference copies."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view_as(q)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view_as(k)
        values = (
            ("q", q.view(-1, self.num_heads, self.head_dim)),
            ("k", k.view(-1, self.num_kv_heads, self.head_dim)),
            ("v", v.view(-1, self.num_kv_heads, self.head_dim)),
        )
        for name, value in values:
            getattr(self, f"hook_{name}")(value)
            getattr(self, f"_buf_{name}")[:n_tokens].copy_(value)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:n_tokens].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class ApertusCompareDecoderLayer(ApertusPDecoderLayer):
    """One fused pre-norm Apertus block with independent copies."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.self_attn.__class__ = ApertusCompareAttention
        self.mlp.__class__ = ApertusCompareMLP

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tokens = hidden_states.shape[0]
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.attention_layernorm(
                hidden_states, residual
            )
        self.hook_resid_pre(residual)
        self._buf_resid_pre[:n_tokens].copy_(residual)
        self.hook_ln1(hidden_states)
        self._buf_ln1[:n_tokens].copy_(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        self.hook_attn_out(hidden_states)
        self._buf_attn_out[:n_tokens].copy_(hidden_states)

        hidden_states, residual = self.feedforward_layernorm(
            hidden_states, residual
        )
        self.hook_resid_mid(residual)
        self._buf_resid_mid[:n_tokens].copy_(residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[:n_tokens].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[:n_tokens].copy_(hidden_states)
        hidden_states = self.mlp(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[:n_tokens].copy_(hidden_states)
        return hidden_states, residual


class ApertusCompareModel(ApertusPModel):
    """Concrete compiled Apertus backbone containing reference copies."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusCompareDecoderLayer,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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
        hidden_states, final_residual = self.norm(hidden_states, residual)
        self.hook_resid_final(final_residual)
        self._buf_resid_final[: final_residual.shape[0]].copy_(final_residual)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[: hidden_states.shape[0]].copy_(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class ApertusCompareForCausalLM(ApertusPForCausalLM):
    """Apertus compare oracle used only by storage-value tests."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=ApertusCompareModel,
            layer_type=ApertusCompareDecoderLayer,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            self._buf_token_ids[: input_ids.shape[0]].copy_(input_ids)
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super(ApertusPForCausalLM, self).compute_logits(hidden_states)
        if logits is not None:
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
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
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
        for layer_no in range(model.start_layer, model.end_layer):
            layer = model.layers[layer_no]
            if isinstance(layer, PPMissingLayer):
                continue
            for name in (
                "resid_pre",
                "ln1",
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
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
            layer.mlp._buf_mlp_post = torch.empty(
                max_len,
                layer.mlp.down_proj.input_size_per_partition,
                device=device,
                dtype=dtype,
            )
            layer.self_attn._buf_q = torch.empty(
                max_len,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for name in ("k", "v"):
                setattr(
                    layer.self_attn,
                    f"_buf_{name}",
                    torch.empty(
                        max_len,
                        num_kv_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                )
            layer.self_attn._buf_z = torch.empty(
                max_len,
                num_heads * head_dim,
                device=device,
                dtype=dtype,
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
                "attn_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer, f"_buf_{name}"
                )
            buffers[f"mlp_post_L{layer_no}"] = layer.mlp._buf_mlp_post
            for name in ("q", "k", "v", "z"):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer.self_attn,
                    f"_buf_{name}",
                )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["ApertusCompareForCausalLM"]
