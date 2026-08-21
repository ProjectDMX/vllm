# SPDX-License-Identifier: Apache-2.0
"""Gemma 3 DMI model with independent buffers for transport comparison."""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from .gemma3_p import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3MLP,
    Gemma3Model,
    Gemma3PForCausalLM,
)


class Gemma3CompareMLP(Gemma3MLP):
    """Gemma 3 MLP that captures the exact post-activation hook tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        self._buf_mlp_post[: x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        return x


class Gemma3CompareAttention(Gemma3Attention):
    """Gemma 3 attention that captures normalized Q/K and pre-projection Z."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k_by_head = self.k_norm(k_by_head)
        v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))

        self.hook_q(q_by_head)
        self._buf_q[: q_by_head.shape[0]].copy_(q_by_head)
        self.hook_k(k_by_head)
        self._buf_k[: k_by_head.shape[0]].copy_(k_by_head)
        self.hook_v(v_by_head)
        self._buf_v[: v_by_head.shape[0]].copy_(v_by_head)

        q = q_by_head.flatten(-2, -1)
        k = k_by_head.flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[: attn_output.shape[0]].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma3CompareDecoderLayer(Gemma3DecoderLayer):
    """Gemma 3 decoder that captures every canonical residual/norm boundary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args,
            **kwargs,
            attention_type=Gemma3CompareAttention,
            mlp_type=Gemma3CompareMLP,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            resid_pre = hidden_states
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(resid_pre)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            resid_pre = hidden_states + residual
            if self.hook_resid_pre.enabled:
                self.hook_resid_pre(resid_pre)
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        self._buf_resid_pre[: resid_pre.shape[0]].copy_(resid_pre)
        self.hook_ln1(hidden_states)
        self._buf_ln1[: hidden_states.shape[0]].copy_(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        self.hook_attn_out(hidden_states)
        self._buf_attn_out[: hidden_states.shape[0]].copy_(hidden_states)

        resid_mid = hidden_states + residual
        if self.hook_resid_mid.enabled:
            self.hook_resid_mid(resid_mid)
        self._buf_resid_mid[: resid_mid.shape[0]].copy_(resid_mid)
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        self.hook_ln2(hidden_states)
        self._buf_ln2[: hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[: hidden_states.shape[0]].copy_(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {-1: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    },
)
class Gemma3CompareModel(Gemma3Model):
    """Gemma 3 backbone that captures model-wide hook tensors."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Gemma3CompareDecoderLayer,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
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
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        resid_final = hidden_states + residual
        if self.hook_resid_final.enabled:
            self.hook_resid_final(resid_final)
        self._buf_resid_final[: resid_final.shape[0]].copy_(resid_final)
        hidden_states, _ = self.norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states


class Gemma3CompareForCausalLM(Gemma3PForCausalLM):
    """Gemma 3 compare model used only by transport-value tests."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=Gemma3CompareModel,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            self._buf_token_ids[: input_ids.shape[0]].copy_(input_ids)
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        self._buf_final_logits[: logits.shape[0]].copy_(logits)
        return logits

    def allocate_compare_buffers(
        self, max_len: int, vllm_config: VllmConfig
    ) -> None:
        """Allocate the independent D2D reference buffers."""

        config = self.config
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        dtype = vllm_config.model_config.dtype
        device = "cuda"

        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        intermediate_size = config.intermediate_size // tp_size

        model = self.model
        model._buf_embed = torch.empty(
            max_len, hidden_size, device=device, dtype=dtype
        )
        model._buf_resid_final = torch.empty(
            max_len, hidden_size, device=device, dtype=dtype
        )
        model._buf_final_ln = torch.empty(
            max_len, hidden_size, device=device, dtype=dtype
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
                        max_len, hidden_size, device=device, dtype=dtype
                    ),
                )
            layer.mlp._buf_mlp_post = torch.empty(
                max_len, intermediate_size, device=device, dtype=dtype
            )
            attention = layer.self_attn
            attention._buf_q = torch.empty(
                max_len, num_heads, head_dim, device=device, dtype=dtype
            )
            attention._buf_k = torch.empty(
                max_len, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            attention._buf_v = torch.empty(
                max_len, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            attention._buf_z = torch.empty(
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
            max_len, device=device, dtype=torch.int32
        )

    def get_ref_buffers(self) -> dict[str, torch.Tensor]:
        """Return every independently captured reference tensor."""

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
                    layer.self_attn, f"_buf_{name}"
                )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["Gemma3CompareForCausalLM"]
