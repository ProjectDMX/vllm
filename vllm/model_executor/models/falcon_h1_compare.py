# SPDX-License-Identifier: Apache-2.0
"""Falcon-H1 DMI model with independent transport-reference buffers."""

from __future__ import annotations

from itertools import islice

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors

from .falcon_h1_p import (
    FalconH1PAttentionDecoderLayer,
    FalconH1PForCausalLM,
    FalconH1PMLP,
    FalconH1PModel,
    FalconH1PParallelHybrid,
)
from .utils import PPMissingLayer


class FalconH1CompareMLP(FalconH1PMLP):
    """Capture Falcon-H1's scaled post-activation MLP tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x[:, : self.intermediate_size // self.tp_size] *= (
            self.gate_multiplier
        )
        x = self.act_fn(x)
        self.hook_post(x)
        self._buf_mlp_post[: x.shape[0]].copy_(x)
        x, _ = self.down_proj(x)
        x = x * self.down_multiplier
        return x


class FalconH1CompareAttention(FalconH1PAttentionDecoderLayer):
    """Capture scaled K and the remaining canonical attention tensors."""

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

        q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
        k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        self.hook_q(q_by_head)
        self._buf_q[: q_by_head.shape[0]].copy_(q_by_head)
        self.hook_k(k_by_head)
        self._buf_k[: k_by_head.shape[0]].copy_(k_by_head)
        self.hook_v(v_by_head)
        self._buf_v[: v_by_head.shape[0]].copy_(v_by_head)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[: attn_output.shape[0]].copy_(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class FalconH1CompareParallelHybrid(FalconH1PParallelHybrid):
    """Capture every declared hybrid-block boundary in execution order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            attention_type=FalconH1CompareAttention,
            mlp_type=FalconH1CompareMLP,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        self.hook_resid_pre(residual)
        self._buf_resid_pre[: residual.shape[0]].copy_(residual)
        hidden_states = self.input_layernorm(hidden_states)
        self.hook_ln1(hidden_states)
        self._buf_ln1[: hidden_states.shape[0]].copy_(hidden_states)

        attn_hidden, _ = self.self_attn(
            positions=positions,
            hidden_states=(
                hidden_states * self.attention_in_multiplier
            ),
            residual=residual,
            **kwargs,
        )
        attn_capture = attn_hidden * self.attn_out_multiplier
        self.hook_attn_out(attn_capture)
        self._buf_attn_out[: attn_capture.shape[0]].copy_(attn_capture)

        ssm_capture = hidden_states * self.ssm_in_multiplier
        self.hook_ssm_in(ssm_capture)
        self._buf_ssm_in[: ssm_capture.shape[0]].copy_(ssm_capture)
        ssm_hidden, _ = self.mamba(
            hidden_states=hidden_states * self.ssm_in_multiplier,
            residual=residual,
            **kwargs,
        )
        ssm_capture = ssm_hidden * self.ssm_out_multiplier
        self.hook_ssm_out(ssm_capture)
        self._buf_ssm_out[: ssm_capture.shape[0]].copy_(ssm_capture)

        hidden_states = (attn_hidden * self.attn_out_multiplier) + (
            ssm_hidden * self.ssm_out_multiplier
        )
        hidden_states = hidden_states + residual
        self.hook_resid_mid(hidden_states)
        self._buf_resid_mid[: hidden_states.shape[0]].copy_(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        self.hook_ln2(hidden_states)
        self._buf_ln2[: hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[: hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[: hidden_states.shape[0]].copy_(hidden_states)
        return residual + hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {0: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    }
)
class FalconH1CompareModel(FalconH1PModel):
    """Falcon-H1 compare backbone with model-wide D2D references."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=FalconH1CompareParallelHybrid,
        )

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
            self.hook_embed(hidden_states)
            self._buf_embed[: hidden_states.shape[0]].copy_(hidden_states)
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

        self.hook_resid_final(hidden_states)
        self._buf_resid_final[: hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.final_layernorm(hidden_states)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states


class FalconH1CompareForCausalLM(FalconH1PForCausalLM):
    """Falcon-H1 compare model used only by transport-value tests."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=FalconH1CompareModel,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is not None and get_pp_group().is_first_rank:
            self.hook_token_ids(input_ids)
            self._buf_token_ids[: input_ids.shape[0]].copy_(input_ids)
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
        self.hook_final_logits(logits)
        self._buf_final_logits[: logits.shape[0]].copy_(logits)
        return logits

    def allocate_compare_buffers(
        self, max_len: int, vllm_config: VllmConfig
    ) -> None:
        config = self.config
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        dtype = vllm_config.model_config.dtype
        device = "cuda"
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads // tp_size
        num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        intermediate_size = config.intermediate_size // tp_size

        model = self.model
        for name in ("embed", "resid_final", "final_ln"):
            setattr(
                model,
                f"_buf_{name}",
                torch.empty(
                    max_len,
                    hidden_size,
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
                "ssm_in",
                "ssm_out",
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
                        hidden_size,
                        device=device,
                        dtype=dtype,
                    ),
                )
            layer.feed_forward._buf_mlp_post = torch.empty(
                max_len,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            attention = layer.self_attn
            attention._buf_q = torch.empty(
                max_len,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for name in ("k", "v"):
                setattr(
                    attention,
                    f"_buf_{name}",
                    torch.empty(
                        max_len,
                        num_kv_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ),
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
                "ssm_in",
                "ssm_out",
                "resid_mid",
                "ln2",
                "mlp_in",
                "mlp_out",
            ):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer, f"_buf_{name}"
                )
            buffers[f"mlp_post_L{layer_no}"] = (
                layer.feed_forward._buf_mlp_post
            )
            for name in ("q", "k", "v", "z"):
                buffers[f"{name}_L{layer_no}"] = getattr(
                    layer.self_attn, f"_buf_{name}"
                )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["FalconH1CompareForCausalLM"]
