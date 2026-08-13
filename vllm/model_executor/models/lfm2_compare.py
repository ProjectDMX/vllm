# SPDX-License-Identifier: Apache-2.0
"""LFM2 DMI model with independent transport-reference buffers."""

from __future__ import annotations

from itertools import islice

import torch
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.sequence import IntermediateTensors

from .lfm2_p import (
    Lfm2PAttention,
    Lfm2PAttentionDecoderLayer,
    Lfm2PForCausalLM,
    Lfm2PMLP,
    Lfm2PModel,
    Lfm2PShortConvDecoderLayer,
)
from .utils import PPMissingLayer


class Lfm2CompareMLP(Lfm2PMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x = self.act_fn(gate_up)
        self.hook_post(x)
        self._buf_mlp_post[: x.shape[0]].copy_(x)
        x, _ = self.w2(x)
        return x


class Lfm2CompareAttention(Lfm2PAttention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = q.view(n_tokens, self.num_heads, self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads, self.head_dim)
        v_by_head = v.view(
            n_tokens, self.num_kv_heads, self.head_dim
        )
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        self.hook_q(q)
        self._buf_q[:n_tokens].copy_(q)
        self.hook_k(k)
        self._buf_k[:n_tokens].copy_(k)
        self.hook_v(v_by_head)
        self._buf_v[:n_tokens].copy_(v_by_head)
        q, k = self.rotary_emb(positions, q, k)
        q = q.view(n_tokens, self.num_heads * self.head_dim)
        k = k.view(n_tokens, self.num_kv_heads * self.head_dim)
        attn_output = self.attn(q, k, v)
        self.hook_z(attn_output)
        self._buf_z[:n_tokens].copy_(attn_output)
        output, _ = self.out_proj(attn_output)
        return output


class Lfm2CompareAttentionDecoderLayer(Lfm2PAttentionDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            attention_type=Lfm2CompareAttention,
            mlp_type=Lfm2CompareMLP,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            self.hook_resid_pre(hidden_states)
            self._buf_resid_pre[: hidden_states.shape[0]].copy_(
                hidden_states
            )
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            resid_pre = hidden_states + residual
            self.hook_resid_pre(resid_pre)
            self._buf_resid_pre[: resid_pre.shape[0]].copy_(resid_pre)
            hidden_states, residual = self.operator_norm(
                hidden_states, residual
            )
        self.hook_ln1(hidden_states)
        self._buf_ln1[: hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        self.hook_attn_out(hidden_states)
        self._buf_attn_out[: hidden_states.shape[0]].copy_(hidden_states)
        resid_mid = hidden_states + residual
        self.hook_resid_mid(resid_mid)
        self._buf_resid_mid[: resid_mid.shape[0]].copy_(resid_mid)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[: hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[: hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


class Lfm2CompareShortConvDecoderLayer(Lfm2PShortConvDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, mlp_type=Lfm2CompareMLP)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            self.hook_resid_pre(hidden_states)
            self._buf_resid_pre[: hidden_states.shape[0]].copy_(
                hidden_states
            )
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            resid_pre = hidden_states + residual
            self.hook_resid_pre(resid_pre)
            self._buf_resid_pre[: resid_pre.shape[0]].copy_(resid_pre)
            hidden_states, residual = self.operator_norm(
                hidden_states, residual
            )
        self.hook_ln1(hidden_states)
        self._buf_ln1[: hidden_states.shape[0]].copy_(hidden_states)
        self.hook_conv_in(hidden_states)
        self._buf_conv_in[: hidden_states.shape[0]].copy_(hidden_states)
        output = torch.empty_like(hidden_states)
        self.short_conv(hidden_states, output)
        self.hook_conv_out(output)
        self._buf_conv_out[: output.shape[0]].copy_(output)
        resid_mid = output + residual
        self.hook_resid_mid(resid_mid)
        self._buf_resid_mid[: resid_mid.shape[0]].copy_(resid_mid)
        hidden_states, residual = self.ffn_norm(output, residual)
        self.hook_ln2(hidden_states)
        self._buf_ln2[: hidden_states.shape[0]].copy_(hidden_states)
        self.hook_mlp_in(hidden_states)
        self._buf_mlp_in[: hidden_states.shape[0]].copy_(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        self.hook_mlp_out(hidden_states)
        self._buf_mlp_out[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "b"},
        "positions": {-1: "b"},
        "intermediate_tensors": {0: "b"},
        "inputs_embeds": {0: "b"},
    }
)
class Lfm2CompareModel(Lfm2PModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            attention_layer_type=Lfm2CompareAttentionDecoderLayer,
            conv_layer_type=Lfm2CompareShortConvDecoderLayer,
        )

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
        for layer in islice(
            self.layers, self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        resid_final = hidden_states + residual
        self.hook_resid_final(resid_final)
        self._buf_resid_final[: resid_final.shape[0]].copy_(resid_final)
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        self.hook_final_ln(hidden_states)
        self._buf_final_ln[: hidden_states.shape[0]].copy_(hidden_states)
        return hidden_states


class Lfm2CompareForCausalLM(Lfm2PForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            model_type=Lfm2CompareModel,
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
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        self.hook_final_logits(logits)
        self._buf_final_logits[: logits.shape[0]].copy_(logits)
        return logits

    def allocate_compare_buffers(
        self, max_len: int, vllm_config: VllmConfig
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
                layer.feed_forward.w2.input_size_per_partition,
                device=device,
                dtype=dtype,
            )
            if config.layer_types[layer_no] == "full_attention":
                layer._buf_attn_out = torch.empty(
                    max_len,
                    config.hidden_size,
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
            else:
                for name in ("conv_in", "conv_out"):
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
            if self.config.layer_types[layer_no] == "full_attention":
                buffers[f"attn_out_L{layer_no}"] = layer._buf_attn_out
                for name in ("q", "k", "v", "z"):
                    buffers[f"{name}_L{layer_no}"] = getattr(
                        layer.self_attn, f"_buf_{name}"
                    )
            else:
                for name in ("conv_in", "conv_out"):
                    buffers[f"{name}_L{layer_no}"] = getattr(
                        layer, f"_buf_{name}"
                    )
        buffers["final_logits"] = self._buf_final_logits
        buffers["token_ids"] = self._buf_token_ids
        return buffers


__all__ = ["Lfm2CompareForCausalLM"]
