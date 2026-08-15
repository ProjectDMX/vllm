# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.kda import (
    chunk_kda,
    chunk_kda_scaled_dot_kkt_fwd,
    recompute_w_u_fwd,
)
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.utils.torch_utils import set_random_seed


# This branch predates the shared constant introduced upstream; its KDA path
# uses the same fixed chunk size directly.
FLA_CHUNK_SIZE = 64


def _chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KDA output using PyTorch for the final [K, V] state contraction."""
    chunk_size = FLA_CHUNK_SIZE
    batch_size, seq_len, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]

    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    a, aqk = chunk_kda_scaled_dot_kkt_fwd(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    a = solve_tril(a, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a,
        gk=g,
        cu_seqlens=cu_seqlens,
    )
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    h = torch.empty(
        batch_size,
        num_chunks,
        num_heads,
        head_dim,
        value_dim,
        device=q.device,
        dtype=torch.float32,
    )
    v_new = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            state = initial_state[batch_idx, head_idx].float().clone()
            for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
                end = min(start + chunk_size, seq_len)
                h[batch_idx, chunk_idx, head_idx] = state
                new_value = (
                    u[batch_idx, start:end, head_idx].float()
                    - w[batch_idx, start:end, head_idx].float() @ state
                )
                v_new[batch_idx, start:end, head_idx] = new_value.to(v.dtype)
                state = state * torch.exp(
                    g[batch_idx, end - 1, head_idx].float()
                ).unsqueeze(-1)
                state += (
                    kg[batch_idx, start:end, head_idx].float().transpose(0, 1)
                    @ v_new[batch_idx, start:end, head_idx].float()
                )
            final_state[batch_idx, head_idx] = state

    output = torch.empty_like(v)
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                qg = (
                    q[batch_idx, start:end, head_idx].float()
                    * scale
                    * torch.exp(g[batch_idx, start:end, head_idx].float())
                )
                # h is defined as [num_chunks, num_heads, K, V].
                ref = qg @ h[batch_idx, chunk_idx, head_idx].float()
                local_mask = torch.tril(
                    torch.ones(
                        chunk_len,
                        chunk_len,
                        dtype=torch.bool,
                        device=q.device,
                    )
                )
                local_a = aqk[
                    batch_idx,
                    start:end,
                    head_idx,
                    :chunk_len,
                ].float()
                local_a = local_a.masked_fill(~local_mask, 0)
                ref += local_a @ v_new[batch_idx, start:end, head_idx].float()
                output[batch_idx, start:end, head_idx] = ref.to(v.dtype)

    return output, final_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_chunk_kda_layout_regression() -> None:
    """Guard KDA's K,V recurrent-state layout for Kimi-like dimensions."""
    set_random_seed(123)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    seq_len = 512
    num_heads = 4
    head_dim = 256
    scale = head_dim**-0.5

    q = F.normalize(
        torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype),
        p=2,
        dim=-1,
    )
    v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    # KDA passes log gates to chunk_kda after the fused gate projection.
    g = F.logsigmoid(
        torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    ).to(dtype)
    beta = torch.rand(1, seq_len, num_heads, device=device, dtype=torch.float32).to(dtype)
    initial_state = torch.zeros(
        1, num_heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.long)

    actual_out, actual_state = chunk_kda(
        q=q,
        k=k,
        v=v.clone(),
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    expected_out, expected_state = _chunk_kda_reference(
        q=q,
        k=k,
        v=v.clone(),
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=1e-2, rtol=1e-2)
