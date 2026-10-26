"""Correctness tests for the S60 NSA Torch reference."""

from __future__ import annotations

import os

import pytest
import torch

from backends._enflame.ops.nsa import (
    parallel_nsa_torch_baseline,
)


def _gcu_device() -> str:
    pytest.importorskip("torch_gcu")

    if not hasattr(torch, "gcu"):
        pytest.skip("torch_gcu is unavailable")

    count = torch.gcu.device_count()
    if count == 0:
        pytest.skip("no GCU device is available")

    index = int(os.environ.get("S60_DEVICE", "0"))
    if index < 0 or index >= count:
        pytest.skip(
            f"requested GCU {index}, available count is {count}"
        )

    torch.gcu.set_device(index)
    return f"gcu:{index}"


def _naive_nsa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices_cpu: torch.Tensor,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    batch, tokens, query_heads, _ = q.shape
    kv_heads = k.shape[2]
    group_size = query_heads // kv_heads
    value_dim = v.shape[-1]

    output = torch.empty(
        (batch, tokens, query_heads, value_dim),
        device=q.device,
        dtype=q.dtype,
    )

    for batch_id in range(batch):
        for query_id in range(tokens):
            for kv_head in range(kv_heads):
                selected_tokens = []

                for block_id in block_indices_cpu[
                    batch_id,
                    query_id,
                    kv_head,
                ].tolist():
                    for offset in range(block_size):
                        token_id = block_id * block_size + offset
                        if 0 <= token_id <= query_id:
                            selected_tokens.append(token_id)

                token_tensor = torch.tensor(
                    selected_tokens,
                    dtype=torch.int32,
                    device=q.device,
                )

                selected_k = torch.index_select(
                    k[batch_id, :, kv_head],
                    0,
                    token_tensor,
                ).float()

                selected_v = torch.index_select(
                    v[batch_id, :, kv_head],
                    0,
                    token_tensor,
                ).float()

                for group_id in range(group_size):
                    query_head = (
                        kv_head * group_size + group_id
                    )
                    scores = torch.mv(
                        selected_k,
                        q[
                            batch_id,
                            query_id,
                            query_head,
                        ].float(),
                    )
                    probabilities = torch.softmax(
                        scores * scale,
                        dim=0,
                    )
                    output[
                        batch_id,
                        query_id,
                        query_head,
                    ] = torch.mv(
                        selected_v.transpose(0, 1),
                        probabilities,
                    ).to(q.dtype)

    return output


@pytest.mark.parallel_nsa
def test_parallel_nsa_torch_baseline_matches_naive() -> None:
    device = _gcu_device()
    torch.manual_seed(42)

    batch = 1
    tokens = 8
    kv_heads = 1
    query_heads = 2
    head_dim = 8
    block_size = 2
    topk = 2
    scale = head_dim**-0.5

    q = torch.randn(
        (batch, tokens, query_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (batch, tokens, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn(
        (batch, tokens, kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    block_indices_cpu = torch.zeros(
        (batch, tokens, kv_heads, topk),
        dtype=torch.int32,
    )

    for query_id in range(tokens):
        block_indices_cpu[
            :,
            query_id,
            :,
            1,
        ] = query_id // block_size

    block_indices = block_indices_cpu.to(device)

    actual = parallel_nsa_torch_baseline(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=topk,
        block_size=block_size,
        scale=scale,
        max_batch_query=8,
    )

    expected = _naive_nsa_reference(
        q=q,
        k=k,
        v=v,
        block_indices_cpu=block_indices_cpu,
        block_size=block_size,
        scale=scale,
    )

    torch.gcu.synchronize()

    assert torch.isfinite(actual.float()).all().item()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.02,
        rtol=0.02,
    )
