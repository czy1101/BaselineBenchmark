"""Correctness tests for the S60 MSA Torch reference."""

from __future__ import annotations

import os

import pytest
import torch

from backends._enflame.ops.msa import (
    BLOCK_SIZE,
    torch_gcu_msa_decode,
    torch_gcu_msa_prefill_b1,
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


def _dense_causal_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    total_q, _, head_dim = q.shape

    keys = kv_cache[
        0,
        0,
        :total_q,
        :head_dim,
    ]
    values = kv_cache[
        0,
        0,
        :total_q,
        head_dim:,
    ]

    output = torch.empty_like(q)

    for query_id in range(total_q):
        valid_keys = keys[: query_id + 1]
        valid_values = values[: query_id + 1]

        logits = torch.matmul(
            q[query_id].float(),
            valid_keys.float().transpose(0, 1),
        )
        probabilities = torch.softmax(
            logits * scale,
            dim=-1,
        ).to(valid_values.dtype)

        output[query_id] = torch.matmul(
            probabilities,
            valid_values,
        )

    return output


@pytest.mark.minimax_sparse_attention
def test_torch_gcu_msa_prefill_matches_dense_causal() -> None:
    device = _gcu_device()
    torch.manual_seed(42)

    total_q = 8
    num_kv_heads = 1
    num_query_heads = 2
    head_dim = 16
    index_dim = 8
    topk = 1
    scale = head_dim**-0.5

    idx_q = torch.randn(
        (total_q, num_kv_heads, index_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    index_kv_cache = torch.randn(
        (1, BLOCK_SIZE, index_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q = torch.randn(
        (total_q, num_query_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        (
            1,
            num_kv_heads,
            BLOCK_SIZE,
            2 * head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.zeros(
        (1, 1),
        dtype=torch.int32,
        device=device,
    )

    scores, selected, actual = (
        torch_gcu_msa_prefill_b1(
            idx_q=idx_q,
            index_kv_cache=index_kv_cache,
            q=q,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=total_q,
            prefix_len=0,
            topk=topk,
            init_blocks=1,
            local_blocks=1,
            sm_scale=scale,
        )
    )

    expected = _dense_causal_reference(
        q=q,
        kv_cache=kv_cache,
        scale=scale,
    )

    torch.gcu.synchronize()

    assert scores.shape == (
        num_kv_heads,
        total_q,
        1,
    )
    assert selected.shape == (
        num_kv_heads,
        total_q,
        topk,
    )
    assert torch.all(selected == 0).item()
    assert torch.isfinite(actual.float()).all().item()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.03,
        rtol=0.03,
    )


@pytest.mark.minimax_sparse_attention
def test_torch_gcu_msa_decode_matches_dense_attention() -> None:
    device = _gcu_device()
    torch.manual_seed(43)

    batch = 1
    decode_query_len = 1
    sequence_length = 8
    num_kv_heads = 1
    num_query_heads = 2
    head_dim = 16
    index_dim = 8
    topk = 1
    scale = head_dim**-0.5

    idx_q = torch.randn(
        (
            batch * decode_query_len,
            num_kv_heads,
            index_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    index_kv_cache = torch.randn(
        (1, BLOCK_SIZE, index_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    q = torch.randn(
        (
            batch * decode_query_len,
            num_query_heads,
            head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(
        (
            1,
            num_kv_heads,
            BLOCK_SIZE,
            2 * head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    block_table = torch.zeros(
        (batch, 1),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(
        [sequence_length],
        dtype=torch.int32,
        device=device,
    )

    selected, actual = torch_gcu_msa_decode(
        idx_q=idx_q,
        index_kv_cache=index_kv_cache,
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        decode_query_len=decode_query_len,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
        num_kv_heads=num_kv_heads,
        sm_scale=scale,
    )

    keys = kv_cache[
        0,
        0,
        :sequence_length,
        :head_dim,
    ]
    values = kv_cache[
        0,
        0,
        :sequence_length,
        head_dim:,
    ]

    logits = torch.matmul(
        q[0].float(),
        keys.float().transpose(0, 1),
    )
    probabilities = torch.softmax(
        logits * scale,
        dim=-1,
    ).to(values.dtype)

    expected = torch.matmul(
        probabilities,
        values,
    ).reshape_as(actual)

    torch.gcu.synchronize()

    assert selected.shape == (
        num_kv_heads,
        batch * decode_query_len,
        topk,
    )
    assert torch.all(selected == 0).item()
    assert torch.isfinite(actual.float()).all().item()

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=0.03,
        rtol=0.03,
    )
