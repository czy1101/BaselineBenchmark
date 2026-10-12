"""Standalone input construction for S60 MSA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


BLOCK_SIZE = 128
HEAD_DIM = 128
INDEX_DIM = 128


@dataclass
class MSAInputs:
    idx_q: torch.Tensor
    index_kv_cache: torch.Tensor
    q: torch.Tensor
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    cu_q: torch.Tensor
    max_seq_len: int
    sm_scale: float


def make_prefill_inputs(
    *,
    sequence_length: int,
    num_kv_heads: int,
    group_size: int,
    device: str,
    dtype: torch.dtype,
) -> MSAInputs:
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    num_query_heads = (
        num_kv_heads * group_size
    )
    block_count = (
        sequence_length + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    idx_q = torch.randn(
        (
            sequence_length,
            num_kv_heads,
            INDEX_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    index_kv_cache = torch.randn(
        (
            block_count,
            BLOCK_SIZE,
            INDEX_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    q = torch.randn(
        (
            sequence_length,
            num_query_heads,
            HEAD_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    kv_cache = torch.randn(
        (
            block_count,
            num_kv_heads,
            BLOCK_SIZE,
            2 * HEAD_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    block_table = torch.arange(
        block_count,
        dtype=torch.int32,
        device=device,
    ).reshape(1, block_count)
    seq_lens = torch.tensor(
        [sequence_length],
        dtype=torch.int32,
        device=device,
    )
    prefix_lens = torch.zeros(
        1,
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.tensor(
        [0, sequence_length],
        dtype=torch.int32,
        device=device,
    )

    return MSAInputs(
        idx_q=idx_q,
        index_kv_cache=index_kv_cache,
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        cu_q=cu_q,
        max_seq_len=sequence_length,
        sm_scale=HEAD_DIM**-0.5,
    )


def make_decode_inputs(
    *,
    sequence_lengths: tuple[int, ...],
    decode_query_len: int,
    num_kv_heads: int,
    group_size: int,
    device: str,
    dtype: torch.dtype,
) -> MSAInputs:
    if not sequence_lengths:
        raise ValueError(
            "sequence_lengths must not be empty"
        )
    if decode_query_len <= 0:
        raise ValueError(
            "decode_query_len must be positive"
        )
    if min(sequence_lengths) < decode_query_len:
        raise ValueError(
            "all sequences must contain decode queries"
        )

    batch = len(sequence_lengths)
    max_seq_len = max(sequence_lengths)
    max_blocks = (
        max_seq_len + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    page_count = batch * max_blocks
    total_q = batch * decode_query_len
    num_query_heads = (
        num_kv_heads * group_size
    )

    idx_q = torch.randn(
        (
            total_q,
            num_kv_heads,
            INDEX_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    index_kv_cache = torch.randn(
        (
            page_count,
            BLOCK_SIZE,
            INDEX_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    q = torch.randn(
        (
            total_q,
            num_query_heads,
            HEAD_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    kv_cache = torch.randn(
        (
            page_count,
            num_kv_heads,
            BLOCK_SIZE,
            2 * HEAD_DIM,
        ),
        dtype=dtype,
        device=device,
    )
    block_table = torch.arange(
        page_count,
        dtype=torch.int32,
        device=device,
    ).reshape(batch, max_blocks)
    seq_lens = torch.tensor(
        sequence_lengths,
        dtype=torch.int32,
        device=device,
    )
    prefix_lens = torch.zeros(
        batch,
        dtype=torch.int32,
        device=device,
    )
    cu_q = torch.arange(
        0,
        total_q + 1,
        decode_query_len,
        dtype=torch.int32,
        device=device,
    )

    return MSAInputs(
        idx_q=idx_q,
        index_kv_cache=index_kv_cache,
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        cu_q=cu_q,
        max_seq_len=max_seq_len,
        sm_scale=HEAD_DIM**-0.5,
    )
