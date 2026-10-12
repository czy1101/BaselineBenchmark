"""Pure Torch correctness reference for S60 NSA forward.

This module is a correctness and performance baseline. It is not a
vendor-optimized implementation and does not depend on Triton, TLE,
FlagAttention, or FlagGems.
"""

from __future__ import annotations

import torch


def parallel_nsa_torch_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: int | torch.Tensor = 16,
    block_size: int = 64,
    scale: float | None = None,
    max_batch_query: int = 256,
) -> torch.Tensor:
    """Pure PyTorch fixed-length NSA selected-attention baseline.

    Shapes:
        q: [B, T, HQ, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        block_indices: [B, T, H, S]

    This implementation intentionally uses only PyTorch operators.
    It does not import FlagGems, Triton kernels, or TLE.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")

    B, T, HQ, K = q.shape
    Bk, Tk, H, Kk = k.shape
    Bv, Tv, Hv, V = v.shape

    if (Bk, Tk, Kk) != (B, T, K):
        raise ValueError("q and k shapes are incompatible")

    if (Bv, Tv, Hv) != (B, T, H):
        raise ValueError("k and v shapes are incompatible")

    if HQ % H != 0:
        raise ValueError("HQ must be divisible by H")

    if block_indices.shape[:3] != (B, T, H):
        raise ValueError("block_indices has an incompatible shape")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if scale is None:
        scale = K**-0.5

    G = HQ // H
    S = block_indices.shape[-1]
    N = S * block_size

    # Keep flattened row arithmetic in int32 because GCU does not
    # support int64 values inside the compiled device path.
    batch_ids = torch.arange(
        B,
        dtype=torch.int32,
        device=q.device,
    ).view(B, 1, 1, 1)

    head_ids = torch.arange(
        H,
        dtype=torch.int32,
        device=q.device,
    ).view(1, 1, H, 1)

    block_offsets = torch.arange(
        block_size,
        dtype=torch.int32,
        device=q.device,
    ).view(1, 1, 1, 1, block_size)

    slot_ids = torch.arange(
        S,
        dtype=torch.int32,
        device=q.device,
    ).view(1, 1, 1, S, 1)

    k_rows = k.reshape(B * T * H, K)
    v_rows = v.reshape(B * T * H, V)

    query_chunk_size = max(
        1,
        min(
            T,
            max_batch_query // max(B, 1),
        ),
    )

    output_chunks = []

    for query_start in range(0, T, query_chunk_size):
        query_end = min(T, query_start + query_chunk_size)
        Q = query_end - query_start
        M = B * Q * H

        selected_blocks = block_indices[
            :,
            query_start:query_end,
            :,
            :,
        ].to(torch.int32)

        token_ids = (
            selected_blocks.unsqueeze(-1) * block_size
            + block_offsets
        )

        in_bounds = (
            (token_ids >= 0)
            & (token_ids < T)
        )

        query_ids = torch.arange(
            query_start,
            query_end,
            dtype=torch.int32,
            device=q.device,
        ).view(1, Q, 1, 1, 1)

        causal = token_ids <= query_ids

        if isinstance(block_counts, int):
            slot_valid = slot_ids < min(block_counts, S)
        else:
            counts = block_counts[
                :,
                query_start:query_end,
                :,
            ].to(torch.int32)

            slot_valid = (
                slot_ids
                < counts.unsqueeze(-1).unsqueeze(-1)
            )

        valid = in_bounds & causal & slot_valid

        safe_token_ids = torch.where(
            in_bounds,
            token_ids,
            torch.zeros_like(token_ids),
        )

        safe_token_ids = safe_token_ids.reshape(
            B,
            Q,
            H,
            N,
        )

        valid = valid.reshape(
            B,
            Q,
            H,
            N,
        )

        flattened_rows = (
            (
                batch_ids * T
                + safe_token_ids
            )
            * H
            + head_ids
        ).reshape(-1)

        selected_k = torch.index_select(
            k_rows,
            0,
            flattened_rows,
        ).reshape(B, Q, H, N, K)

        selected_v = torch.index_select(
            v_rows,
            0,
            flattened_rows,
        ).reshape(B, Q, H, N, V)

        q_chunk = q[
            :,
            query_start:query_end,
            :,
            :,
        ].reshape(B, Q, H, G, K)

        q_matrix = q_chunk.reshape(M, G, K).float()
        k_matrix = selected_k.reshape(M, N, K).float()

        scores = torch.bmm(
            q_matrix,
            k_matrix.transpose(1, 2),
        )

        scores = scores * float(scale)

        valid_matrix = valid.reshape(
            M,
            1,
            N,
        ).expand(-1, G, -1)

        scores = scores.masked_fill(
            ~valid_matrix,
            torch.finfo(torch.float32).min,
        )

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        # Ensure an all-invalid query produces zero rather than
        # a uniform contribution from the placeholder indices.
        probabilities = probabilities.masked_fill(
            ~valid_matrix,
            0.0,
        )

        v_matrix = selected_v.reshape(M, N, V).float()

        output = torch.bmm(
            probabilities,
            v_matrix,
        ).to(q.dtype)

        output = output.reshape(
            B,
            Q,
            H,
            G,
            V,
        ).reshape(B, Q, HQ, V)

        output_chunks.append(output)

    return torch.cat(output_chunks, dim=1)
