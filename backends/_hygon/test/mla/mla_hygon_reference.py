"""Pure PyTorch dense FlashMLA reference for Hygon validation.

The interface follows FlagGems ``flash_mla``: q is [B,Sq,Hq,D], blocked_k is
[num_blocks, block_size,Hkv,D], and block_table maps each request to blocks.
The first ``dv`` channels of the cache are used as values.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def flash_mla_hygon_reference(
    q: torch.Tensor,
    block_table: torch.Tensor,
    blocked_k: torch.Tensor,
    max_seqlen_pad: int,
    block_size: int,
    b: int,
    s_q: int,
    cache_seqlens: torch.Tensor,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference dense MLA with paged KV cache.

    Returns output [B,Sq,Hq,dv] and LSE [B,Hq,Sq], both in float32.
    """
    if q.shape != (b, s_q, h_q, d):
        raise ValueError(f"q must be {(b,s_q,h_q,d)}, got {tuple(q.shape)}")
    if blocked_k.ndim != 4 or blocked_k.shape[1] != block_size:
        raise ValueError("blocked_k must be [num_blocks,block_size,h_kv,D]")
    if blocked_k.shape[2] != h_kv or blocked_k.shape[3] != d:
        raise ValueError("blocked_k head/dimension mismatch")
    if dv > d or h_q % h_kv:
        raise ValueError("require dv<=d and h_q divisible by h_kv")
    if block_table.shape[0] != b:
        raise ValueError("block_table batch mismatch")
    if cache_seqlens.numel() != b:
        raise ValueError("cache_seqlens must have one entry per request")
    if max_seqlen_pad % block_size:
        raise ValueError("max_seqlen_pad must be divisible by block_size")

    device = q.device
    out = torch.empty((b, s_q, h_q, dv), dtype=torch.float32, device=device)
    lse = torch.empty((b, h_q, s_q), dtype=torch.float32, device=device)
    qf = q.float()
    scale = 1.0 / math.sqrt(d)
    repeat = h_q // h_kv
    for bi in range(b):
        seqlen = int(cache_seqlens[bi].item())
        nblocks = (seqlen + block_size - 1) // block_size
        ids = block_table[bi, :nblocks].long()
        kv = blocked_k.index_select(0, ids).reshape(-1, h_kv, d)[:seqlen]
        k = kv.transpose(0, 1).float().repeat_interleave(repeat, dim=0)
        v = k[..., :dv]
        qi = qf[bi].transpose(0, 1)  # [Hq,Sq,D]
        scores = torch.matmul(qi, k.transpose(-1, -2)) * scale
        if causal:
            # FlashMLA aligns the query suffix with the cache sequence.
            mask = torch.ones((s_q, seqlen), dtype=torch.bool, device=device).tril(
                diagonal=seqlen - s_q
            )
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        lse_i = torch.logsumexp(scores, dim=-1)
        out_i = torch.softmax(scores, dim=-1, dtype=torch.float32).matmul(v)
        out[bi] = out_i.transpose(0, 1)
        lse[bi] = lse_i
    return out, lse


flash_mla = flash_mla_hygon_reference
