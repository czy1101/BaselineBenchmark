"""Fused BF16 sparse-attention kernels for Hygon Triton-on-HIP.

The kernel keeps online-softmax state in registers and consumes paged K/V
directly.  It deliberately avoids the HIP fallback's per-query K/V packing
and the four large intermediate tensors (K pack, V pack, scores, probs).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _sparse_prefill_kernel(
    q_ptr,
    kv_ptr,
    topk_ptr,
    table_ptr,
    cu_ptr,
    lens_ptr,
    prefix_ptr,
    out_ptr,
    total_q: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    GROUP: tl.constexpr,
    MAX_QUERY: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE_LOG2E: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvh: tl.constexpr,
    stride_kvp: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_th: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tt: tl.constexpr,
    stride_bt: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    local = tl.program_id(0)
    kh = tl.program_id(1)
    req = tl.program_id(2)
    q_start = tl.load(cu_ptr + req)
    q_len = tl.load(cu_ptr + req + 1) - q_start
    if local >= q_len:
        return
    qid = q_start + local
    if qid >= total_q:
        return

    seq_len = tl.load(lens_ptr + req)
    qabs = tl.load(prefix_ptr + req) + local
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < GROUP
    q = tl.load(
        q_ptr
        + qid * stride_qn
        + (kh * GROUP + offs_h[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=hmask[:, None],
        other=0.0,
    )

    m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    for slot in range(TOPK):
        logical = tl.load(
            topk_ptr + kh * stride_th + qid * stride_tn + slot * stride_tt
        ).to(tl.int32)
        logical_ok = logical >= 0
        safe_logical = tl.maximum(logical, 0)
        page = tl.load(
            table_ptr + req * stride_bt + safe_logical,
            mask=logical_ok,
            other=0,
        ).to(tl.int64)
        token = logical * BLOCK_N + offs_n
        nmask = logical_ok & (token < seq_len) & (token <= qabs)
        base = kv_ptr + page * stride_kvb + kh * stride_kvh
        k = tl.load(
            base
            + offs_d[:, None] * stride_kvd
            + offs_n[None, :] * stride_kvp,
            mask=nmask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, k, out_dtype=tl.float32)
        logits = logits * SCALE_LOG2E
        logits = tl.where(hmask[:, None] & nmask[None, :], logits, float("-inf"))
        page_max = tl.max(logits, axis=1)
        page_has_value = page_max > -1.0e20
        m_new = tl.where(page_has_value, tl.maximum(m, page_max), m)
        alpha = tl.where(page_has_value, tl.exp2(m - m_new), 1.0)
        p = tl.where(
            page_has_value[:, None], tl.exp2(logits - m_new[:, None]), 0.0
        )
        p = tl.where(nmask[None, :] & hmask[:, None], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(
            base
            + offs_n[:, None] * stride_kvp
            + (BLOCK_D + offs_d[None, :]) * stride_kvd,
            mask=nmask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m = m_new

    result = acc / tl.where(l > 0.0, l, 1.0)[:, None]
    tl.store(
        out_ptr
        + qid * stride_on
        + (kh * GROUP + offs_h[:, None]) * stride_oh
        + offs_d[None, :] * stride_od,
        result,
        mask=hmask[:, None],
    )


@triton.jit
def _sparse_decode_kernel(
    q_ptr,
    kv_ptr,
    topk_ptr,
    table_ptr,
    lens_ptr,
    out_ptr,
    total_q: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    GROUP: tl.constexpr,
    DECODE_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE_LOG2E: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvh: tl.constexpr,
    stride_kvp: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_th: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tt: tl.constexpr,
    stride_bt: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    qid = tl.program_id(0)
    kh = tl.program_id(1)
    if qid >= total_q:
        return
    req = qid // DECODE_LEN
    local = qid - req * DECODE_LEN
    seq_len = tl.load(lens_ptr + req)
    qabs = seq_len - DECODE_LEN + local
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < GROUP
    q = tl.load(
        q_ptr
        + qid * stride_qn
        + (kh * GROUP + offs_h[:, None]) * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=hmask[:, None],
        other=0.0,
    )
    m = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    for slot in range(TOPK):
        logical = tl.load(
            topk_ptr + kh * stride_th + qid * stride_tn + slot * stride_tt
        ).to(tl.int32)
        logical_ok = logical >= 0
        safe_logical = tl.maximum(logical, 0)
        page = tl.load(
            table_ptr + req * stride_bt + safe_logical,
            mask=logical_ok,
            other=0,
        ).to(tl.int64)
        token = logical * BLOCK_N + offs_n
        nmask = logical_ok & (token <= qabs)
        base = kv_ptr + page * stride_kvb + kh * stride_kvh
        k = tl.load(
            base
            + offs_d[:, None] * stride_kvd
            + offs_n[None, :] * stride_kvp,
            mask=nmask[None, :],
            other=0.0,
        )
        logits = tl.dot(q, k, out_dtype=tl.float32) * SCALE_LOG2E
        logits = tl.where(hmask[:, None] & nmask[None, :], logits, float("-inf"))
        page_max = tl.max(logits, axis=1)
        page_has_value = page_max > -1.0e20
        m_new = tl.where(page_has_value, tl.maximum(m, page_max), m)
        alpha = tl.where(page_has_value, tl.exp2(m - m_new), 1.0)
        p = tl.where(
            page_has_value[:, None], tl.exp2(logits - m_new[:, None]), 0.0
        )
        p = tl.where(nmask[None, :] & hmask[:, None], p, 0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(
            base
            + offs_n[:, None] * stride_kvp
            + (BLOCK_D + offs_d[None, :]) * stride_kvd,
            mask=nmask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m = m_new
    result = acc / tl.where(l > 0.0, l, 1.0)[:, None]
    tl.store(
        out_ptr
        + qid * stride_on
        + (kh * GROUP + offs_h[:, None]) * stride_oh
        + offs_d[None, :] * stride_od,
        result,
        mask=hmask[:, None],
    )


def _launch_config(group: int) -> tuple[int, int, int]:
    # Matrix-core lowering requires a useful M tile. Padding G=6/12 to 16 is
    # substantially cheaper than falling back to scalar dot products.
    block_h = max(16, triton.next_power_of_2(group))
    default_warps = 2 if block_h <= 16 else 4
    warps = int(os.getenv("MSA_HYGON_TRITON_WARPS", default_warps))
    stages = int(os.getenv("MSA_HYGON_TRITON_STAGES", "1"))
    if warps not in (1, 2, 4, 8):
        raise ValueError("MSA_HYGON_TRITON_WARPS must be one of 1,2,4,8")
    if stages not in (1, 2, 3, 4):
        raise ValueError("MSA_HYGON_TRITON_STAGES must be in [1,4]")
    return block_h, warps, stages


def sparse_prefill(q, kv, topk, table, cu, lens, prefix, max_query, kv_heads, scale, out):
    group = q.shape[1] // kv_heads
    block_h, warps, stages = _launch_config(group)
    grid = (int(max_query), int(kv_heads), int(lens.numel()))
    _sparse_prefill_kernel[grid](
        q, kv, topk, table, cu, lens, prefix, out,
        q.shape[0], q.shape[1], kv_heads, group, int(max_query), topk.shape[2],
        float(scale) * LOG2E,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        topk.stride(0), topk.stride(1), topk.stride(2), table.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=block_h, BLOCK_D=128, BLOCK_N=128,
        num_warps=warps, num_stages=stages,
    )


def sparse_decode(q, kv, topk, table, lens, kv_heads, scale, out, decode_len):
    group = q.shape[1] // kv_heads
    block_h, warps, stages = _launch_config(group)
    grid = (q.shape[0], int(kv_heads))
    _sparse_decode_kernel[grid](
        q, kv, topk, table, lens, out,
        q.shape[0], q.shape[1], kv_heads, group, int(decode_len), topk.shape[2],
        float(scale) * LOG2E,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        topk.stride(0), topk.stride(1), topk.stride(2), table.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=block_h, BLOCK_D=128, BLOCK_N=128,
        num_warps=warps, num_stages=stages,
    )


__all__ = ["sparse_prefill", "sparse_decode"]
