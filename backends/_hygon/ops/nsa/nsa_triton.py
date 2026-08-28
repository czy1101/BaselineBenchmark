"""GQA-packed/query-tiled Triton selected NSA kernel for FlagTree/HCU.

The kernel maps one program to one KV head and computes all query heads that
share that KV head. This keeps the reduction dimension in real matrix tiles
(`[G,D] x [D,64]` and `[G,64] x [64,DV]`) instead of issuing `M=1` dots.
It is opt-in from :mod:`nsa_hip` until validated on the target BW1000 image.
``NSA_HYGON_TRITON_QTILE`` groups adjacent query positions in one program;
the default of 1 preserves the validated baseline, while 2/4 are opt-in
for A/B testing on gfx936.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch


_BLOCK_CACHE = OrderedDict()
_BLOCK_CACHE_LIMIT = 3


def _pack_kv_blocks(k: torch.Tensor, v: torch.Tensor, block_size: int):
    """Cache block-major K/V layouts used by the fused selected kernel."""
    b, t, h, d = k.shape
    dv = v.shape[-1]
    nblocks = (t + block_size - 1) // block_size
    key = (int(k.data_ptr()), int(v.data_ptr()), str(k.device), str(k.dtype),
           tuple(k.shape), tuple(v.shape), int(block_size))
    cached = _BLOCK_CACHE.get(key)
    if cached is not None:
        _BLOCK_CACHE.move_to_end(key)
        return cached
    padded = nblocks * block_size
    if padded != t:
        pad = padded - t
        k_work = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v_work = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    else:
        k_work, v_work = k, v
    # K: [B,H,N,D,BS], V: [B,H,N,BS,DV].
    k_blocks = k_work.reshape(b, nblocks, block_size, h, d).permute(
        0, 3, 1, 4, 2).contiguous()
    v_blocks = v_work.reshape(b, nblocks, block_size, h, dv).permute(
        0, 3, 1, 2, 4).contiguous()
    cached = (k_blocks, v_blocks)
    _BLOCK_CACHE[key] = cached
    _BLOCK_CACHE.move_to_end(key)
    while len(_BLOCK_CACHE) > _BLOCK_CACHE_LIMIT:
        _BLOCK_CACHE.popitem(last=False)
    return cached


def _triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _selected_gqa_kernel(
        q_ptr, k_ptr, v_ptr, bi_ptr, out_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_kb, stride_kt, stride_kh, stride_kd,
        stride_kx,
        stride_vb, stride_vt, stride_vh, stride_vd,
        stride_vx,
        stride_ib, stride_it, stride_ih, stride_is,
        stride_ob, stride_ot, stride_oh, stride_od,
        T, NT, HQ, H, DV, block_counts, scale_log2,
        G: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        SLOTS: tl.constexpr, BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr, IS_BF16: tl.constexpr,
        Q_TILE: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        pid = tl.program_id(0)
        hv = pid % H
        tmp = pid // H
        t0 = tmp % NT
        b = tmp // NT

        qh = hv * G + tl.arange(0, G)
        d = tl.arange(0, BLOCK_D)
        dv = tl.arange(0, BLOCK_DV)
        offs = tl.arange(0, BLOCK_SIZE)

        # A program handles several adjacent query positions.  This reduces
        # scheduler/launch overhead while retaining per-token block tables.
        for qoff in range(0, Q_TILE):
            t = t0 * Q_TILE + qoff
            valid_t = t < T
            q_ptrs = (q_ptr + b * stride_qb + t * stride_qt +
                      qh[:, None] * stride_qh + d[None, :] * stride_qd)
            q_tile = tl.load(q_ptrs, mask=valid_t, other=0.0)

            acc = tl.zeros((G, BLOCK_DV), dtype=tl.float32)
            m_i = tl.full((G,), -float("inf"), dtype=tl.float32)
            l_i = tl.zeros((G,), dtype=tl.float32)

            for slot in range(0, SLOTS):
                block_id = tl.load(
                    bi_ptr + b * stride_ib + t * stride_it + hv * stride_ih +
                    slot * stride_is, mask=valid_t, other=0
                )
                tok = block_id * BLOCK_SIZE + offs
                valid = valid_t & (slot < block_counts) & (tok <= t) & (tok < T)

                if PACKED:
                    k_ptrs = (k_ptr + b * stride_kb + hv * stride_kh +
                              block_id * stride_kt + d[:, None] * stride_kd +
                              offs[None, :] * stride_kx)
                else:
                    k_ptrs = (k_ptr + b * stride_kb + tok[None, :] * stride_kt +
                              hv * stride_kh + d[:, None] * stride_kd)
                k_tile = tl.load(k_ptrs, mask=valid[None, :], other=0.0)
                scores = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale_log2
                scores = tl.where(valid[None, :], scores, -float("inf"))

                tile_m = tl.max(scores, axis=1)
                tile_has = tile_m > -1.0e20
                new_m = tl.maximum(m_i, tile_m)
                new_m = tl.where(tile_has, new_m, m_i)
                alpha = tl.where(tile_has, tl.exp2(m_i - new_m), 1.0)
                p = tl.where(tile_has[:, None], tl.exp2(scores - new_m[:, None]), 0.0)
                p = tl.where(valid[None, :], p, 0.0)

                if PACKED:
                    v_ptrs = (v_ptr + b * stride_vb + hv * stride_vh +
                              block_id * stride_vt + offs[:, None] * stride_vx +
                              dv[None, :] * stride_vd)
                else:
                    v_ptrs = (v_ptr + b * stride_vb + tok[:, None] * stride_vt +
                              hv * stride_vh + dv[None, :] * stride_vd)
                v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
                if IS_BF16:
                    pv = tl.dot(p.to(tl.bfloat16), v_tile,
                                out_dtype=tl.float32)
                else:
                    pv = tl.dot(p.to(tl.float16), v_tile,
                                out_dtype=tl.float32)
                acc = acc * alpha[:, None] + pv
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = new_m

            out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
            out_ptrs = (out_ptr + b * stride_ob + t * stride_ot +
                        qh[:, None] * stride_oh + dv[None, :] * stride_od)
            tl.store(out_ptrs, out, mask=valid_t)

    return _selected_gqa_kernel


def selected_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       block_indices: torch.Tensor, block_counts: int,
                       block_size: int, scale: float) -> torch.Tensor:
    """Run the GQA-packed selected attention kernel."""
    if not (q.is_cuda and k.is_cuda and v.is_cuda and block_indices.is_cuda):
        raise ValueError("Triton NSA requires CUDA/HCU tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
            block_indices.is_contiguous()):
        raise ValueError("Triton NSA requires contiguous tensors")

    b, t, hq, d = q.shape
    bk, tk, h, dk = k.shape
    bv, tv, hv, dv = v.shape
    if (b != bk or b != bv or t != tk or t != tv or h != hv or
            h != block_indices.shape[2] or d != dk or hq % h != 0 or
            block_indices.shape[:3] != (b, t, h)):
        raise ValueError("incompatible selected-attention shapes")
    if block_indices.dtype != torch.int32 or block_size != 64:
        raise ValueError("current Triton path requires int32 indices and block_size=64")

    slots = min(int(block_counts), int(block_indices.shape[-1]))
    if slots <= 0:
        return torch.zeros((b, t, hq, dv), device=q.device, dtype=q.dtype)
    g = hq // h
    if g not in (1, 2, 4, 8, 16, 32) or d not in (64, 128, 256, 512):
        raise ValueError("unsupported GQA or key tile dimension")
    if dv not in (64, 128, 256, 512):
        raise ValueError("unsupported value tile dimension")

    packed = os.environ.get("NSA_HYGON_TRITON_BLOCK_PACK", "0") == "1"
    if packed:
        k_run, v_run = _pack_kv_blocks(k, v, block_size)
    else:
        k_run, v_run = k, v
    out = torch.empty((b, t, hq, dv), device=q.device, dtype=torch.float32)
    kernel = _triton_kernel()
    # gfx936/BW1000 has better occupancy with two wavefront groups for this
    # register-heavy GQA tile. Keep an environment override for A/B tests.
    warps = int(os.environ.get("NSA_HYGON_TRITON_WARPS", "2"))
    # Keep the validated baseline as default. Query tiling remains opt-in for
    # workload-specific A/B testing because full-shape results can vary.
    q_tile = int(os.environ.get("NSA_HYGON_TRITON_QTILE", "1"))
    if q_tile not in (1, 2, 4, 8):
        raise ValueError("NSA_HYGON_TRITON_QTILE must be one of 1,2,4,8")
    ntiles = (t + q_tile - 1) // q_tile
    kernel[(b * ntiles * h,)](
        q, k_run, v_run, block_indices, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        *( (k_run.stride(0), k_run.stride(2), k_run.stride(1),
            k_run.stride(3), k_run.stride(4)) if packed else
           (k_run.stride(0), k_run.stride(1), k_run.stride(2),
            k_run.stride(3), 0) ),
        *( (v_run.stride(0), v_run.stride(2), v_run.stride(1),
            v_run.stride(4), v_run.stride(3)) if packed else
           (v_run.stride(0), v_run.stride(1), v_run.stride(2),
            v_run.stride(3), 0) ),
        block_indices.stride(0), block_indices.stride(1),
        block_indices.stride(2), block_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        t, ntiles, hq, h, dv, slots, float(scale) * 1.4426950408889634,
        G=g, BLOCK_SIZE=block_size, SLOTS=slots,
        BLOCK_D=d, BLOCK_DV=dv, IS_BF16=(q.dtype == torch.bfloat16),
        Q_TILE=q_tile,
        PACKED=packed,
        num_warps=warps,
    )
    return out.to(v.dtype)


__all__ = ["selected_attention"]
