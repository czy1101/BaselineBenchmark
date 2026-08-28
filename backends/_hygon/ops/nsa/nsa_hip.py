"""NSA implementation for Hygon.

The fixed-length selected/compression paths use the native wave64 HIP kernel
in :mod:`nsa_hip_kernel.cu`.  The Python implementation remains as a fallback
for varlen, window and gated composite calls which are not part of that hot
path.  This keeps the public FlagGems API while avoiding the enormous Python
gather/matmul launch overhead on BW1000.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load


_EXT = None


def load_extension(verbose: bool = False):
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="nsa_hygon_hip_ext_v1_wave64",
            sources=[str(Path(__file__).with_name("nsa_hip_kernel.cu"))],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=verbose,
        )
    return _EXT


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("incompatible batch/sequence dimensions")
    if k.shape[2] != v.shape[2] or k.shape[3] != q.shape[3]:
        raise ValueError("incompatible key/value head or key dimensions")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("query heads must be divisible by key/value heads")


def _mean_pool(k: torch.Tensor, v: torch.Tensor, block_size: int,
               cu_seqlens: Optional[torch.Tensor] = None):
    """Mean-pool K/V into compression blocks, retaining input dtype."""
    if cu_seqlens is None:
        b, t, h, d = k.shape
        n = (t + block_size - 1) // block_size
        pad = n * block_size - t
        if pad:
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
        return (k.reshape(b, n, block_size, h, d).float().mean(2).to(k.dtype),
                v.reshape(b, n, block_size, h, v.shape[-1]).float().mean(2).to(v.dtype))

    if k.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    offsets = cu_seqlens.detach().cpu().tolist()
    parts_k, parts_v = [], []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        ki, vi = _mean_pool(k[:, s:e], v[:, s:e], block_size)
        parts_k.append(ki)
        parts_v.append(vi)
    return torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)


def _compression_fixed(q, k, v, block_size, scale, tile_q=64):
    """Compression attention for one fixed-length sequence batch."""
    b, t, hq, d = q.shape
    tc = k.shape[1]
    h = k.shape[2]
    g = hq // h
    # Keep Q/K/V in their native FP16/BF16 type for DTK MFMA.  Converting the
    # entire attention tile to FP32 forces a much slower vector GEMM.  We only
    # promote the score/softmax and final accumulation where numerically
    # needed.
    qf, kf, vf = q, k, v
    # Head expansion is small (G is normally 16) and lets matmul select DTK.
    kh = kf.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    vh = vf.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    out = torch.zeros(b, t, hq, v.shape[-1], device=q.device, dtype=torch.float32)
    lse = torch.zeros(b, t, hq, device=q.device, dtype=torch.float32)
    block_ids = torch.arange(tc, device=q.device)

    for t0 in range(0, t, tile_q):
        t1 = min(t, t0 + tile_q)
        qt = qf[:, t0:t1].permute(0, 2, 1, 3)  # [B,HQ,Q,D]
        scores = torch.matmul(qt, kh.transpose(-1, -2)) * scale
        pos = torch.arange(t0, t1, device=q.device)
        # The reference exposes a compressed block only after it is complete.
        valid = block_ids[None, :] < ((pos[:, None] + 1) // block_size)
        valid = valid[None, None, :, :]
        safe_scores = scores.masked_fill(~valid, -1.0e30)
        has = valid.any(-1)
        weights = torch.softmax(safe_scores, dim=-1)
        weights = torch.where(has[..., None], weights, torch.zeros_like(weights))
        ot = torch.matmul(weights, vh)
        lt = torch.logsumexp(safe_scores, dim=-1)
        lt = torch.where(has, lt, torch.zeros_like(lt))
        out[:, t0:t1] = ot.permute(0, 2, 1, 3)
        lse[:, t0:t1] = lt.permute(0, 2, 1)
    return out.to(v.dtype), lse


def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """NSA compressed attention, matching FlagGems' public API.

    ``q`` is ``[B,T,HQ,D]`` and compressed ``k/v`` are ``[B,TC,H,D/V]``.
    Returns ``(output, lse)``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is None:
        # Production fixed-length path: one wave per query head, online
        # softmax and direct K/V loads in the HIP kernel.
        if (q.is_cuda and k.is_cuda and v.is_cuda and q.is_contiguous() and
                k.is_contiguous() and v.is_contiguous() and
                q.dtype in (torch.float16, torch.bfloat16) and
                k.shape[0] == q.shape[0] and k.shape[2] == v.shape[2] and
                q.shape[2] % k.shape[2] == 0 and q.shape[-1] == k.shape[-1] and
                v.shape[-1] <= 256 and q.shape[-1] <= 256):
            return load_extension(False).nsa_compression(
                q, k, v, int(block_size), float(scale))
        return _compression_fixed(q, k, v, block_size, float(scale))
    if q.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    offsets = cu_seqlens.detach().cpu().tolist()
    oq, ol = [], []
    k_off = 0
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        n = (e - s + block_size - 1) // block_size
        oi, li = _compression_fixed(
            q[:, s:e], k[:, k_off:k_off + n], v[:, k_off:k_off + n],
            block_size, float(scale)
        )
        oq.append(oi)
        ol.append(li)
        k_off += n
    return torch.cat(oq, dim=1), torch.cat(ol, dim=1)


def _selected_fixed(q, k, v, block_indices, block_counts, block_size,
                    scale, tile_q=None):
    """Selected sparse attention for fixed-length inputs."""
    b, t, hq, d = q.shape
    h, dv = v.shape[2], v.shape[3]
    g = hq // h

    # Opt-in Triton-on-HCU fused path.  Keep the DTK path unchanged by
    # default so existing correctness/performance numbers remain reproducible.
    if (os.environ.get("NSA_HYGON_USE_TRITON", "0") == "1" and
            isinstance(block_counts, int) and q.is_cuda and k.is_cuda and
            v.is_cuda and block_indices.is_cuda and
            q.dtype in (torch.float16, torch.bfloat16) and
            block_size == 64 and
            q.shape[-1] in (64, 128, 256, 512) and
            v.shape[-1] in (64, 128, 256, 512) and
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
            block_indices.is_contiguous() and block_indices.dtype == torch.int32):
        try:
            from nsa_triton import selected_attention
            return selected_attention(q, k, v, block_indices, int(block_counts),
                                      int(block_size), float(scale))
        except (ImportError, RuntimeError, ValueError):
            if os.environ.get("NSA_HYGON_TRITON_STRICT", "0") == "1":
                raise
    if tile_q is None:
        # Larger tiles amortize gather/GEMM launch overhead on long context
        # workloads.  Keep the tile bounded for the B=4 production shape.
        # 512 is the best measured working set for the current DTK GEMM
        # implementation.  Larger tiles exceed its cache/tiling sweet spot
        # and regress despite fewer launches; retain 128 for B>1.
        tile_q = 512 if b == 1 else 128
    smax = block_indices.shape[-1]
    # On the current BW1000 DTK build, FP32 bmm is the tuned path; native
    # FP16/BF16 bmm falls back to a slower implementation.  Keep the GEMM
    # operands in FP32 and cast only the final result back to input dtype.
    qf, kf, vf = q.float(), k.float(), v.float()
    # [B,H,T,D] and [B,H,T,DV] provide a gather-friendly layout.
    kh = kf.permute(0, 2, 1, 3)
    vh = vf.permute(0, 2, 1, 3)
    out = torch.zeros(b, t, hq, dv, device=q.device, dtype=torch.float32)
    offs = torch.arange(block_size, device=q.device)
    if isinstance(block_counts, int):
        counts = None
        count_scalar = min(int(block_counts), smax)
    else:
        counts = block_counts.to(device=q.device)
        count_scalar = smax

    for t0 in range(0, t, tile_q):
        t1 = min(t, t0 + tile_q)
        qt = qf[:, t0:t1].reshape(b, t1 - t0, h, g, d).permute(0, 2, 3, 1, 4)
        bi = block_indices[:, t0:t1].permute(0, 2, 1, 3).long()
        if counts is None:
            slot_valid = (torch.arange(smax, device=q.device) < count_scalar).view(1, 1, 1, smax)
        else:
            cv = counts[:, t0:t1].permute(0, 2, 1).long()
            slot_valid = torch.arange(smax, device=q.device).view(1, 1, 1, smax) < cv[..., None]
        token_idx = bi[..., None] * block_size + offs
        token_idx = token_idx.reshape(b, h, t1 - t0, smax * block_size)
        in_range = token_idx < t
        slot_valid = slot_valid[..., None].expand(b, h, t1 - t0, smax, block_size)
        slot_valid = slot_valid.reshape(b, h, t1 - t0, smax * block_size)
        token_idx_safe = token_idx.clamp(0, max(t - 1, 0))
        if os.environ.get("NSA_HYGON_PACK", "0") == "1":
            kt, vt = load_extension(False).nsa_pack_selected(
                k, v, block_indices, int(t0), int(t1 - t0), int(block_size)
            )
            kt, vt = kt.float(), vt.float()
        else:
            gather_idx = token_idx_safe[..., None].expand(b, h, t1 - t0, smax * block_size, d)
            kt = torch.gather(
                kh[:, :, None, :, :].expand(b, h, t1 - t0, t, d), 3, gather_idx
            )
            gather_idx_v = token_idx_safe[..., None].expand(b, h, t1 - t0, smax * block_size, dv)
            vt = torch.gather(
                vh[:, :, None, :, :].expand(b, h, t1 - t0, t, dv), 3, gather_idx_v
            )
        # Keep the GQA dimension inside GEMM instead of flattening it into
        # the batch.  The old layout issued millions of 1xD GEMMs; this one
        # issues [G,D]x[D,L] GEMMs, which is the shape DTK can lower to MFMA.
        qcount = t1 - t0
        nmat = b * h * qcount
        qmat = qt.permute(0, 1, 3, 2, 4).reshape(nmat, g, d)
        kmat = kt.reshape(nmat, smax * block_size, d).transpose(1, 2)
        scores = torch.bmm(qmat, kmat).reshape(
            b, h, qcount, g, smax * block_size
        ).permute(0, 1, 3, 2, 4).float() * scale
        qpos = torch.arange(t0, t1, device=q.device).view(1, 1, 1, t1 - t0, 1)
        causal = token_idx_safe[:, :, None, :, :] <= qpos
        valid = slot_valid[:, :, None, :, :] & in_range[:, :, None, :, :] & causal
        safe = scores.masked_fill(~valid, -1.0e30)
        has = valid.any(-1)
        weights = torch.softmax(safe, dim=-1)
        weights = torch.where(has[..., None], weights, torch.zeros_like(weights))
        wmat = weights.permute(0, 1, 3, 2, 4).reshape(
            nmat, g, smax * block_size
        )
        vmat = vt.reshape(nmat, smax * block_size, dv)
        ot = torch.bmm(wmat, vmat).float().reshape(
            b, h, qcount, g, dv
        ).permute(0, 1, 3, 2, 4)
        out[:, t0:t1] = ot.permute(0, 3, 1, 2, 4).reshape(b, t1 - t0, hq, dv)
    return out.to(v.dtype)


def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor | None,
    block_counts: int,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Portable Top-K block selector used when ``g_cmp`` is supplied."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    b, t, hq, d = q.shape
    h = k.shape[2]
    g = hq // h
    s = min(int(block_counts), max(1, k.shape[1]))
    # Importance is computed against compressed representations.  We keep
    # this path intentionally simple; production callers normally pass the
    # precomputed block_indices and skip Top-K selection.
    scores = torch.einsum(
        "bthgd,bnhd->bthgn", q.float().reshape(b, t, h, g, d), k.float()
    ) * scale
    scores = scores.mean(3)  # [B,T,H,TC]
    tc = k.shape[1]
    pos = torch.arange(t, device=q.device)
    valid = torch.arange(tc, device=q.device)[None, :] < ((pos[:, None] + 1) // block_size)
    scores = scores.masked_fill(~valid[None, :, None, :], -1.0e30)
    top = scores.topk(s, dim=-1).indices
    return top.to(torch.int32)


def _window_attention(q, k, v, window_size, scale, tile_q=32):
    """Small portable sliding-window path used when flash-attn is absent."""
    b, t, hq, d = q.shape
    h, dv = k.shape[2], v.shape[3]
    g = hq // h
    qf, kf, vf = q.float(), k.float(), v.float()
    kh = kf.permute(0, 2, 1, 3).repeat_interleave(g, 1)
    vh = vf.permute(0, 2, 1, 3).repeat_interleave(g, 1)
    out = torch.zeros(b, t, hq, dv, device=q.device, dtype=torch.float32)
    for t0 in range(0, t, tile_q):
        t1 = min(t, t0 + tile_q)
        qt = qf[:, t0:t1].permute(0, 2, 1, 3)
        ids = torch.arange(t, device=q.device)
        valid = (ids[None, :] <= torch.arange(t0, t1, device=q.device)[:, None])
        valid &= ids[None, :] >= torch.arange(t0, t1, device=q.device)[:, None] - window_size + 1
        sc = torch.matmul(qt, kh.transpose(-1, -2)) * scale
        sc = sc.masked_fill(~valid[None, None], -1.0e30)
        out[:, t0:t1] = torch.matmul(torch.softmax(sc, -1), vh).permute(0, 2, 1, 3)
    return out.to(v.dtype)


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    block_counts: int | torch.Tensor = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Native Sparse Attention forward path matching FlagGems semantics."""
    _check_qkv(q, k, v)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if block_indices is None:
        if g_cmp is None:
            raise ValueError("block_indices is required when g_cmp is absent")
        k_cmp, v_cmp = _mean_pool(k, v, block_size, cu_seqlens)
        _, lse = parallel_nsa_compression(q, k_cmp, v_cmp, block_size, scale, cu_seqlens)
        block_indices = parallel_nsa_topk(
            q, k_cmp, lse, int(block_counts) if isinstance(block_counts, int) else int(block_counts.max()),
            block_size, scale, cu_seqlens
        )

    if cu_seqlens is None:
        # The scalar wave64 kernel is retained for correctness experiments,
        # but is not the production path: it cannot use BW1000 MFMA.  The
        # DTK-backed tiled bmm path is substantially faster for HQ=256 and
        # remains the default until a fused GEMM kernel is available.
        use_wave64 = os.environ.get("NSA_HYGON_USE_WAVE64", "0") == "1"
        if (use_wave64 and g_cmp is None and g_slc is None and g_swa is None and
                block_indices is not None and isinstance(block_counts, int) and
                q.is_cuda and k.is_cuda and v.is_cuda and
                block_indices.is_cuda and block_indices.dtype == torch.int32 and
                q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
                block_indices.is_contiguous() and
                q.dtype in (torch.float16, torch.bfloat16) and
                q.shape[-1] <= 256 and v.shape[-1] <= 256):
            return load_extension(False).nsa_forward(
                q, k, v, block_indices, int(block_counts), int(block_size),
                float(scale))
        o_slc = _selected_fixed(q, k, v, block_indices, block_counts, block_size, float(scale))
        o = o_slc if g_slc is None else o_slc * g_slc.unsqueeze(-1).to(o_slc.dtype)
        if g_cmp is not None:
            k_cmp, v_cmp = _mean_pool(k, v, block_size)
            o_cmp, _ = _compression_fixed(q, k_cmp, v_cmp, block_size, float(scale))
            o = o + o_cmp * g_cmp.unsqueeze(-1).to(o.dtype)
        if window_size > 0:
            o_swa = _window_attention(q, k, v, window_size, float(scale))
            gate = g_swa if g_swa is not None else 1.0
            o = o + o_swa * (gate.unsqueeze(-1) if torch.is_tensor(gate) else gate)
        return o.to(q.dtype)

    if q.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    offsets = cu_seqlens.detach().cpu().tolist()
    outs = []
    for i in range(len(offsets) - 1):
        s0, e0 = int(offsets[i]), int(offsets[i + 1])
        ci = None if block_indices is None else block_indices[:, s0:e0]
        gi = lambda x: None if x is None else x[:, s0:e0]
        outs.append(parallel_nsa(
            q[:, s0:e0], k[:, s0:e0], v[:, s0:e0], gi(g_cmp), gi(g_slc), gi(g_swa),
            ci, block_counts, block_size, window_size, scale, None
        ))
    return torch.cat(outs, dim=1)


# Names used by the Hygon baseline harness.
nsa_forward = parallel_nsa
nsa_compression = parallel_nsa_compression

__all__ = [
    "parallel_nsa", "parallel_nsa_compression", "parallel_nsa_topk",
    "nsa_forward", "nsa_compression",
]
