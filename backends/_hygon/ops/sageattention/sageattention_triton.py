"""SageAttention INT8-QK/FP16-PV kernels for FlagTree Triton-on-HIP."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634


@triton.jit
def _quant_block_kernel(
    x_ptr, y_ptr, scale_ptr, length,
    sx0: tl.constexpr, sxh: tl.constexpr, sxn: tl.constexpr,
    sy0: tl.constexpr, syh: tl.constexpr, syn: tl.constexpr,
    ss0: tl.constexpr, ssh: tl.constexpr,
    MULT: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    rn = block * BLOCK + tl.arange(0, BLOCK)
    rd = tl.arange(0, D)
    mask = rn[:, None] < length
    x = tl.load(
        x_ptr + batch * sx0 + head * sxh + rn[:, None] * sxn + rd[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32) * MULT
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.max(amax, axis=0)
    scale = tl.maximum(amax / 127.0, 1.0e-12)
    z = x / scale
    z = z + 0.5 * tl.where(z >= 0.0, 1.0, -1.0)
    tl.store(
        y_ptr + batch * sy0 + head * syh + rn[:, None] * syn + rd[None, :],
        z.to(tl.int8), mask=mask,
    )
    tl.store(scale_ptr + batch * ss0 + head * ssh + block, scale)


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, qs_ptr, ks_ptr, out_ptr, mask_ptr, lse_ptr,
    q_len, kv_len,
    sq0: tl.constexpr, sqh: tl.constexpr, sqn: tl.constexpr,
    sk0: tl.constexpr, skh: tl.constexpr, skn: tl.constexpr,
    sv0: tl.constexpr, svh: tl.constexpr, svn: tl.constexpr,
    so0: tl.constexpr, soh: tl.constexpr, son: tl.constexpr,
    sm0: tl.constexpr, smh: tl.constexpr, smm: tl.constexpr,
    smn: tl.constexpr, QH: tl.constexpr, GROUP: tl.constexpr,
    D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASK_KIND: tl.constexpr, RETURN_LSE: tl.constexpr,
):
    block_m = tl.program_id(0)
    qh = tl.program_id(1)
    batch = tl.program_id(2)
    kvh = qh // GROUP
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, D)
    qmask = rm < q_len
    q = tl.load(
        q_ptr + batch * sq0 + qh * sqh + rm[:, None] * sqn + rd[None, :],
        mask=qmask[:, None], other=0,
    )
    q_blocks = tl.cdiv(q_len, BLOCK_M)
    k_blocks = tl.cdiv(kv_len, BLOCK_N)
    q_scale = tl.load(qs_ptr + (batch * QH + qh) * q_blocks + block_m)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    for block_n in range(0, k_blocks):
        start_n = block_n * BLOCK_N
        nmask = rn < kv_len - start_n
        k = tl.load(
            k_ptr + batch * sk0 + kvh * skh
            + rd[:, None] + (start_n + rn[None, :]) * skn,
            mask=nmask[None, :], other=0,
        )
        k_scale = tl.load(
            ks_ptr + (batch * (QH // GROUP) + kvh) * k_blocks + block_n
        )
        logits = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        logits *= q_scale * k_scale
        logits = tl.where(qmask[:, None] & nmask[None, :],
                          logits, float("-inf"))
        if MASK_KIND == 1:
            allowed = tl.load(
                mask_ptr + batch * sm0 + qh * smh + rm[:, None] * smm
                + (start_n + rn[None, :]) * smn,
                mask=qmask[:, None] & nmask[None, :], other=False,
            )
            logits = tl.where(allowed, logits, float("-inf"))
        elif MASK_KIND == 2:
            bias = tl.load(
                mask_ptr + batch * sm0 + qh * smh + rm[:, None] * smm
                + (start_n + rn[None, :]) * smn,
                mask=qmask[:, None] & nmask[None, :], other=float("-inf"),
            ).to(tl.float32)
            logits += bias

        page_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, page_max)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(logits - m_new[:, None])
        p = tl.where(qmask[:, None] & nmask[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        v = tl.load(
            v_ptr + batch * sv0 + kvh * svh
            + (start_n + rn[:, None]) * svn + rd[None, :],
            mask=nmask[:, None], other=0.0,
        )
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    result = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(
        out_ptr + batch * so0 + qh * soh + rm[:, None] * son + rd[None, :],
        result, mask=qmask[:, None],
    )
    if RETURN_LSE:
        lse = tl.log2(tl.where(l_i > 0.0, l_i, 1.0)) + m_i
        tl.store(lse_ptr + (batch * QH + qh) * q_len + rm, lse, mask=qmask)


def _layout(tensor, tensor_layout):
    if tensor_layout == "HND":
        return tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.stride(0), tensor.stride(1), tensor.stride(2)
    if tensor_layout == "NHD":
        return tensor.shape[0], tensor.shape[2], tensor.shape[1], tensor.shape[3], tensor.stride(0), tensor.stride(2), tensor.stride(1)
    raise ValueError(f"tensor_layout {tensor_layout} not supported")


def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    if km is not None:
        k = k - km
    b, qh, q_len, dim, q0, qhs, qn = _layout(q, tensor_layout)
    _, kh, kv_len, kdim, k0, khs, kn = _layout(k, tensor_layout)
    if dim != kdim:
        raise ValueError("q and k head dimensions differ")
    qi = torch.empty_like(q, dtype=torch.int8)
    ki = torch.empty_like(k, dtype=torch.int8)
    _, _, _, _, qi0, qihs, qin = _layout(qi, tensor_layout)
    _, _, _, _, ki0, kihs, kin = _layout(ki, tensor_layout)
    qs = torch.empty((b, qh, triton.cdiv(q_len, BLKQ)), device=q.device, dtype=torch.float32)
    ks = torch.empty((b, kh, triton.cdiv(kv_len, BLKK)), device=k.device, dtype=torch.float32)
    sm_scale = dim ** -0.5 if sm_scale is None else float(sm_scale)
    _quant_block_kernel[(triton.cdiv(q_len, BLKQ), qh, b)](
        q, qi, qs, q_len, q0, qhs, qn, qi0, qihs, qin,
        qs.stride(0), qs.stride(1), MULT=sm_scale * LOG2E,
        D=dim, BLOCK=BLKQ, num_warps=4,
    )
    _quant_block_kernel[(triton.cdiv(kv_len, BLKK), kh, b)](
        k, ki, ks, kv_len, k0, khs, kn, ki0, kihs, kin,
        ks.stride(0), ks.stride(1), MULT=1.0,
        D=dim, BLOCK=BLKK, num_warps=4,
    )
    return qi, qs, ki, ks


def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
            output_dtype=torch.float16, return_lse=False, maxnreg=None):
    del maxnreg  # HIPOptions on BW1000 does not support NVIDIA maxnreg.
    b, qh, q_len, dim, q0, qhs, qn = _layout(q, tensor_layout)
    _, kh, kv_len, kdim, k0, khs, kn = _layout(k, tensor_layout)
    _, vkh, v_len, vdim, v0, vhs, vn = _layout(v, tensor_layout)
    if kdim != dim or vdim != dim or v_len != kv_len or vkh != kh:
        raise ValueError("incompatible q/k/v shapes")
    if qh % kh:
        raise ValueError("query heads must be divisible by KV heads")
    out = torch.empty(q.shape, device=q.device, dtype=output_dtype)
    _, _, _, _, o0, ohs, on = _layout(out, tensor_layout)
    lse = torch.empty((b, qh, q_len), device=q.device, dtype=torch.float32) if return_lse else torch.empty(0, device=q.device)
    if attn_mask is None:
        mask_kind, mask_arg = 0, q
        sm0 = smh = smm = smn = 0
    else:
        mask_arg = attn_mask
        mask_kind = 1 if attn_mask.dtype == torch.bool else 2
        sm0, smh, smm, smn = attn_mask.stride()
    # BLOCK_M / BLOCK_N are coupled to the quantisation block sizes (BLKQ=128,
    # BLKK=64): q_scale has one entry per 128 query tokens and k_scale one per
    # 64 KV tokens.  Changing either here alone de-quantises with the wrong
    # scale silently, so they must move together with per_block_int8().
    block_m, block_n = 128, 64
    warps = int(os.getenv("SAGEATTENTION_HYGON_WARPS", "4" if dim == 64 else "8"))
    # num_stages must stay <= 3 on BW1000.  With 4 the HCU software pipeliner
    # destroys the scalar k_scale load inside the K loop while its result is
    # still live, aborting compilation with
    #   "LLVM ERROR: operation destroyed but still has uses"
    # on every head_dim=128 launch (and any config pipelined at 4 stages).
    stages = int(os.getenv("SAGEATTENTION_HYGON_STAGES", "3"))
    stages = min(stages, 3)
    _attention_kernel[(triton.cdiv(q_len, block_m), qh, b)](
        q, k, v, q_scale, k_scale, out, mask_arg, lse, q_len, kv_len,
        q0, qhs, qn, k0, khs, kn, v0, vhs, vn, o0, ohs, on,
        sm0, smh, smm, smn, QH=qh, GROUP=qh // kh, D=dim,
        BLOCK_M=block_m, BLOCK_N=block_n, MASK_KIND=mask_kind,
        RETURN_LSE=return_lse, num_warps=warps, num_stages=stages,
    )
    return out, lse


__all__ = ["forward", "per_block_int8"]
