"""Differentiable PyTorch/DTK semantic reference for FlagGems chunk_gla."""

from __future__ import annotations

import math
from typing import Optional

import torch


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    state_v_first: bool,
    cu_seqlens: Optional[torch.Tensor],
) -> None:
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/g and v must be rank-4 BTHD tensors")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k and g must have identical [B,T,H,K] shapes")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("v must share B,T,H with q")
    tensors = (q, k, v, g)
    if any(not x.is_cuda for x in tensors):
        raise ValueError("all inputs must be on the Hygon HIP device")
    if any(x.device != q.device for x in tensors):
        raise ValueError("all inputs must be on the same device")
    if any(x.dtype != q.dtype for x in tensors):
        raise ValueError("q, k, v and g must share dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes are float16, bfloat16 and float32")

    B, _, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    if cu_seqlens is not None and B != 1:
        raise ValueError("packed variable-length mode requires B=1")
    if initial_state is not None:
        if initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be float32")
        expected = (N, H, V, K) if state_v_first else (N, H, K, V)
        if tuple(initial_state.shape) != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        if initial_state.device != q.device:
            raise ValueError("initial_state must share the input device")


def _one_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run [T,H,*] inputs with an FP32 [H,K,V] recurrent state."""
    outputs = []
    for ti in range(q.shape[0]):
        qf = q[ti].float()
        kf = k[ti].float()
        vf = v[ti].float()
        decay = torch.exp(g[ti].float()).unsqueeze(-1)
        state = state * decay + kf.unsqueeze(-1) * vf.unsqueeze(-2)
        outputs.append(torch.einsum("hk,hkv->hv", qf, state) * scale)
    if outputs:
        out = torch.stack(outputs, dim=0).to(v.dtype)
    else:
        out = v.new_empty((0, v.shape[-2], v.shape[-1]))
    return out, state


def chunk_gla_hygon(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Full public chunk_gla semantics using only installed PyTorch/DTK.

    This intentionally favors clarity and differentiability over speed.  It is
    the correctness oracle for the fused HIP forward/backward implementation.
    """
    _validate(q, k, v, g, initial_state, state_v_first, cu_seqlens)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    B, T, H, K = q.shape
    V = v.shape[-1]

    h0 = initial_state
    if h0 is not None and state_v_first:
        h0 = h0.transpose(-1, -2)

    outputs = []
    finals = []
    if cu_seqlens is None:
        for bi in range(B):
            state = (
                torch.zeros(H, K, V, device=q.device, dtype=torch.float32)
                if h0 is None
                else h0[bi]
            )
            out, state = _one_sequence(
                q[bi], k[bi], v[bi], g[bi], state, scale
            )
            outputs.append(out)
            finals.append(state)
        o = torch.stack(outputs, dim=0)
    else:
        offsets_source = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        offsets = [int(x) for x in offsets_source.detach().cpu().tolist()]
        if not offsets or offsets[0] != 0 or offsets[-1] != T:
            raise ValueError("cu_seqlens must start at 0 and end at packed T")
        if any(e < s for s, e in zip(offsets, offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        for ni, (bos, eos) in enumerate(zip(offsets, offsets[1:])):
            state = (
                torch.zeros(H, K, V, device=q.device, dtype=torch.float32)
                if h0 is None
                else h0[ni]
            )
            out, state = _one_sequence(
                q[0, bos:eos], k[0, bos:eos], v[0, bos:eos],
                g[0, bos:eos], state, scale,
            )
            outputs.append(out)
            finals.append(state)
        o = torch.cat(outputs, dim=0).unsqueeze(0)

    if not output_final_state:
        return o, None
    ht = torch.stack(finals, dim=0)
    if state_v_first:
        ht = ht.transpose(-1, -2).contiguous()
    return o, ht


chunk_gla = chunk_gla_hygon
