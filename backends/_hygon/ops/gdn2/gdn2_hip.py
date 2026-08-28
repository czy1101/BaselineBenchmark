"""JIT loader and Python API for the pure HIP GDN2 extension."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_EXT = None


def load_extension(verbose: bool = False):
    global _EXT
    if _EXT is None:
        source = Path(__file__).with_name("gdn2_hip_kernel.cu")
        _EXT = load(
            # Bump when the C++/HIP extension ABI gains exported methods.
            # This avoids importing a stale same-named .so from PyTorch's JIT
            # cache while developing the staged BT64 Chunk/WY path.
            name="gdn2_hygon_hip_ext_chunk13",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-ffast-math"],
            verbose=verbose,
        )
    return _EXT


@torch.no_grad()
def chunk_gdn2_hip(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, b: torch.Tensor, w: torch.Tensor, *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    chunk_size: int = 16,
    return_intermediate_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    **kwargs,
):
    del cu_seqlens_cpu, chunk_indices, kwargs
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q/k/g/b must have identical [B,T,H,K] shapes")
    if v.shape != w.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("v/w must be [B,T,H,V] and share B,T,H with q")
    if q.shape[-1] not in (64, 128, 256):
        raise ValueError("HIP optimized path supports K=64,128,256")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        x = g.float()
        H, K = g.shape[-2:]
        if dt_bias is not None:
            x = x + dt_bias.reshape(H, K).float()
        A = A_log.reshape(H, 1).float().exp()
        if safe_gate or lower_bound is not None:
            if lower_bound is None or not (-5.0 <= lower_bound < 0.0):
                raise ValueError("lower_bound must be in [-5,0) for safe gate")
            x = float(lower_bound) * torch.sigmoid(A * x)
        else:
            x = -A * F.softplus(x)
        g = x.to(q.dtype)

    if initial_state is not None:
        if initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be float32")
        if state_v_first:
            initial_state = initial_state.transpose(-1, -2).contiguous()

    ext = load_extension()

    def run(qs, ks, vs, gs, bs, ws, h0):
        return ext.forward(
            qs.contiguous(), ks.contiguous(), vs.contiguous(), gs.contiguous(),
            bs.contiguous(), ws.contiguous(), h0, scale,
        )

    # Fast common path: one fused launch for regular fixed-length inference.
    if cu_seqlens is None and not return_intermediate_states:
        o, ht = run(q, k, v, g, b, w, initial_state)
    else:
        B, T, H, K = q.shape
        V = v.shape[-1]
        if cu_seqlens is None:
            sequences = [(bi, 0, T) for bi in range(B)]
        else:
            if B != 1:
                raise ValueError("packed varlen mode requires B=1")
            offsets = cu_seqlens.detach().cpu().tolist()
            sequences = [(0, int(s), int(e)) for s, e in zip(offsets, offsets[1:])]
        o = torch.empty_like(v)
        finals, sequence_states = [], []
        for ni, (bi, bos, eos) in enumerate(sequences):
            hcur = None if initial_state is None else initial_state[ni:ni+1]
            chunk_states = []
            for start in range(bos, eos, chunk_size):
                end = min(start + chunk_size, eos)
                if return_intermediate_states:
                    if hcur is None:
                        hcur = torch.zeros(1,H,K,V,device=q.device,dtype=torch.float32)
                    chunk_states.append(hcur[0].to(torch.bfloat16))
                oc, hcur = run(
                    q[bi:bi+1,start:end], k[bi:bi+1,start:end],
                    v[bi:bi+1,start:end], g[bi:bi+1,start:end],
                    b[bi:bi+1,start:end], w[bi:bi+1,start:end], hcur,
                )
                o[bi:bi+1,start:end] = oc
            finals.append(hcur[0])
            if return_intermediate_states:
                sequence_states.append(torch.stack(chunk_states))
        ht = torch.stack(finals)

    final = ht.transpose(-1, -2).contiguous() if state_v_first else ht
    if not output_final_state:
        final = None
    if return_intermediate_states:
        # Fixed-length matches [N,NT,H,K,V]. Varlen is padded to max NT.
        max_nt = max(x.shape[0] for x in sequence_states)
        hs = torch.zeros(len(sequence_states),max_nt,H,K,V,device=q.device,dtype=torch.bfloat16)
        for ni, x in enumerate(sequence_states):
            hs[ni,:x.shape[0]] = x
        if state_v_first:
            hs = hs.transpose(-1, -2).contiguous()
        return o, final, hs
    return o, final


chunk_gdn2 = chunk_gdn2_hip
