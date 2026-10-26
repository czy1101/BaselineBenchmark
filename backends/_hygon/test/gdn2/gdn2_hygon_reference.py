"""Portable GDN2 inference reference for Hygon DCU / PyTorch-HIP.

This is a correctness implementation, not the final performance kernel.  It
contains no CUDA PTX, Triton TLE, TMA, or NVIDIA-specific assumptions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _effective_log_decay(
    g: torch.Tensor,
    *,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Return the per-token, per-K natural-log decay in float32."""
    x = g.float()
    if not use_gate_in_kernel:
        return x

    if A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    if dt_bias is not None:
        # Hopper accepts [H*K]; also accept the more readable [H,K].
        x = x + dt_bias.reshape(g.shape[-2], g.shape[-1]).float()

    A = A_log.float().exp().reshape(1, 1, -1, 1)
    if safe_gate:
        if lower_bound is None or not (-5.0 <= lower_bound < 0.0):
            raise ValueError("safe gate requires lower_bound in [-5, 0)")
        return float(lower_bound) * torch.sigmoid(A * x)
    return -A * F.softplus(x)


def _run_one_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    erase: torch.Tensor,
    write: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    return_intermediate_states: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Exact recurrent form corresponding to the Hopper chunk/WY algorithm.

    Shapes: q/k/g/erase=[T,H,K], v/write=[T,H,V], state=[H,K,V].
    All recurrence arithmetic is float32 to provide a stable oracle.
    """
    outputs = []
    states = []
    for t in range(q.shape[0]):
        qt = q[t].float()
        kt = k[t].float()
        vt = v[t].float()
        bt = erase[t].float()
        wt = write[t].float()

        # Each K row of the state has its own decay.
        state = state * torch.exp(log_decay[t]).unsqueeze(-1)

        # GDN2 erase is vector-valued on K; write is vector-valued on V.
        correction = wt * vt - torch.einsum("hk,hkv->hv", bt * kt, state)
        state = state + kt.unsqueeze(-1) * correction.unsqueeze(-2)

        # Causal output includes the current token's state update. This is the
        # recurrent equivalent of Aqk's lower triangle including its diagonal.
        out = float(scale) * torch.einsum("hk,hkv->hv", qt, state)
        outputs.append(out)
        if return_intermediate_states:
            states.append(state.clone())

    o = torch.stack(outputs, dim=0) if outputs else v.float().clone()
    hs = torch.stack(states, dim=0) if states else None
    return o, state, hs


@torch.no_grad()
def chunk_gdn2_hygon(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    chunk_size: int = 64,
    return_intermediate_states: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
):
    """Hygon-compatible GDN2 forward oracle.

    The public signature intentionally follows the Hopper implementation.
    ``chunk_size`` and ``chunk_indices`` do not affect the recurrent result.
    Variable-length input follows the packed convention B=1 with cu_seqlens.
    """
    del cu_seqlens_cpu, chunk_indices
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/g/b and v/w must be rank-4 BTHK/BTHV tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g and b must have identical [B,T,H,K] shapes")
    if v.shape != w.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("v/w must be [B,T,H,V] and share B,T,H with q")
    if q.shape[-1] > 256:
        raise ValueError("the Hopper-compatible contract requires K <= 256")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be float32")

    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else float(scale)
    if use_qk_l2norm_in_kernel:
        q_work = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k_work = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    else:
        q_work, k_work = q, k
    decay = _effective_log_decay(
        g,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    if cu_seqlens is None:
        ranges = [(i, 0, T) for i in range(B)]
    else:
        if B != 1:
            raise ValueError("packed varlen mode requires B=1")
        offsets = cu_seqlens.detach().cpu().tolist()
        ranges = [(0, int(s), int(e)) for s, e in zip(offsets, offsets[1:])]

    out = torch.empty_like(v)
    finals = []
    all_states = []
    for n, (batch, start, end) in enumerate(ranges):
        if initial_state is None:
            state = torch.zeros((H, K, V), device=q.device, dtype=torch.float32)
        else:
            supplied = initial_state[n if cu_seqlens is not None else batch]
            state = supplied.transpose(-1, -2) if state_v_first else supplied
            state = state.float().clone()
        seq_o, state, seq_h = _run_one_sequence(
            q_work[batch, start:end], k_work[batch, start:end],
            v[batch, start:end], decay[batch, start:end],
            b[batch, start:end], w[batch, start:end], state, scale,
            return_intermediate_states,
        )
        out[batch, start:end] = seq_o.to(v.dtype)
        finals.append(state.transpose(-1, -2) if state_v_first else state)
        if seq_h is not None:
            all_states.append(seq_h.transpose(-1, -2) if state_v_first else seq_h)

    final_state = torch.stack(finals) if output_final_state else None
    if return_intermediate_states:
        # Unlike the Hopper kernel's chunk-boundary tensor, the oracle returns
        # every token state. Varlen sequences are returned as a Python list.
        h = all_states[0] if len(all_states) == 1 else all_states
        return out, final_state, h
    return out, final_state


# Drop-in name for a backend dispatch module.
chunk_gdn2 = chunk_gdn2_hygon
