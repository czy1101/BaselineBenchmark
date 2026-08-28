# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, explicit PyTorch recurrence used as the correctness reference."""

from __future__ import annotations

import torch


def torch_recurrent_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
):
    """Evaluate equal-length GLA with an FP32 recurrent state.

    This function is deliberately simple and slow.  It belongs to the
    baseline correctness harness and must not be imported by production code.
    """

    del cu_seqlens_cpu
    if state_v_first:
        raise NotImplementedError("the torch reference does not support state_v_first")
    if cu_seqlens is not None:
        raise NotImplementedError("the torch reference does not support cu_seqlens")
    if q.ndim != 4 or q.shape != k.shape or q.shape != g.shape:
        raise ValueError("q, k and g must have shape [B, T, H, K]")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B, T, H, V]")

    dtype = q.dtype
    qf, kf, vf, gf = (x.float() for x in (q, k, v, g))
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if scale is None:
        scale = key_dim**-0.5

    if initial_state is None:
        state = torch.zeros(
            (batch, heads, key_dim, value_dim),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        expected = (batch, heads, key_dim, value_dim)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        state = initial_state.float()

    out = torch.empty(
        (batch, sequence, heads, value_dim),
        device=q.device,
        dtype=torch.float32,
    )
    for timestep in range(sequence):
        qt = qf[:, timestep] * float(scale)
        kt = kf[:, timestep]
        vt = vf[:, timestep]
        forget = gf[:, timestep].exp()
        state = state * forget[..., None] + kt[..., None] * vt[..., None, :]
        out[:, timestep] = (qt[..., None] * state).sum(-2)

    final_state = state if output_final_state else None
    return out.to(dtype), final_state


__all__ = ["torch_recurrent_chunk_gla"]
