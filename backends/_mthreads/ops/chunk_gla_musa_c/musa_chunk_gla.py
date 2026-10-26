# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Standalone MUSA C baseline implementation of recurrent/chunked GLA.

Build the extension from the repository root with
``python BaselineBenchmark/backends/_mthreads/ops/chunk_gla_musa_c/``
``build_musa_chunk_gla.py build_ext --inplace`` in a MUSA environment.
This package is a benchmark baseline and is independent of the production
Triton/TLE implementation under ``src/flaggems_vllm``.
"""

from __future__ import annotations

import torch

try:
    from . import _musa_chunk_gla
except (ImportError, OSError):
    _musa_chunk_gla = None


def is_available() -> bool:
    return _musa_chunk_gla is not None and hasattr(_musa_chunk_gla, "forward")


def _check_inputs(q, k, v, g, initial_state, state_v_first, cu_seqlens):
    if not is_available():
        raise RuntimeError(
            "The native MUSA GLA extension is not built. Run "
            "python BaselineBenchmark/backends/_mthreads/ops/"
            "chunk_gla_musa_c/build_musa_chunk_gla.py build_ext --inplace "
            "in the MUSA container."
        )
    if q.device.type != "musa":
        raise ValueError("native MUSA GLA requires MUSA tensors")
    if state_v_first:
        raise NotImplementedError("state_v_first is not implemented by native MUSA GLA")
    if cu_seqlens is not None:
        raise NotImplementedError(
            "varlen cu_seqlens is not implemented by native MUSA GLA"
        )
    if q.ndim != 4 or k.shape != q.shape or g.shape != q.shape:
        raise ValueError("q, k and g must have shape [B, T, H, K]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must have shape [B, T, H, V]")
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
    ):
        raise ValueError("native MUSA GLA currently requires contiguous inputs")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native MUSA GLA supports float16, bfloat16 and float32")
    if initial_state is not None:
        B, _, H, K = q.shape
        V = v.shape[-1]
        if initial_state.shape != (B, H, K, V):
            raise ValueError("initial_state must have shape [B, H, K, V]")
        if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous float32")


class _NativeMUSAGLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        initial_state_arg = (
            initial_state
            if initial_state is not None
            else torch.empty(0, device=q.device, dtype=torch.float32)
        )
        out, final_state, checkpoints, chunk_decay = _musa_chunk_gla.forward(
            q, k, v, g, float(scale), initial_state_arg, bool(output_final_state)
        )
        ctx.save_for_backward(q, k, v, g, initial_state_arg, checkpoints, chunk_decay)
        ctx.scale = float(scale)
        ctx.has_initial_state = initial_state is not None
        ctx.output_final_state = bool(output_final_state)
        return out, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, checkpoints, chunk_decay = ctx.saved_tensors
        # PyTorch may provide non-contiguous upstream gradients for reductions
        # such as ``output.sum()``.  The native MUSA kernel uses flat pointer
        # arithmetic and therefore requires contiguous gradient tensors.
        do = do.contiguous()
        if dht is None:
            dht = torch.empty(0, device=q.device, dtype=torch.float32)
        else:
            dht = dht.contiguous()
        dq, dk, dv, dg, dh0 = _musa_chunk_gla.backward(
            q,
            k,
            v,
            g,
            do,
            dht,
            initial_state,
            checkpoints,
            chunk_decay,
            ctx.scale,
        )
        if not ctx.has_initial_state:
            dh0 = None
        return dq, dk, dv, dg, None, dh0, None


def musa_chunk_gla(
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
    """Run the optional native MUSA GLA path with autograd support."""

    del cu_seqlens_cpu
    _check_inputs(q, k, v, g, initial_state, state_v_first, cu_seqlens)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _NativeMUSAGLAFunction.apply(
        q, k, v, g, float(scale), initial_state, bool(output_final_state)
    )


__all__ = ["is_available", "musa_chunk_gla"]
