# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Correctness checks for the MUSA C chunk GLA baseline."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from BaselineBenchmark.backends._mthreads.ops.chunk_gla_musa_c import (
    is_available as musa_c_available,
    musa_chunk_gla,
    torch_recurrent_chunk_gla,
)

try:
    from flag_attn.runtime.backend._mthreads.gated_linear_attention.chunk_gla import (
        ChunkGLAFunction,
    )
except ImportError:
    ChunkGLAFunction = None


FORWARD_CASES = (
    ("small", 1, 16, 2, 32),
    ("non_power_of_two", 1, 73, 3, 96),
    ("wide_128", 1, 128, 2, 128),
    ("wide_256", 1, 64, 2, 256),
)

BACKWARD_CASES = (
    ("small", 1, 8, 2, 4),
    ("wide_128", 1, 16, 1, 128),
)


def _musa_triton_chunk_gla(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Call the production Triton/TLE path without routing through a wrapper."""

    if ChunkGLAFunction is None:
        raise RuntimeError("the MUSA Triton/TLE chunk GLA path is unavailable")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ChunkGLAFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    )


def _require_musa_baselines():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("this baseline requires an available MUSA device")
    if ChunkGLAFunction is None:
        pytest.skip("the production MUSA Triton/TLE implementation is unavailable")
    if not musa_c_available():
        pytest.skip(
            "the MUSA C baseline extension is not built; run its "
            "build_musa_chunk_gla.py first"
        )


@pytest.mark.parametrize("case_name,B,T,H,D", FORWARD_CASES)
def test_chunk_gla_musa_c_forward(case_name, B, T, H, D):
    _require_musa_baselines()
    dtype = torch.bfloat16
    device = "musa"
    torch.manual_seed(0)

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn_like(q))
    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": True,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }

    with torch.no_grad():
        ref_out, ref_state = torch_recurrent_chunk_gla(q, k, v, g, **kwargs)
        triton_out, triton_state = _musa_triton_chunk_gla(q, k, v, g, **kwargs)
        musa_c_out, musa_c_state = musa_chunk_gla(q, k, v, g, **kwargs)

    torch.testing.assert_close(triton_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(triton_state, ref_state, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(musa_c_out, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(musa_c_state, ref_state, atol=5e-2, rtol=5e-2)


def _run_backward(function, base, kwargs):
    q, k, v, g_logit = [tensor.detach().clone().requires_grad_() for tensor in base]
    g = F.logsigmoid(g_logit)
    out, state = function(q, k, v, g, **kwargs)
    (out.float().square().sum() + state.square().sum()).backward()
    return (
        out.detach(),
        state.detach(),
        (q.grad, k.grad, v.grad, g_logit.grad),
    )


def _assert_gradients_close(label, actual, expected):
    for grad_name, grad, ref_grad in zip(("q", "k", "v", "g_logit"), actual, expected):
        try:
            torch.testing.assert_close(grad, ref_grad, atol=8e-2, rtol=8e-2)
        except AssertionError:
            diff = (grad.float() - ref_grad.float()).abs()
            pytest.fail(
                f"{label} gradient mismatch: {grad_name}; "
                f"max_abs={diff.max().item():.6g}; "
                f"max_ref={ref_grad.float().abs().max().item():.6g}"
            )


@pytest.mark.parametrize("case_name,B,T,H,D", BACKWARD_CASES)
def test_chunk_gla_musa_c_backward(case_name, B, T, H, D):
    _require_musa_baselines()
    dtype = torch.bfloat16
    device = "musa"
    torch.manual_seed(1)
    base = tuple(torch.randn(B, T, H, D, device=device, dtype=dtype) for _ in range(4))
    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": True,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }

    ref_out, ref_state, ref_grads = _run_backward(
        torch_recurrent_chunk_gla, base, kwargs
    )
    for label, function in (
        ("Triton/TLE", _musa_triton_chunk_gla),
        ("MUSA C", musa_chunk_gla),
    ):
        out, state, grads = _run_backward(function, base, kwargs)
        torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state, ref_state, atol=2e-2, rtol=2e-2)
        _assert_gradients_close(label, grads, ref_grads)
