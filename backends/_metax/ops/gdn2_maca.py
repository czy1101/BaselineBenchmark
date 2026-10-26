# This file is modified and supported by the Moonshot AI Team

"""Python interface for the native ``gdn2_maca.cu`` extension."""

from __future__ import annotations

import math

import torch

try:
    import _gdn2_maca_cuda
except ImportError as exc:  # pragma: no cover - exercised before local build
    raise ImportError(
        "_gdn2_maca_cuda is not built. Run: python setup.py build_ext --inplace"
    ) from exc


@torch.compiler.disable
def gdn2_maca(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    transpose_state_layout: bool = False,
    state_v_first: bool | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    implementation: str = "auto",
    **_ignored,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run native MACA GDN2.

    ``chunk64_bmm_v5`` selects V5's fused three-BMM matrix-core path.
    ``chunk64_bmm`` retains V4, ``chunk64`` retains V3, and ``persistent`` /
    ``global`` retain V2/V1 recurrent correctness fallbacks. ``auto`` selects
    V5 for its supported fixed-length path and otherwise selects persistent.
    """
    del cu_seqlens_cpu

    if implementation not in {
        "auto",
        "chunk64_bmm_v5",
        "chunk64_bmm",
        "chunk64",
        "persistent",
        "global",
    }:
        raise ValueError(
            "implementation must be 'auto', 'chunk64_bmm_v5', "
            "'chunk64_bmm', 'chunk64', "
            "'persistent' or 'global'"
        )

    if state_v_first is not None:
        transpose_state_layout = state_v_first
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q and v must be rank-4 tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g and b must have identical [B,T,H,K] shapes")
    if v.shape != w.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("v/w must match each other and q on B/T/H")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("native MACA GDN2 supports float16 and bfloat16")
    if any(t.dtype != q.dtype for t in (k, v, g, b, w)):
        raise TypeError("q/k/v/g/b/w must have the same dtype")
    if not q.is_cuda:
        raise ValueError("inputs must be on the MACA/CUDA device")

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if use_gate_in_kernel and A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None or not (-5.0 <= lower_bound < 0.0):
            raise ValueError("safe_gate requires lower_bound in [-5,0)")

    chunk64_supported = (
        chunk_size == 64
        and cu_seqlens is None
        and not transpose_state_layout
        and not use_qk_l2norm_in_kernel
        and not use_gate_in_kernel
    )
    selected = implementation
    if selected == "auto":
        selected = "chunk64_bmm_v5" if chunk64_supported else "persistent"
    if selected in {"chunk64_bmm_v5", "chunk64_bmm", "chunk64"} and not chunk64_supported:
        raise ValueError(
            "chunk64 paths currently require chunk_size=64, fixed-length input, "
            "state_v_first=False, use_qk_l2norm_in_kernel=False, and "
            "use_gate_in_kernel=False"
        )

    device = q.device

    def empty(dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(0, device=device, dtype=dtype)

    h0 = empty(torch.float32) if initial_state is None else initial_state.contiguous()
    cu = empty(torch.long) if cu_seqlens is None else cu_seqlens.to(
        device=device, dtype=torch.long
    ).contiguous()
    A = empty(torch.float32) if A_log is None else A_log.to(
        device=device, dtype=torch.float32
    ).contiguous()
    bias = empty(torch.float32) if dt_bias is None else dt_bias.to(
        device=device, dtype=torch.float32
    ).contiguous()

    contiguous_inputs = tuple(
        tensor.contiguous() for tensor in (q, k, v, g, b, w)
    )
    if selected == "chunk64_bmm_v5":
        out, final_state = _gdn2_maca_cuda.forward_chunk64_bmm_v5(
            *contiguous_inputs,
            h0,
            float(scale),
        )
    elif selected == "chunk64_bmm":
        out, final_state = _gdn2_maca_cuda.forward_chunk64_bmm(
            *contiguous_inputs,
            h0,
            float(scale),
        )
    elif selected == "chunk64":
        out, final_state = _gdn2_maca_cuda.forward_chunk64(
            *contiguous_inputs,
            h0,
            float(scale),
        )
    else:
        out, final_state = _gdn2_maca_cuda.forward(
            *contiguous_inputs,
            h0,
            cu,
            A,
            bias,
            float(scale),
            bool(output_final_state),
            bool(transpose_state_layout),
            bool(use_qk_l2norm_in_kernel),
            bool(use_gate_in_kernel),
            bool(safe_gate),
            float(lower_bound or 0.0),
            lower_bound is not None,
            selected == "persistent",
        )
    return out, final_state if output_final_state else None


chunk_gdn2 = gdn2_maca

__all__ = ["gdn2_maca", "chunk_gdn2"]
