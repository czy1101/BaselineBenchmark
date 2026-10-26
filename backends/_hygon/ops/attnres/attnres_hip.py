from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "attnes_hip_kernel.cu"
BUILD_DIR = Path("/tmp/flaggems_hygon_attnes")
LIBRARY = BUILD_DIR / "attnes_hip_kernel.so"

_LIBRARY = None


def _ptr(tensor):
    return ctypes.c_void_p(0 if tensor is None else tensor.data_ptr())


def _load_library():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        hipcc = os.environ.get("HIPCC", "/opt/dtk/bin/hipcc")
        subprocess.run(
            [
                hipcc,
                "-O3",
                "-ffast-math",
                "-std=c++17",
                "-fPIC",
                "-shared",
                str(SOURCE),
                "-o",
                str(LIBRARY),
            ],
            check=True,
        )

    library = ctypes.CDLL(str(LIBRARY))
    library.attnes_forward.restype = None
    library.attnes_forward.argtypes = (
        [ctypes.c_void_p] * 13
        + [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
    )
    return library


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
) -> torch.Tensor:
    global _LIBRARY

    if not residuals or len(residuals) > 9:
        raise ValueError("AttnRes HIP Baseline supports 1-9 sources")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("only float16 and bfloat16 are supported")

    if query.ndim == 2 and query.shape[0] == 1:
        query = query.reshape(-1)
    elif query.ndim != 1:
        raise ValueError("query must be a shared [hidden] vector")

    hidden = query.numel()
    first = residuals[0].reshape(-1, residuals[0].shape[-1])
    rows = first.shape[0]

    query = query.contiguous()
    rms_weight = rms_weight.reshape(-1).contiguous()

    if hidden != first.shape[1]:
        raise ValueError("query and residual hidden sizes must match")

    packed = [
        tensor.reshape(rows, hidden).contiguous()
        for tensor in residuals
    ]
    packed += [packed[0]] * (9 - len(packed))

    if output_rms_weight is not None:
        output_rms_weight = output_rms_weight.reshape(-1).contiguous()

    output = torch.empty((rows, hidden), device=query.device, dtype=query.dtype)

    if _LIBRARY is None:
        _LIBRARY = _load_library()

    _LIBRARY.attnes_forward(
        _ptr(query),
        *[_ptr(tensor) for tensor in packed],
        _ptr(rms_weight),
        _ptr(output_rms_weight),
        _ptr(output),
        ctypes.c_int64(rows),
        ctypes.c_int(hidden),
        ctypes.c_int(len(residuals)),
        ctypes.c_float(rms_eps),
        ctypes.c_float(scale),
        ctypes.c_int(0 if query.dtype == torch.float16 else 1),
        ctypes.c_int(1 if output_rms_weight is not None else 0),
    )

    return output


attnres = fused_attnres

__all__ = ["attnres", "fused_attnres"]
