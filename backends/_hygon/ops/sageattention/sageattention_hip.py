"""Public SageAttention interface for Hygon BW1000."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import torch


_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    root = Path(os.getenv("TRITON_ROOT", "/workspace/FlagTree"))
    candidates = sorted(root.glob("build/lib.linux-*-cpython-310"))
    candidates.append(root / "python")
    for candidate in reversed(candidates):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    _BACKEND = importlib.import_module("sageattention_triton")
    return _BACKEND


def _check_float(name, x):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"{name} must be a Hygon FP16/BF16 tensor")
    if x.ndim != 4 or not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous rank 4")


@torch.no_grad()
def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None,
                   tensor_layout="HND"):
    _check_float("q", q)
    _check_float("k", k)
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("BW1000 v1 supports head_dim 64 or 128")
    if BLKQ != 128 or BLKK != 64:
        raise NotImplementedError("BW1000 v1 uses BLKQ=128 and BLKK=64")
    return _backend().per_block_int8(
        q, k, km=km, BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale,
        tensor_layout=tensor_layout,
    )


@torch.no_grad()
def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
            output_dtype=torch.float16, return_lse=False, maxnreg=None):
    if not q.is_cuda or q.dtype != torch.int8 or not q.is_contiguous():
        raise TypeError("q must be contiguous Hygon INT8")
    if not k.is_cuda or k.dtype != torch.int8 or not k.is_contiguous():
        raise TypeError("k must be contiguous Hygon INT8")
    _check_float("v", v)
    if v.dtype != torch.float16:
        raise NotImplementedError("BW1000 v1 uses FP16 values")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("q_scale and k_scale must be FP32")
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("output_dtype must be FP16 or BF16")
    if maxnreg is not None and maxnreg <= 0:
        raise ValueError("maxnreg must be positive")
    return _backend().forward(
        q, k, v, q_scale, k_scale, tensor_layout=tensor_layout,
        attn_mask=attn_mask, output_dtype=output_dtype,
        return_lse=return_lse, maxnreg=maxnreg,
    )


def backend_name():
    _backend()
    return "flagtree_triton_hip_int8_qk_fp16_pv"


__all__ = ["forward", "per_block_int8", "backend_name"]
