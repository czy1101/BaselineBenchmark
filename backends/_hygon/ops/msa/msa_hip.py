"""MiniMax M3 MSA BF16 inference backend for Hygon DTK/HIP."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import load


SPARSE_BLOCK_SIZE = 128
HEAD_DIM = 128
_EXT = None
_TRITON_BACKEND = None
_TRITON_CHECKED = False


def load_extension(verbose: bool = False):
    global _EXT
    if _EXT is None:
        _EXT = load(
            name="msa_hygon_hip_ext_v2_score_reduce",
            sources=[str(Path(__file__).with_name("msa_hip_kernel.cu"))],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-ffast-math"],
            extra_ldflags=["-lhipblas"],
            verbose=verbose,
        )
    return _EXT


def _load_triton_backend():
    """Load the bundled FlagTree HCU Triton backend when it is available."""
    global _TRITON_BACKEND, _TRITON_CHECKED
    if os.getenv("MSA_HYGON_USE_TRITON", "1") == "0":
        return None
    if _TRITON_CHECKED:
        return _TRITON_BACKEND
    _TRITON_CHECKED = True
    try:
        _TRITON_BACKEND = importlib.import_module("msa_triton")
        return _TRITON_BACKEND
    except ModuleNotFoundError as first_error:
        # FlagTree is bundled in the BW1000 image but is not necessarily on
        # PYTHONPATH.  Prefer its built Python package, then its source tree.
        root = Path(os.getenv("TRITON_ROOT", "/workspace/FlagTree"))
        candidates = sorted(root.glob("build/lib.linux-*-cpython-310"))
        candidates.append(root / "python")
        for candidate in reversed(candidates):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            _TRITON_BACKEND = importlib.import_module("msa_triton")
            return _TRITON_BACKEND
        except ModuleNotFoundError:
            # A machine without FlagTree keeps the verified HIP/hipBLAS path.
            if first_error.name not in {"triton", "msa_triton"}:
                raise
            return None


def sparse_backend_name() -> str:
    return (
        "triton_fused_online+hip_decode_b1"
        if _load_triton_backend() is not None
        else "hipblas_pack"
    )


def _bf16_contiguous(name: str, tensor: torch.Tensor, rank: int) -> None:
    if not tensor.is_cuda or tensor.dtype != torch.bfloat16 or tensor.ndim != rank:
        raise TypeError(f"{name} must be a rank-{rank} Hygon/HIP BF16 tensor")
    if not tensor.is_contiguous():
        raise NotImplementedError(f"{name} must be contiguous")
    if tensor.requires_grad:
        raise NotImplementedError("MSA Hygon v1 is inference-only")


def _int32_contiguous(name: str, tensor: torch.Tensor, rank: int) -> None:
    if not tensor.is_cuda or tensor.dtype != torch.int32 or tensor.ndim != rank:
        raise TypeError(f"{name} must be a rank-{rank} Hygon/HIP int32 tensor")
    if not tensor.is_contiguous():
        raise NotImplementedError(f"{name} must be contiguous")


def _int32_strided(name: str, tensor: torch.Tensor, rank: int) -> None:
    if not tensor.is_cuda or tensor.dtype != torch.int32 or tensor.ndim != rank:
        raise TypeError(f"{name} must be a rank-{rank} Hygon/HIP int32 tensor")


@torch.no_grad()
def minimax_m3_index_score(
    idx_q,
    index_kv_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
):
    _bf16_contiguous("idx_q", idx_q, 3)
    _bf16_contiguous("index_kv_cache", index_kv_cache, 3)
    _int32_contiguous("block_table", block_table, 2)
    _int32_contiguous("cu_seqlens_q", cu_seqlens_q, 1)
    _int32_contiguous("seq_lens", seq_lens, 1)
    _int32_contiguous("prefix_lens", prefix_lens, 1)
    if idx_q.shape[-1] != HEAD_DIM or index_kv_cache.shape[-2:] != (
        SPARSE_BLOCK_SIZE,
        HEAD_DIM,
    ):
        raise NotImplementedError("v1 requires block=128 and head_dim=128")
    if idx_q.shape[1] != num_kv_heads:
        raise ValueError("index query heads must equal num_kv_heads")
    return load_extension(False).index_score_prefill(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        int(max_query_len),
        int(max_seq_len),
        int(num_kv_heads),
    )


@torch.no_grad()
def minimax_m3_index_topk(
    score,
    cu_seqlens_q,
    prefix_lens,
    max_query_len,
    topk,
    init_blocks,
    local_blocks,
    out=None,
):
    if not score.is_cuda or score.dtype != torch.float32 or score.ndim != 3:
        raise TypeError("score must be rank-3 Hygon/HIP FP32")
    if not score.is_contiguous():
        raise NotImplementedError("score must be contiguous")
    _int32_contiguous("cu_seqlens_q", cu_seqlens_q, 1)
    _int32_contiguous("prefix_lens", prefix_lens, 1)
    if topk < 1 or topk > 16:
        raise NotImplementedError("v1 supports topk in [1,16]")
    if out is not None:
        _int32_strided("out", out, 3)
    return load_extension(False).index_topk_prefill(
        score,
        cu_seqlens_q,
        prefix_lens,
        int(max_query_len),
        int(topk),
        int(init_blocks),
        int(local_blocks),
        out,
    )


@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    score_out=None,
):
    _bf16_contiguous("idx_q", idx_q, 3)
    _bf16_contiguous("index_kv_cache", index_kv_cache, 3)
    _int32_contiguous("block_table", block_table, 2)
    _int32_contiguous("seq_lens", seq_lens, 1)
    if decode_query_len > max_decode_query_len:
        raise ValueError("decode_query_len exceeds max_decode_query_len")
    if idx_q.shape[0] != seq_lens.numel() * decode_query_len:
        raise ValueError("decode total_q mismatch")
    if score_out is not None:
        if (
            not score_out.is_cuda
            or score_out.dtype != torch.float32
            or score_out.ndim != 3
        ):
            raise TypeError("score_out must be a rank-3 Hygon/HIP FP32 tensor")
    return load_extension(False).index_decode_score(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        int(max_seq_len),
        int(init_blocks),
        int(local_blocks),
        int(num_kv_heads),
        int(decode_query_len),
        score_out,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len,
    out=None,
    score_out=None,
):
    _bf16_contiguous("idx_q", idx_q, 3)
    _bf16_contiguous("index_kv_cache", index_kv_cache, 3)
    _int32_contiguous("block_table", block_table, 2)
    _int32_contiguous("seq_lens", seq_lens, 1)
    if topk < 1 or topk > 16:
        raise NotImplementedError("v1 supports topk in [1,16]")
    if decode_query_len > max_decode_query_len:
        raise ValueError("decode_query_len exceeds max_decode_query_len")
    if out is not None:
        _int32_strided("out", out, 3)
    if score_out is not None:
        if (
            not score_out.is_cuda
            or score_out.dtype != torch.float32
            or score_out.ndim != 3
        ):
            raise TypeError("score_out must be a rank-3 Hygon/HIP FP32 tensor")
    return load_extension(False).index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        int(max_seq_len),
        int(topk),
        int(init_blocks),
        int(local_blocks),
        int(num_kv_heads),
        int(decode_query_len),
        out,
        score_out,
    )


def _check_sparse(q, kv_cache, topk_idx, block_table, output, num_kv_heads):
    _bf16_contiguous("q", q, 3)
    _bf16_contiguous("kv_cache", kv_cache, 4)
    _int32_strided("topk_idx", topk_idx, 3)
    _int32_contiguous("block_table", block_table, 2)
    _bf16_contiguous("output", output, 3)
    if q.shape[-1] != HEAD_DIM or kv_cache.shape[-2:] != (
        SPARSE_BLOCK_SIZE,
        2 * HEAD_DIM,
    ):
        raise NotImplementedError("v1 requires block=128 and head_dim=128")
    if q.shape[1] % num_kv_heads:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if topk_idx.shape[0] != num_kv_heads or topk_idx.shape[-1] > 16:
        raise NotImplementedError("topk layout/count is unsupported")
    if topk_idx.shape[1] < q.shape[0]:
        raise ValueError("topk_idx has fewer query rows than q")
    if output.shape != q.shape:
        raise ValueError("output shape must equal q shape")


@torch.no_grad()
def minimax_m3_sparse_attn(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
):
    _check_sparse(q, kv_cache, topk_idx, block_table, output, num_kv_heads)
    _int32_contiguous("cu_seqlens_q", cu_seqlens_q, 1)
    _int32_contiguous("seq_lens", seq_lens, 1)
    _int32_contiguous("prefix_lens", prefix_lens, 1)
    if k_scale is not None or v_scale is not None:
        raise NotImplementedError("FP8 KV scaling is outside the BF16 v1 path")
    triton_backend = _load_triton_backend()
    if triton_backend is not None:
        triton_backend.sparse_prefill(
            q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens,
            prefix_lens, int(max_query_len), int(num_kv_heads), float(sm_scale),
            output,
        )
        return
    load_extension(False).sparse_prefill(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        int(max_query_len),
        int(num_kv_heads),
        float(sm_scale),
        output,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
    k_scale=None,
    v_scale=None,
):
    _check_sparse(q, kv_cache, topk_idx, block_table, output, num_kv_heads)
    _int32_contiguous("seq_lens", seq_lens, 1)
    if k_scale is not None or v_scale is not None:
        raise NotImplementedError("FP8 KV scaling is outside the BF16 v1 path")
    triton_backend = _load_triton_backend()
    # The fused Triton kernel wins once there are enough independent requests,
    # while a single-request decode is launch/occupancy limited.  The verified
    # HIP kernel is ~28% faster for B=1 on BW1000 (0.225 ms vs 0.310 ms).
    # Keep an override for controlled A/B and future compiler revisions.
    triton_decode_b1 = os.getenv("MSA_HYGON_TRITON_DECODE_B1", "0") == "1"
    use_triton_decode = (
        triton_backend is not None
        and (seq_lens.numel() > 1 or triton_decode_b1)
    )
    if use_triton_decode:
        triton_backend.sparse_decode(
            q, kv_cache, topk_idx, block_table, seq_lens, int(num_kv_heads),
            float(sm_scale), output, int(decode_query_len),
        )
        return
    load_extension(False).sparse_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        int(num_kv_heads),
        float(sm_scale),
        output,
        int(decode_query_len),
    )


__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_decode",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "load_extension",
    "sparse_backend_name",
]
