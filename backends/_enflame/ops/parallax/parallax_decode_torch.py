"""Pure Torch-GCU Parallax decode baseline for Enflame GCU.

This module uses PyTorch operators dispatched through torch_gcu/TopsAten. It
does not import the TLE, Triton, or TopsCC implementations.
"""

from __future__ import annotations

import math

import torch

try:
    import torch_gcu as _torch_gcu  # noqa: F401
except ImportError:
    _torch_gcu = None


def torch_gcu_parallax_decode_available() -> bool:
    return _torch_gcu is not None and hasattr(torch, "gcu") and torch.gcu.is_available()


def _window_size_to_left(window_size: int | tuple[int, int] | None) -> int:
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        left = window_size
    else:
        try:
            left, right = window_size
        except (TypeError, ValueError) as exc:
            raise ValueError("window_size must be None, an int, or a (left, right) pair") from exc
        if right is not None and right > 0:
            raise ValueError("single-token decode requires window_size right <= 0")
    if not isinstance(left, int) or left < -1:
        raise ValueError("window_size left must be -1 or non-negative")
    return left


def _validate_inputs(q, r, k, v, seqused_k, out) -> tuple[int, int, int, int, int]:
    for name, tensor in (("q", q), ("r", r), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if q.ndim != 4 or r.shape != q.shape or q.shape[1] != 1:
        raise ValueError("q and r must have the same (B, 1, H_q, D) shape")
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("k and v must have the same (B, L, H_kv, D) shape")
    batch, _, query_heads, dim = q.shape
    kv_batch, length, kv_heads, kv_dim = k.shape
    if batch < 1 or length < 1 or query_heads < 1 or kv_heads < 1:
        raise ValueError("B, L, H_q and H_kv must be positive")
    if kv_batch != batch or kv_dim != dim:
        raise ValueError("q/r and k/v batch size and head dimension must match")
    if query_heads % kv_heads:
        raise ValueError("GQA requires H_q to be divisible by H_kv")
    if dim not in (64, 128):
        raise ValueError("Parallax decode supports head_dim in {64, 128}")
    if any(t.device.type != "gcu" for t in (q, r, k, v)):
        raise ValueError("q, r, k and v must be GCU tensors")
    if any(t.device != q.device for t in (r, k, v)):
        raise ValueError("q, r, k and v must be on the same GCU device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Parallax decode requires float16 or bfloat16")
    if any(t.dtype != q.dtype for t in (r, k, v)):
        raise TypeError("q, r, k and v must have the same dtype")
    if seqused_k is not None:
        if not isinstance(seqused_k, torch.Tensor):
            raise TypeError("seqused_k must be a torch.Tensor")
        if seqused_k.shape != (batch,) or seqused_k.dtype != torch.int32:
            raise ValueError(f"seqused_k must be int32 with shape ({batch},)")
        if seqused_k.device != q.device:
            raise ValueError("seqused_k must be on the same GCU device as q")
        if int(seqused_k.min().item()) < 1 or int(seqused_k.max().item()) > length:
            raise ValueError(f"seqused_k values must be in [1, {length}]")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q's shape, dtype and device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    return batch, length, query_heads, kv_heads, dim


def _decode_dense_torch(q, r, k, v, scale, seqused_k, window_left):
    """Compute decode in FP32, keeping GQA K/V shared across query heads."""
    batch, _, query_heads, dim = q.shape
    length, kv_heads = k.shape[1:3]
    groups = query_heads // kv_heads

    query = q.reshape(batch, kv_heads, groups, 1, dim).float()
    secondary_query = r.reshape(batch, kv_heads, groups, 1, dim).float()
    key = k.float().permute(0, 2, 1, 3).unsqueeze(2)
    value = v.float().permute(0, 2, 1, 3).unsqueeze(2)
    key_t = key.transpose(-1, -2)
    primary_scores = torch.matmul(query, key_t) * scale
    secondary_scores = torch.matmul(secondary_query, key_t)

    if seqused_k is not None or window_left >= 0:
        active_lengths = (
            seqused_k if seqused_k is not None
            else torch.full((batch,), length, dtype=torch.int32, device=q.device)
        )
        positions = torch.arange(length, dtype=torch.int32, device=q.device)
        valid = positions[None, :] < active_lengths[:, None]
        if window_left >= 0:
            valid = valid & (
                positions[None, :] >= (active_lengths[:, None] - window_left)
            )
        primary_scores = primary_scores.masked_fill(
            ~valid[:, None, None, None, :], -float("inf")
        )

    pivot = primary_scores.amax(dim=-1, keepdim=True)
    safe_pivot = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    weights = torch.exp(primary_scores - safe_pivot)
    weighted_secondary = weights * secondary_scores
    denominator = weights.sum(dim=-1, keepdim=True)
    secondary_sum = weighted_secondary.sum(dim=-1, keepdim=True)
    mean_value = torch.matmul(weights, value)
    secondary_value = torch.matmul(weighted_secondary, value)
    inverse = torch.where(
        denominator > 0, denominator.reciprocal(), torch.zeros_like(denominator)
    )
    result = mean_value * inverse * (1.0 + secondary_sum * inverse)
    result = result - secondary_value * inverse
    return result.reshape(batch, 1, query_heads, dim).to(q.dtype).contiguous()


def parallax_attn_with_kvcache_torch_gcu(
    q: torch.Tensor,
    r: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_size: int | tuple[int, int] | None = None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch-GCU equivalent of Parallax decode's dense-KV serving interface."""
    if page_table is not None:
        raise NotImplementedError("paged KV (page_table) is not implemented")
    _, _, _, _, dim = _validate_inputs(q, r, k_cache, v_cache, seqused_k, out)
    window_left = _window_size_to_left(window_size)
    actual_scale = dim ** -0.5 if scale is None else float(scale)
    result = _decode_dense_torch(
        q, r, k_cache, v_cache, actual_scale, seqused_k, window_left
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def parallax_decode_torch_gcu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | None = None,
    *,
    window_size_left: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch-GCU equivalent of ``parallax_decode`` for one query token."""
    return parallax_attn_with_kvcache_torch_gcu(
        q, r, k, v, window_size=window_size_left, scale=qk_scale, out=out
    )


parallax_decode_torch = parallax_decode_torch_gcu
parallax_attn_with_kvcache_torch = parallax_attn_with_kvcache_torch_gcu

__all__ = [
    "parallax_attn_with_kvcache_torch",
    "parallax_attn_with_kvcache_torch_gcu",
    "parallax_decode_torch",
    "parallax_decode_torch_gcu",
    "torch_gcu_parallax_decode_available",
]
