"""Storage helpers shared by the native TOPSCC FP8 implementations."""

from __future__ import annotations

import torch


def raw_fp8_bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.int8:
        return tensor.view(torch.uint8)
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None and tensor.dtype == fp8_dtype:
        return tensor.view(torch.uint8)
    raise TypeError("FP8 tensors must use float8_e4m3fn, uint8, or int8 storage")


def decode_fp8e4m3fn(tensor: torch.Tensor) -> torch.Tensor:
    value = raw_fp8_bits(tensor).to(torch.int32)
    exponent = torch.bitwise_and(torch.bitwise_right_shift(value, 3), 0xF)
    mantissa = torch.bitwise_and(value, 0x7)
    sign = torch.where(
        torch.bitwise_and(value, 0x80) != 0,
        -torch.ones((), dtype=torch.float32, device=value.device),
        torch.ones((), dtype=torch.float32, device=value.device),
    )
    mantissa_f = mantissa.to(torch.float32)
    decoded = torch.where(
        exponent == 0,
        mantissa_f * 0.001953125,
        (1.0 + mantissa_f * 0.125) * torch.exp2(exponent.float() - 7.0),
    ) * sign
    return torch.where(
        (exponent == 0xF) & (mantissa == 0x7),
        torch.full_like(decoded, float("nan")),
        decoded,
    )


def logical_cache(cache: torch.Tensor, layout: str, block_size: int) -> torch.Tensor:
    if cache.ndim != 4:
        raise ValueError("K/V cache must be rank 4")
    if layout not in {"NHD", "HND"}:
        raise ValueError("layout must be NHD or HND")
    if cache.shape[1] == block_size:
        return cache
    if layout == "HND" and cache.shape[2] == block_size:
        return cache.permute(0, 2, 1, 3)
    raise ValueError(f"cache must contain a {block_size}-token block dimension")


# Private compatibility names used inside the TOPSCC modules.
_raw_fp8_bits = raw_fp8_bits
_logical_cache = logical_cache

__all__ = ["decode_fp8e4m3fn", "logical_cache", "raw_fp8_bits"]
