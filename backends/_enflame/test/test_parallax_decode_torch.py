"""Correctness tests for the pure Torch-GCU Parallax decode baseline."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch


OPS = Path(__file__).resolve().parents[1] / "ops" / "parallax"
sys.path.insert(0, str(OPS))
from parallax_decode_torch import (
    parallax_attn_with_kvcache_torch_gcu,
    parallax_decode_torch_gcu,
    torch_gcu_parallax_decode_available,
)


# Same ten (B, L, HQ, HKV, D) decode shapes as the FlagAttention benchmark.
CASES = (
    (1, 512, 8, 8, 64),
    (1, 512, 8, 8, 128),
    (1, 1024, 8, 8, 128),
    (1, 2048, 8, 8, 128),
    (1, 4096, 8, 8, 128),
    (1, 16384, 8, 8, 128),
    (2, 1000, 8, 2, 128),
    (2, 2048, 8, 2, 64),
    (2, 2048, 8, 2, 128),
    (1, 8192, 8, 2, 128),
)


@pytest.fixture(scope="module")
def gcu_device():
    if not torch_gcu_parallax_decode_available():
        pytest.skip("torch_gcu environment required")
    index = int(os.environ.get("PARALLAX_TEST_DEVICE", "0"))
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")


def _inputs(device, dtype, batch, length, query_heads, kv_heads, dim):
    generator = torch.Generator(device=device).manual_seed(
        2026 + batch * 1009 + length * 17 + query_heads + dim
    )
    q = torch.randn(batch, 1, query_heads, dim, device=device, dtype=dtype, generator=generator)
    r = torch.randn(batch, 1, query_heads, dim, device=device, dtype=dtype, generator=generator) * 0.5
    k = torch.randn(batch, length, kv_heads, dim, device=device, dtype=dtype, generator=generator)
    v = torch.randn(batch, length, kv_heads, dim, device=device, dtype=dtype, generator=generator)
    return q, r, k, v


def _reference(q, r, k, v, scale, window_left=-1):
    """Independent pointwise FP32 oracle (no batched matmul)."""
    groups = q.shape[2] // k.shape[2]
    key = k.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
    value = v.float().repeat_interleave(groups, dim=2).permute(0, 2, 1, 3)
    primary = (q[:, 0].float()[:, :, None, :] * key).sum(-1) * scale
    secondary = (r[:, 0].float()[:, :, None, :] * key).sum(-1)
    if window_left >= 0:
        first = max(k.shape[1] - window_left, 0)
        positions = torch.arange(k.shape[1], device=k.device)
        primary = primary.masked_fill(positions[None, None, :] < first, -float("inf"))
    pivot = primary.amax(dim=-1, keepdim=True)
    pivot = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    weights = torch.exp(primary - pivot)
    weighted_secondary = weights * secondary
    denominator = weights.sum(dim=-1, keepdim=True)
    inverse = torch.where(denominator > 0, denominator.reciprocal(), 0)
    value_sum = (weights[..., None] * value).sum(dim=2)
    secondary_value = (weighted_secondary[..., None] * value).sum(dim=2)
    result = value_sum * inverse * (1 + weighted_secondary.sum(-1, keepdim=True) * inverse)
    result -= secondary_value * inverse
    return result[:, None]


def _relative_error(actual, expected):
    difference = (actual.float() - expected.float()).abs().max().item()
    normalizer = max(expected.float().abs().max().item(), 1e-6)
    return difference / normalizer


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("batch,length,query_heads,kv_heads,dim", CASES)
def test_decode_torch_matches_reference(
    gcu_device, dtype, batch, length, query_heads, kv_heads, dim
):
    q, r, k, v = _inputs(
        gcu_device, dtype, batch, length, query_heads, kv_heads, dim
    )
    scale = 1 / math.sqrt(dim)
    actual = parallax_decode_torch_gcu(q, r, k, v, scale)
    expected = _reference(q, r, k, v, scale)
    assert actual.shape == q.shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    assert _relative_error(actual, expected) < 1e-2


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_decode_torch_seqused_window_and_out(gcu_device, dtype):
    batch, length, query_heads, kv_heads, dim = 2, 513, 8, 2, 64
    q, r, k, v = _inputs(
        gcu_device, dtype, batch, length, query_heads, kv_heads, dim
    )
    lengths = (257, 501)
    seqused_k = torch.tensor(lengths, device=gcu_device, dtype=torch.int32)
    out = torch.empty_like(q)
    actual = parallax_attn_with_kvcache_torch_gcu(
        q, r, k, v, seqused_k=seqused_k, window_size=(73, 0), out=out
    )
    expected = torch.cat(
        [
            _reference(
                q[i : i + 1], r[i : i + 1],
                k[i : i + 1, :active], v[i : i + 1, :active],
                dim ** -0.5, window_left=73,
            )
            for i, active in enumerate(lengths)
        ],
        dim=0,
    )
    assert actual is out
    assert _relative_error(actual, expected) < 1e-2


def test_decode_torch_noncontiguous_inputs(gcu_device):
    batch, length, heads, dim = 2, 128, 8, 64
    q, r, k, v = _inputs(
        gcu_device, torch.bfloat16, batch, length, heads, heads, dim
    )
    q = torch.cat((q, q), dim=1)[:, -1:]
    r = torch.cat((r, r), dim=1)[:, -1:]
    assert not q.is_contiguous() and not r.is_contiguous()
    actual = parallax_decode_torch_gcu(q, r, k, v)
    expected = _reference(q, r, k, v, dim ** -0.5)
    assert _relative_error(actual, expected) < 1e-2
