# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
from pathlib import Path

import pytest
import torch


OPS_DIR = Path(__file__).resolve().parents[1] / "ops" / "sage_attention"
sys.path.insert(0, str(OPS_DIR))

import attn_qk_int8_per_block_native as native
from quant_per_block import per_block_int8


def _gcu_available():
    try:
        __import__("torch_gcu")
        return hasattr(torch, "gcu") and bool(torch.gcu.is_available())
    except Exception:
        return False


requires_gcu = pytest.mark.skipif(
    not _gcu_available(), reason="requires an available Enflame GCU"
)

requires_long_gcu_tests = pytest.mark.skipif(
    os.environ.get("FLAG_ATTN_RUN_LONG_GCU_TESTS") != "1",
    reason="set FLAG_ATTN_RUN_LONG_GCU_TESTS=1 to run the 20 long-shape checks",
)


_LONG_BATCH_SEQ_SHAPES = (
    (1, 1024),
    (4, 1024),
    (1, 4096),
    (1, 8192),
    (1, 16384),
)


def _canonical_hnd(tensor, tensor_layout):
    return tensor if tensor_layout == "HND" else tensor.transpose(1, 2)


def _reference(q, k, v, q_scale, k_scale, tensor_layout):
    q = _canonical_hnd(q, tensor_layout).cpu().float()
    k = _canonical_hnd(k, tensor_layout).cpu().float()
    v = _canonical_hnd(v, tensor_layout).cpu().float()
    q_rows = q_scale.cpu().float().repeat_interleave(128, dim=-1)[
        ..., : q.shape[-2], None
    ]
    k_rows = k_scale.cpu().float().repeat_interleave(64, dim=-1)[
        ..., : k.shape[-2], None
    ]
    q = q * q_rows
    k = k * k_rows
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2)) * math.log(2.0)
    output = torch.matmul(torch.softmax(logits, dim=-1), v)
    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return output


def _direct_inputs(
    tensor_layout,
    batch_size=1,
    num_query_heads=2,
    num_kv_heads=2,
    qo_len=129,
    kv_len=70,
    head_dim=64,
    int8_limit=8,
):
    if tensor_layout == "HND":
        q_shape = (batch_size, num_query_heads, qo_len, head_dim)
        kv_shape = (batch_size, num_kv_heads, kv_len, head_dim)
    else:
        q_shape = (batch_size, qo_len, num_query_heads, head_dim)
        kv_shape = (batch_size, kv_len, num_kv_heads, head_dim)
    q = torch.randint(
        -int8_limit, int8_limit, q_shape, device="gcu", dtype=torch.int8
    )
    k = torch.randint(
        -int8_limit, int8_limit, kv_shape, device="gcu", dtype=torch.int8
    )
    v = torch.randn(kv_shape, device="gcu", dtype=torch.float16)
    q_scale = 0.005 + 0.015 * torch.rand(
        (batch_size, num_query_heads, (qo_len + 127) // 128),
        device="gcu",
        dtype=torch.float32,
    )
    k_scale = 0.005 + 0.015 * torch.rand(
        (batch_size, num_kv_heads, (kv_len + 63) // 64),
        device="gcu",
        dtype=torch.float32,
    )
    return q, k, v, q_scale, k_scale


@pytest.mark.sage_attention
@requires_gcu
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_torch_gcu_matches_reference_with_tails_and_gqa(
    tensor_layout, num_kv_heads, output_dtype
):
    torch.manual_seed(2026)
    inputs = _direct_inputs(
        tensor_layout,
        num_query_heads=2,
        num_kv_heads=num_kv_heads,
        qo_len=129,
        kv_len=70,
        head_dim=64,
    )
    q, k, v, q_scale, k_scale = inputs
    actual, lse = native.forward(
        *inputs,
        tensor_layout=tensor_layout,
        output_dtype=output_dtype,
        backend="torch-gcu",
    )
    expected = _reference(
        q, k, v, q_scale, k_scale, tensor_layout
    )
    atol = 4e-2 if output_dtype == torch.bfloat16 else 3e-2
    rtol = 4e-2 if output_dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(actual.cpu().float(), expected, atol=atol, rtol=rtol)
    assert actual.dtype == output_dtype
    assert tuple(actual.shape) == tuple(q.shape)
    assert lse.device.type == "cpu"
    assert lse.numel() == 0


@pytest.mark.sage_attention
@requires_gcu
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_torch_gcu_accepts_per_block_quantized_inputs(tensor_layout):
    torch.manual_seed(9)
    q = torch.randn((1, 2, 128, 128), device="gcu", dtype=torch.float16)
    k = torch.randn((1, 1, 128, 128), device="gcu", dtype=torch.float16)
    v = torch.randn_like(k)
    if tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, tensor_layout=tensor_layout
    )
    actual, _ = native.forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        backend="torch-gcu",
    )
    expected = _reference(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout,
    )
    torch.testing.assert_close(
        actual.cpu().float(), expected, atol=3e-2, rtol=3e-2
    )


@pytest.mark.sage_attention
@requires_gcu
@requires_long_gcu_tests
@pytest.mark.parametrize("batch_size,seq_len", _LONG_BATCH_SEQ_SHAPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_torch_gcu_matches_triton_on_long_benchmark_shapes(
    batch_size, seq_len, head_dim, output_dtype
):
    """Check every performance-table shape without an O(T^2) CPU reference.

    The small tail/GQA tests above establish the native path against an
    independent CPU reference.  This slow opt-in test then compares the full
    20-shape performance table with the existing Triton implementation.
    """
    torch.manual_seed(2026)
    inputs = _direct_inputs(
        "HND",
        batch_size=batch_size,
        num_query_heads=32,
        num_kv_heads=32,
        qo_len=seq_len,
        kv_len=seq_len,
        head_dim=head_dim,
        int8_limit=16,
    )

    with torch.no_grad():
        actual, actual_lse = native.forward(
            *inputs,
            output_dtype=output_dtype,
            backend="torch-gcu",
        )
        expected, expected_lse = native.forward(
            *inputs,
            output_dtype=output_dtype,
            backend="triton",
        )

    assert bool(torch.isfinite(actual).all().item())
    assert actual_lse.numel() == 0
    assert expected_lse.numel() == 0
    atol = 4e-2 if output_dtype == torch.bfloat16 else 3e-2
    rtol = 4e-2 if output_dtype == torch.bfloat16 else 3e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.sage_attention
def test_auto_uses_triton_for_attention_mask(monkeypatch):
    sentinel = object()

    def fake_triton(*args, **kwargs):
        return sentinel, kwargs

    monkeypatch.setattr(native, "_triton_forward", fake_triton)
    q = torch.zeros((1, 1, 1, 64), dtype=torch.int8)
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    output, options = native.forward(
        q,
        q,
        q.float(),
        torch.ones((1, 1, 1)),
        torch.ones((1, 1, 1)),
        attn_mask=mask,
        backend="auto",
    )
    assert output is sentinel
    assert options["attn_mask"] is mask


@pytest.mark.sage_attention
def test_explicit_native_backend_rejects_attention_mask():
    q = torch.zeros((1, 1, 1, 64), dtype=torch.int8)
    mask = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="attn_mask"):
        native.forward(
            q,
            q,
            q.float(),
            torch.ones((1, 1, 1)),
            torch.ones((1, 1, 1)),
            attn_mask=mask,
            backend="torch-gcu",
        )


@pytest.mark.sage_attention
def test_auto_without_native_backend_resolves_to_triton(monkeypatch):
    unavailable = lambda: (None, "not installed")
    monkeypatch.setattr(native, "_load_torch_gcu_sdpa", unavailable)
    monkeypatch.setattr(native, "_load_vllm_gcu_attention", unavailable)
    monkeypatch.setattr(native, "_load_flash_attention", unavailable)
    assert native.resolve_backend("auto", device="gcu") == "triton"


@pytest.mark.sage_attention
def test_torch_gcu_lse_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(native, "_load_torch_gcu_sdpa", lambda: (lambda: None, ""))
    q = torch.zeros((1, 1, 1, 64), dtype=torch.float16)
    with pytest.raises(native.NativeBackendUnavailable, match="does not expose LSE"):
        native._forward_torch_gcu(q, q, q, return_lse=True)
