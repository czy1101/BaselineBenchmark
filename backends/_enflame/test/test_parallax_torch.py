# Copyright 2026 FlagOS Contributors

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


OPS = Path(__file__).resolve().parents[1] / "ops" / "parallax"
sys.path.insert(0, str(OPS))
from parallax_torch import _dense_parallax, parallel_parallax_torch, torch_gcu_parallax_available
from backends._enflame.benchmarks._parallax_timing import check_close


pytestmark = pytest.mark.parallax


def _cpu_reference(q, r, k, v, scale=None, window_size=None):
    batch, seqlen, query_heads, dim = q.shape
    kv_heads = k.shape[2]
    groups = query_heads // kv_heads
    actual_scale = dim**-0.5 if scale is None else float(scale)

    query = q.reshape(batch, seqlen, kv_heads, groups, dim).permute(0, 2, 3, 1, 4)
    secondary_query = r.reshape(batch, seqlen, kv_heads, groups, dim).permute(0, 2, 3, 1, 4)
    key = k.permute(0, 2, 1, 3).unsqueeze(2)
    value = v.permute(0, 2, 1, 3).unsqueeze(2)

    logits = torch.matmul(query, key.transpose(-1, -2)) * actual_scale
    secondary_logits = torch.matmul(secondary_query, key.transpose(-1, -2))
    rows = torch.arange(seqlen, device=q.device).view(seqlen, 1)
    columns = torch.arange(seqlen, device=q.device).view(1, seqlen)
    valid = columns <= rows
    if window_size is not None:
        valid = valid & (columns >= rows - window_size + 1)
    valid = valid.view(1, 1, 1, seqlen, seqlen)

    probabilities = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)
    weighted_secondary = probabilities * secondary_logits
    mean_value = torch.matmul(probabilities, value)
    mean_secondary = weighted_secondary.sum(dim=-1, keepdim=True)
    secondary_value = torch.matmul(weighted_secondary, value)
    output = mean_value * (1.0 + mean_secondary) - secondary_value
    return output.permute(0, 3, 1, 2, 4).reshape(
        batch, seqlen, query_heads, dim
    )


@pytest.fixture(scope="module")
def gcu_device():
    if not torch_gcu_parallax_available():
        pytest.skip("torch_gcu environment required")
    index = int(os.environ.get("PARALLAX_TEST_DEVICE", "0"))
    if index < 0 or index >= torch.gcu.device_count():
        pytest.skip(f"GCU device {index} is unavailable")
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")


def _make_inputs(device, dtype, batch, length, heads, query_heads, dim, *, strided=False, trainable=(0, 1, 2, 3)):
    inputs = []
    for index, num_heads in enumerate((query_heads, query_heads, heads, heads)):
        tensor = torch.randn(batch, length, num_heads, dim * (2 if strided else 1), device="cpu")
        tensor = tensor.to(device=device, dtype=dtype)
        if strided:
            tensor = tensor[..., ::2]
            assert not tensor.is_contiguous()
        inputs.append(tensor.detach().requires_grad_(index in trainable))
    return tuple(inputs)


def _assert_reference(names, expected, actual, dtype, window=None):
    tolerance = 5e-3 if dtype == torch.float16 else 2e-2
    for name, reference, result in zip(names, expected, actual):
        if window == 1 and name in ("dq", "dr", "dk"):
            # One valid key makes O=V. Retain the established absolute bound
            # for zero gradients, instead of a relative error against zero.
            assert reference.abs().max().item() < 1e-10
            torch.testing.assert_close(
                result.float().cpu(), torch.zeros_like(reference.float()),
                rtol=0, atol=5e-6,
            )
        else:
            check_close(name, reference.detach().to(dtype), result.detach().cpu(), tolerance)


# B, T, H, HQ, D, window, query tile, strided, force streaming path.
_CASES = (
    pytest.param(1, 1, 1, 1, 16, None, 1, False, False, id="single-token"),
    pytest.param(1, 17, 2, 2, 32, None, 4, True, False, id="mha-strided"),
    pytest.param(2, 19, 2, 8, 64, None, 5, False, False, id="gqa-tail"),
    pytest.param(1, 23, 2, 6, 20, None, 1, True, True, id="streaming-one-row"),
    pytest.param(1, 17, 2, 8, 16, 1, 4, False, True, id="window-one"),
    pytest.param(2, 19, 2, 8, 16, 5, 5, True, True, id="streaming-window"),
    pytest.param(1, 35, 2, 16, 100, None, 6, False, True, id="gqa-eight-d100"),
    pytest.param(1, 33, 2, 2, 128, 64, 11, True, True, id="streaming-mha-d128"),
)


@pytest.mark.parametrize("backend", ["cpu", "gcu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,length,heads,query_heads,dim,window,tile,strided,streaming", _CASES)
def test_parallax_torch_forward_backward(
    request, backend, dtype, batch, length, heads, query_heads, dim, window, tile, strided, streaming,
):
    device = request.getfixturevalue("gcu_device") if backend == "gcu" else torch.device("cpu")
    torch.manual_seed(20261010)
    inputs = _make_inputs(device, dtype, batch, length, heads, query_heads, dim, strided=strided)
    reference_inputs = tuple(t.detach().cpu().double().requires_grad_(True) for t in inputs)
    gradient = torch.randn(inputs[0].shape, device=device, dtype=dtype)
    scale = None if window is None else 0.13
    expected = _cpu_reference(*reference_inputs, scale=scale, window_size=window)
    expected_grads = torch.autograd.grad(expected, reference_inputs, gradient.cpu().double())
    budget = batch * query_heads * length * tile
    if backend == "gcu":
        actual = parallel_parallax_torch(
            *inputs, scale=scale, window_size=window,
            max_score_elements=budget, checkpoint_tiles=True if streaming else None,
        )
    else:
        # Device-independent math only; the public entrypoint still requires GCU.
        actual = _dense_parallax(
            *inputs, dim**-0.5 if scale is None else scale, window, budget,
            checkpoint_tiles=True if streaming else None,
        )
    actual_grads = torch.autograd.grad(actual, inputs, gradient)
    _assert_reference(
        ("output", "dq", "dr", "dk", "dv"),
        (expected, *expected_grads), (actual, *actual_grads), dtype, window,
    )


@pytest.mark.parametrize("backend", ["cpu", "gcu"])
@pytest.mark.parametrize("trainable", [(0,), (1,), (2, 3), (3,)])
def test_parallax_torch_partial_gradients(request, backend, trainable):
    device = request.getfixturevalue("gcu_device") if backend == "gcu" else torch.device("cpu")
    torch.manual_seed(23)
    dtype = torch.bfloat16
    inputs = _make_inputs(device, dtype, 1, 19, 2, 4, 16, strided=True, trainable=trainable)
    reference_inputs = tuple(
        t.detach().cpu().double().requires_grad_(index in trainable)
        for index, t in enumerate(inputs)
    )
    gradient = torch.randn(inputs[0].shape, device=device, dtype=dtype)
    expected = _cpu_reference(*reference_inputs, scale=0.13, window_size=7)
    expected_grads = torch.autograd.grad(
        expected, tuple(reference_inputs[i] for i in trainable), gradient.cpu().double(),
    )
    if backend == "gcu":
        actual = parallel_parallax_torch(
            *inputs, scale=0.13, window_size=7, max_score_elements=4 * 19 * 5,
            checkpoint_tiles=True,
        )
    else:
        actual = _dense_parallax(*inputs, 0.13, 7, 4 * 19 * 5, checkpoint_tiles=True)
    actual_grads = torch.autograd.grad(actual, tuple(inputs[i] for i in trainable), gradient)
    names = ("output", *(("dq", "dr", "dk", "dv")[i] for i in trainable))
    _assert_reference(names, (expected, *expected_grads), (actual, *actual_grads), dtype)


@pytest.mark.parametrize("backend", ["cpu", "gcu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_parallax_torch_packed_sequences(request, backend, dtype):
    device = request.getfixturevalue("gcu_device") if backend == "gcu" else torch.device("cpu")
    torch.manual_seed(61)
    inputs = _make_inputs(device, dtype, 1, 27, 2, 4, 16, strided=True)
    reference_inputs = tuple(t.detach().cpu().double().requires_grad_(True) for t in inputs)
    offsets = [0, 0, 7, 7, 19, 27]
    gradient = torch.randn(inputs[0].shape, device=device, dtype=dtype)
    expected = torch.cat([
        _cpu_reference(*(t[:, start:end] for t in reference_inputs), scale=0.13, window_size=7)
        if start != end else reference_inputs[0][:, start:end]
        for start, end in zip(offsets, offsets[1:])
    ], dim=1)
    expected_grads = torch.autograd.grad(expected, reference_inputs, gradient.cpu().double())
    kwargs = {
        "scale": 0.13, "window_size": 7, "max_score_elements": 4 * 27 * 3,
        "checkpoint_tiles": True,
        "cu_seqlens": torch.tensor(offsets, dtype=torch.int32, device=device),
    }
    if backend == "gcu":
        actual = parallel_parallax_torch(*inputs, **kwargs)
    else:
        # Host routing check only. Bypass the device guard, not attention math.
        with patch("parallax_torch._validate_inputs"):
            actual = parallel_parallax_torch(*inputs, **kwargs)
    actual_grads = torch.autograd.grad(actual, inputs, gradient)
    _assert_reference(
        ("output", "dq", "dr", "dk", "dv"),
        (expected, *expected_grads), (actual, *actual_grads), dtype,
    )


def test_parallax_torch_streaming_policy():
    inputs = _make_inputs("cpu", torch.bfloat16, 1, 64, 2, 4, 16)
    with patch("parallax_torch._CHECKPOINT_SCORE_ELEMENTS", 4096):
        output = _dense_parallax(*inputs, 0.25, None, 4 * 64 * 4)
        assert type(output.grad_fn).__name__ == "_StreamingParallaxBackward"
        with torch.no_grad():
            output = _dense_parallax(*inputs, 0.25, None, 4 * 64 * 4)
        assert output.grad_fn is None


def test_parallax_torch_repeated_backward():
    torch.manual_seed(53)
    inputs = _make_inputs("cpu", torch.bfloat16, 2, 19, 2, 8, 16, strided=True)
    gradient = torch.randn(2, 19, 8, 32, device="cpu").to(torch.bfloat16)[..., ::2]
    originals = tuple(t.detach().clone() for t in (*inputs, gradient))
    output = _dense_parallax(*inputs, 0.13, 7, 2 * 8 * 19 * 5, checkpoint_tiles=True)
    first = torch.autograd.grad(output, inputs, gradient, retain_graph=True)
    second = torch.autograd.grad(output, inputs, gradient)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for original, current in zip(originals, (*inputs, gradient)):
        torch.testing.assert_close(original, current, rtol=0, atol=0)


def test_parallax_torch_double_backward():
    torch.manual_seed(59)
    inputs = _make_inputs("cpu", torch.float32, 1, 7, 2, 4, 16)
    gradient = torch.randn_like(inputs[0])
    vectors = tuple(torch.randn_like(t) for t in inputs)
    results = []
    for streaming_backward in (False, True):
        output = _dense_parallax(
            *inputs, 0.13, 5, 4 * 7 * 3,
            checkpoint_tiles=True, streaming_backward=streaming_backward,
        )
        first = torch.autograd.grad(output, inputs, gradient, create_graph=True)
        scalar = sum((g * vector).sum() for g, vector in zip(first, vectors))
        second = torch.autograd.grad(scalar, inputs)
        results.append((*first, *second))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
