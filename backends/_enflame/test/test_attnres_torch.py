# Copyright 2026 FlagOS Contributors
"""Independent correctness tests for the Torch AttnRes implementation.

Run from the repository root:
    pytest backends/_enflame/test/test_attnres_torch.py -vs -ra

CPU cases always run; GCU cases skip when Torch-GCU is unavailable. These
tests do not import FlagAttention or Triton and do not measure performance.
"""

from __future__ import annotations

import os

import pytest
import torch

from backends._enflame.ops.attnres import (
    fused_attnres,
    fused_attnres_torch_baseline,
    torch_gcu_attnres_available,
)


pytestmark = pytest.mark.fused_attnres


@pytest.fixture(autouse=True)
def attnres_environment(monkeypatch):
    # Keep test results independent of the caller's benchmark configuration.
    monkeypatch.setenv("ATTNRES_TORCH_SCORE_MODE", "auto")
    monkeypatch.setenv("ATTNRES_TORCH_OPT_LEVEL", "1")


@pytest.fixture(params=("cpu", "gcu"))
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    if not torch_gcu_attnres_available():
        pytest.skip("Torch-GCU and an available GCU device are required")
    index = int(os.environ.get("ATTNRES_TEST_DEVICE", "0"))
    if index < 0 or index >= torch.gcu.device_count():
        pytest.skip(f"GCU device {index} is unavailable")
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")


def _inputs(device, dtype, sources, shape, output_norm, *, strided=False):
    torch.manual_seed(42)
    hidden = shape[-1]

    def random_tensor(size):
        # Generate the same quantized values on CPU and GCU. Slice after
        # transfer so the tested device actually receives strided inputs.
        stored = (*size[:-1], size[-1] * 2) if strided else size
        value = torch.randn(stored, device="cpu").to(device=device, dtype=dtype)
        return value[..., ::2] if strided else value

    residuals = [random_tensor(shape) for _ in range(sources)]
    query = random_tensor((hidden,))
    weight = random_tensor((hidden,))
    output_weight = random_tensor((hidden,)) if output_norm else None
    if strided:
        assert all(not tensor.is_contiguous() for tensor in (query, weight, *residuals))
    return query, residuals, weight, output_weight


@torch.no_grad()
def _reference(query, residuals, rms_weight, output_rms_weight, *, rms_eps, scale):
    """CPU FP64 equations; no RMSNorm, MM or BMM from the optimized paths."""
    values = torch.stack([value.detach().cpu().double() for value in residuals])
    query = query.detach().cpu().double().reshape(-1)
    weight = rms_weight.detach().cpu().double().reshape(-1)
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + rms_eps)
    scores = (normalized * weight * query).sum(dim=-1) * scale
    probability = scores.softmax(dim=0)
    output = (probability.unsqueeze(-1) * values).sum(dim=0)
    if output_rms_weight is not None:
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + rms_eps)
        output = output * output_rms_weight.detach().cpu().double().reshape(-1)
    return output.to(residuals[0].dtype), probability.float()


def _assert_result(actual, expected, residual):
    assert actual.shape == expected.shape
    assert actual.dtype == residual.dtype
    assert actual.device == residual.device
    assert bool(torch.isfinite(actual).all().item())
    if residual.dtype == torch.float32:
        atol, rtol = 2e-5, 2e-5
    else:
        # BF16 rounding can differ by one ULP; match the public benchmark's
        # output tolerance. Probabilities retain a separate strict FP32 bound.
        atol, rtol = 5e-3, 1e-2
    torch.testing.assert_close(actual.detach().float().cpu(), expected.float(), atol=atol, rtol=rtol)


_CORRECTNESS_CASES = (
    # Representative cases cover paths without multiplying every parameter.
    pytest.param(1, (1, 3, 7168), False, torch.bfloat16, "auto", "1", False,
                 7168**-0.5, 1e-6, id="single-source-D7168"),
    pytest.param(5, (128, 65), False, torch.bfloat16, "auto", "1", False,
                 0.13, 1e-4, id="auto-normalized-128"),
    pytest.param(9, (129, 64), True, torch.bfloat16, "auto", "1", False,
                 0.125, 1e-6, id="auto-factorized-129"),
    pytest.param(5, (2, 7, 4096), True, torch.float16, "auto", "1", False,
                 4096**-0.5, 1e-6, id="fp16-D4096-output-norm"),
    pytest.param(3, (2, 3, 33), True, torch.float32, "factorized", "1", True,
                 -0.2, 1e-4, id="fp32-strided-negative-scale"),
    pytest.param(5, (7, 65), False, torch.bfloat16, "normalized", "2", False,
                 0.13, 1e-4, id="opt2-normalized"),
    pytest.param(3, (7, 65), True, torch.float32, "factorized", "2", False,
                 0.125, 1e-4, id="opt2-factorized"),
)


@pytest.mark.parametrize(
    "sources,shape,output_norm,dtype,score_mode,opt_level,strided,scale,eps", _CORRECTNESS_CASES,
)
def test_attnres_torch_correctness(
    device, monkeypatch, dtype, score_mode, opt_level,
    sources, shape, output_norm, strided, scale, eps,
):
    monkeypatch.setenv("ATTNRES_TORCH_SCORE_MODE", score_mode)
    monkeypatch.setenv("ATTNRES_TORCH_OPT_LEVEL", opt_level)
    args = _inputs(device, dtype, sources, shape, output_norm, strided=strided)
    expected, expected_weights = _reference(*args, rms_eps=eps, scale=scale)
    actual, weights = fused_attnres_torch_baseline(*args, rms_eps=eps, scale=scale, return_weights=True)
    _assert_result(actual, expected, args[1][0])
    assert weights.shape == (sources, *shape[:-1])
    assert weights.dtype == torch.float32 and weights.device == device
    assert bool(torch.isfinite(weights).all().item())
    torch.testing.assert_close(weights.detach().cpu(), expected_weights, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(
        weights.sum(dim=0).cpu(), torch.ones(shape[:-1], dtype=torch.float32), atol=2e-6, rtol=2e-6,
    )
    plain = fused_attnres_torch_baseline(*args, rms_eps=eps, scale=scale, return_weights=False)
    assert isinstance(plain, torch.Tensor)
    _assert_result(plain, expected, args[1][0])


def test_attnres_torch_inference_and_fresh_outputs(device):
    args = _inputs(device, torch.float32, 3, (2, 3, 48), True)
    inputs = (args[0], *args[1], args[2], args[3])
    for value in inputs:
        value.requires_grad_(True)
    snapshots = [value.detach().clone() for value in inputs]
    first, weights = fused_attnres_torch_baseline(*args, scale=0.0, return_weights=True)
    assert not first.requires_grad and first.grad_fn is None
    assert not weights.requires_grad and weights.grad_fn is None
    assert first.data_ptr() not in {value.data_ptr() for value in inputs}
    for value, before in zip(inputs, snapshots):
        torch.testing.assert_close(value.detach(), before, atol=0, rtol=0)
    torch.testing.assert_close(weights.cpu(), torch.full_like(weights.cpu(), 1 / 3), atol=2e-6, rtol=2e-6)
    first_before = first.clone()
    with torch.no_grad():
        args[1][0].add_(0.5)
    second = fused_attnres_torch_baseline(*args, scale=0.0)
    expected, _ = _reference(*args, rms_eps=1e-6, scale=0.0)
    _assert_result(second, expected, args[1][0])
    assert second.data_ptr() != first.data_ptr()
    torch.testing.assert_close(first, first_before, atol=0, rtol=0)


def test_attnres_torch_zero_residuals(device, monkeypatch):
    # Exercise the opt2 factorized epsilon arithmetic at zero norm once.
    monkeypatch.setenv("ATTNRES_TORCH_OPT_LEVEL", "2")
    monkeypatch.setenv("ATTNRES_TORCH_SCORE_MODE", "factorized")
    args = _inputs(device, torch.bfloat16, 5, (3, 65), True)
    for value in args[1]:
        value.zero_()
    actual, weights = fused_attnres(*args, rms_eps=1e-4, return_weights=True)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.2), atol=2e-6, rtol=2e-6)


def test_attnres_torch_empty_rows(device):
    args = _inputs(device, torch.float16, 3, (2, 0, 65), True)
    for return_weights in (False, True):
        result = fused_attnres(*args, return_weights=return_weights)
        output, weights = result if return_weights else (result, None)
        assert output.shape == (2, 0, 65) and output.numel() == 0
        assert output.dtype == torch.float16 and output.device == device
        if return_weights:
            assert weights.shape == (3, 2, 0) and weights.numel() == 0
            assert weights.dtype == torch.float32 and weights.device == device


@pytest.mark.parametrize(
    "invalid,exception,message",
    (
        ("empty-residuals", ValueError, "at least one"),
        ("empty-hidden", ValueError, "non-empty hidden"),
        ("integer-residual", ValueError, "float16, bfloat16 or float32"),
        ("residual-shape", ValueError, "must have shape"),
        ("residual-dtype", ValueError, "same device and dtype"),
        ("query-size", ValueError, "query must contain"),
        ("query-device", ValueError, "query must have"),
    ),
)
def test_attnres_torch_invalid_inputs_cpu(invalid, exception, message):
    query, residuals, weight, output_weight = _inputs("cpu", torch.float32, 2, (3, 16), True)
    if invalid == "empty-residuals":
        residuals = []
    elif invalid == "empty-hidden":
        residuals = [torch.empty(3, 0)]
    elif invalid == "integer-residual":
        residuals = [value.int() for value in residuals]
    elif invalid == "residual-shape":
        residuals[1] = torch.empty(2, 16)
    elif invalid == "residual-dtype":
        residuals[1] = residuals[1].half()
    elif invalid == "query-size":
        query = query[:-1]
    elif invalid == "query-device":
        query = torch.empty(16, device="meta", dtype=torch.float32)
    with pytest.raises(exception, match=message):
        fused_attnres(query, residuals, weight, output_weight)


@pytest.mark.parametrize(
    "name,value",
    (("ATTNRES_TORCH_SCORE_MODE", "unsupported"), ("ATTNRES_TORCH_OPT_LEVEL", "3")),
)
def test_attnres_torch_invalid_configuration_cpu(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        fused_attnres(*_inputs("cpu", torch.float32, 2, (3, 16), False))
