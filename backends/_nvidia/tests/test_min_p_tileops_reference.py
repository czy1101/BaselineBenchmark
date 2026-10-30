# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Numerical validation of native TileOPs Min-P; not a performance benchmark.

Run separately from benchmarks with:
    python -m pytest -q backends/_nvidia/tests/test_min_p_tileops_reference.py
"""

import pytest
import torch

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def tileops_baseline():
    pytest.importorskip("tilelang", minversion="0.1.12")
    pytest.importorskip("tileops")
    from tileops.backend import BUILTIN
    from tileops.sampling import MinPMaskFwdOp

    return MinPMaskFwdOp(target=BUILTIN)


def _check_tileops_baseline(apply, logits, min_p):
    before, thresholds_before = logits.clone(), min_p.clone()
    threshold = logits.float().amax(dim=1, keepdim=True) + min_p.log()[:, None]
    expected = logits.masked_fill(logits.float() < threshold, float("-inf"))
    output = apply(logits, min_p)
    torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(min_p, thresholds_before, rtol=0, atol=0)
    assert output.shape == logits.shape
    assert output.dtype == logits.dtype
    assert output.device == logits.device
    assert output.data_ptr() != logits.data_ptr()


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 128256), torch.bfloat16),
        ((1, 128256), torch.float32),
        ((64, 128256), torch.bfloat16),
        ((256, 128256), torch.float16),
        ((256, 128256), torch.bfloat16),
        ((128, 151936), torch.bfloat16),
        ((17, 129280), torch.bfloat16),
        ((100, 32000), torch.float16),
    ],
)
def test_apply_min_p_tileops_baseline_shapes(tileops_baseline, shape, dtype):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    logits = torch.randn(shape, dtype=dtype, device="cuda", generator=generator)
    min_p = torch.full((shape[0],), 0.1, dtype=torch.float32, device="cuda")
    _check_tileops_baseline(tileops_baseline, logits, min_p)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", [1, 17, 1025])
def test_apply_min_p_tileops_baseline_boundaries(tileops_baseline, dtype, vocab):
    # Exercise scalar-vector fallback, tail masking, disabled/tied rows and NaNs.
    logits = torch.zeros((5, vocab), dtype=dtype, device="cuda")
    logits[0, 0] = float("inf")
    logits[1, :] = float("-inf")
    logits[3, 0] = float("nan")
    logits[4, 0] = 2
    min_p = torch.tensor([0, 0.5, 1, 0.1, 1], dtype=torch.float32, device="cuda")
    _check_tileops_baseline(tileops_baseline, logits, min_p)
