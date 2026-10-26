from __future__ import annotations

import pytest
import torch

from backends._hygon.ops.attnes import fused_attnres as hip_attnres
from flag_attn.FLA.attnres.fused import fused_attnres as flag_attnres


D = 7168
CASES = (
    (2, 1),
    (5, 128),
    (5, 1024),
    (9, 128),
    (9, 8192),
)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("sources,rows", CASES)
@pytest.mark.parametrize("output_norm", (False, True))
def test_attnes_hip_matches_flag_attention(
    dtype,
    sources,
    rows,
    output_norm,
):
    if not torch.cuda.is_available():
        pytest.skip("Hygon device is unavailable")

    torch.manual_seed(2026 + sources + rows)

    query = torch.randn((D,), device="cuda", dtype=dtype)
    residuals = [
        torch.randn((rows, D), device="cuda", dtype=dtype)
        for _ in range(sources)
    ]
    rms_weight = torch.randn((D,), device="cuda", dtype=dtype)

    output_rms_weight = (
        torch.randn((D,), device="cuda", dtype=dtype)
        if output_norm
        else None
    )

    hip_output = hip_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
    )

    flag_output = flag_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(
        hip_output.float(),
        flag_output.float(),
        rtol=5e-2,
        atol=5e-2,
    )
