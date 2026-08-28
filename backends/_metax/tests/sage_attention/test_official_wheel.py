import pytest
import torch

from backends._metax.ops.sage_attention import runtime_version, sageattn


def test_official_wheel_contract():
    if not torch.cuda.is_available():
        pytest.skip("MetaX device is unavailable")

    try:
        runtime_version()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    q = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_before = q.clone()
    k_before = k.clone()
    v_before = v.clone()

    with torch.inference_mode():
        output = sageattn(
            q, k, v, tensor_layout="NHD", is_causal=True
        )

    assert torch.isfinite(output).all()
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)
    assert torch.equal(v, v_before)
