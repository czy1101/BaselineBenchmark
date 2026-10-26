"""Correctness test for Triton NSA GQA/query-tile dispatch."""

import os
import torch

from nsa_hip import parallel_nsa


@torch.no_grad()
def main():
    os.environ.setdefault("NSA_HYGON_USE_TRITON", "1")
    os.environ.setdefault("NSA_HYGON_TRITON_STRICT", "1")
    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.float16
    b, t, h, hq, d, block_size, slots = 1, 127, 2, 32, 64, 64, 2

    q = torch.randn(b, t, hq, d, device=device, dtype=dtype)
    k = torch.randn(b, t, h, d, device=device, dtype=dtype)
    v = torch.randn(b, t, h, d, device=device, dtype=dtype)
    indices = torch.randint(
        0, (t + block_size - 1) // block_size,
        (b, t, h, slots), device=device, dtype=torch.int32,
    )

    ref = torch.zeros(b, t, hq, d, device=device, dtype=torch.float32)
    group = hq // h
    scale = d ** -0.5
    for it in range(t):
        for iq in range(hq):
            ih = iq // group
            ids = indices[0, it, ih].long()[:, None] * block_size
            ids = (ids + torch.arange(block_size, device=device)[None, :]).flatten()
            ids = ids[(ids < t) & (ids <= it)]
            if ids.numel():
                score = k[0, ids, ih].float() @ q[0, it, iq].float()
                ref[0, it, iq] = torch.softmax(score * scale, 0) @ v[0, ids, ih].float()

    old_tile = os.environ.get("NSA_HYGON_TRITON_QTILE")
    for qtile in (1, 2, 4):
        os.environ["NSA_HYGON_TRITON_QTILE"] = str(qtile)
        out = parallel_nsa(q, k, v, block_indices=indices,
                           block_counts=slots, block_size=block_size, scale=scale)
        err = (out.float() - ref).square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-6)
        print(f"NSA Triton GQA qtile={qtile} relative RMSE:", err.item())
        assert torch.isfinite(out).all()
        assert err.item() < 3e-2
    if old_tile is None:
        os.environ.pop("NSA_HYGON_TRITON_QTILE", None)
    else:
        os.environ["NSA_HYGON_TRITON_QTILE"] = old_tile
    print("NSA Triton GQA/query-tile correctness: PASS")


if __name__ == "__main__":
    main()
