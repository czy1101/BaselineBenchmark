"""Correctness/smoke test for stage 1 of the gfx936 BT64 Chunk/WY path."""

import torch

from gdn2_hip import load_extension


def reference(g: torch.Tensor, bt: int = 64) -> torch.Tensor:
    out = torch.empty_like(g, dtype=torch.float32)
    for start in range(0, g.shape[1], bt):
        end = min(start + bt, g.shape[1])
        out[:, start:end] = g[:, start:end].float().cumsum(dim=1)
    return out


def relative_rmse(ref: torch.Tensor, got: torch.Tensor) -> float:
    error = (got.float() - ref.float()).square().mean().sqrt()
    scale = ref.float().square().mean().sqrt().clamp_min(1e-8)
    return float((error / scale).item())


torch.cuda.set_device(0)
torch.manual_seed(2026)
ext = load_extension(verbose=False)

for dtype in (torch.float16, torch.bfloat16):
    for ksize in (64, 128, 256):
        for tokens in (13, 64, 130):
            g = (-0.1 * torch.rand(
                2, tokens, 3, ksize, device="cuda:0", dtype=dtype
            )).contiguous()
            ref = reference(g)
            got = ext.chunk_cumsum(g, 64)
            torch.cuda.synchronize()
            rel = relative_rmse(ref, got)
            max_abs = float((got - ref).abs().max().item())
            print(dtype, "K", ksize, "T", tokens,
                  "relative_rmse", rel, "max_abs", max_abs)
            assert rel < 1e-6

print("GDN2 HIP BT64 chunk cumsum: PASS")
