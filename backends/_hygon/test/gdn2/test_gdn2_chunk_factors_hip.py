"""Validate fused qg/kn/ke factors for the gfx936 BT64 Chunk/WY path."""

import torch

from gdn2_hip import load_extension


def chunk_cumsum_reference(g: torch.Tensor, bt: int = 64) -> torch.Tensor:
    out = torch.empty_like(g, dtype=torch.float32)
    for start in range(0, g.shape[1], bt):
        end = min(start + bt, g.shape[1])
        out[:, start:end] = g[:, start:end].float().cumsum(dim=1)
    return out


def check(ref: torch.Tensor, got: torch.Tensor, name: str) -> None:
    diff = got.float() - ref.float()
    rel = diff.square().mean().sqrt() / (
        ref.float().square().mean().sqrt() + 1e-8
    )
    max_abs = diff.abs().max()
    print(name, "relative_rmse", float(rel), "max_abs", float(max_abs))
    assert float(rel) < 2e-6


torch.cuda.set_device(0)
torch.manual_seed(2026)
ext = load_extension(verbose=False)

for dtype in (torch.float16, torch.bfloat16):
    for ksize in (64, 128, 256):
        for tokens in (13, 64, 130):
            shape = (2, tokens, 3, ksize)
            q = torch.randn(shape, device="cuda:0", dtype=dtype).contiguous()
            k = torch.randn(shape, device="cuda:0", dtype=dtype).contiguous()
            b = torch.rand(shape, device="cuda:0", dtype=dtype).contiguous()
            g = (-0.02 * torch.rand(
                shape, device="cuda:0", dtype=dtype
            )).contiguous()

            G = ext.chunk_cumsum(g, 64)
            qg, kn, ke = ext.chunk_factors(q, k, b, G)
            torch.cuda.synchronize()

            ep = torch.exp(G)
            em = torch.exp(-G)
            prefix = f"{dtype}/K{ksize}/T{tokens}"
            check(q.float() * ep, qg, prefix + "/qg")
            check(k.float() * em, kn, prefix + "/kn")
            check(k.float() * b.float() * ep, ke, prefix + "/ke")

print("GDN2 HIP BT64 chunk factors: PASS")
