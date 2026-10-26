"""Validate causal Aqk/Akk matrices for the gfx936 BT64 Chunk/WY path."""

import math
import torch

from gdn2_hip import load_extension


def to_chunks(x: torch.Tensor, bt: int = 64) -> torch.Tensor:
    b, t, h, d = x.shape
    nt = (t + bt - 1) // bt
    tp = nt * bt
    if tp != t:
        x = torch.cat((x, torch.zeros(
            b, tp - t, h, d, device=x.device, dtype=x.dtype
        )), dim=1)
    return x.view(b, nt, bt, h, d).permute(0, 3, 1, 2, 4).contiguous()


def check(ref: torch.Tensor, got: torch.Tensor, name: str) -> None:
    diff = got.float() - ref.float()
    rel = diff.square().mean().sqrt() / (
        ref.float().square().mean().sqrt() + 1e-8
    )
    max_abs = diff.abs().max()
    print(name, "relative_rmse", float(rel), "max_abs", float(max_abs))
    assert float(rel) < 5e-5


torch.cuda.set_device(0)
torch.manual_seed(2026)
ext = load_extension(verbose=False)

for dtype in (torch.float16, torch.bfloat16):
    for ksize in (64, 128, 256):
        for tokens in (13, 64, 130):
            shape = (1, tokens, 2, ksize)
            q = torch.randn(shape, device="cuda:0", dtype=dtype).contiguous()
            k = torch.randn(shape, device="cuda:0", dtype=dtype).contiguous()
            b = torch.rand(shape, device="cuda:0", dtype=dtype).contiguous()
            g = (-0.02 * torch.rand(
                shape, device="cuda:0", dtype=dtype
            )).contiguous()
            scale = ksize ** -0.5

            G = ext.chunk_cumsum(g, 64)
            qg, kn, ke = ext.chunk_factors(q, k, b, G)
            Aqk, Akk = ext.chunk_scores(qg, kn, ke, scale, 64)
            torch.cuda.synchronize()

            qgc, knc, kec = map(to_chunks, (qg, kn, ke))
            ref_qk = torch.matmul(qgc, knc.transpose(-1, -2)) * scale
            ref_kk = torch.matmul(kec, knc.transpose(-1, -2))
            row = torch.arange(64, device="cuda:0")
            causal = row[:, None] >= row[None, :]
            strict = row[:, None] > row[None, :]
            nt = (tokens + 63) // 64
            valid = (torch.arange(nt * 64, device="cuda:0") < tokens)
            valid = valid.view(nt, 64)
            pair_valid = valid[:, :, None] & valid[:, None, :]
            ref_qk = ref_qk.masked_fill(
                ~(causal.view(1, 1, 1, 64, 64) &
                  pair_valid.view(1, 1, nt, 64, 64)), 0
            )
            ref_kk = ref_kk.masked_fill(
                ~(strict.view(1, 1, 1, 64, 64) &
                  pair_valid.view(1, 1, nt, 64, 64)), 0
            )
            prefix = f"{dtype}/K{ksize}/T{tokens}"
            check(ref_qk, Aqk, prefix + "/Aqk")
            check(ref_kk, Akk, prefix + "/Akk")

print("GDN2 HIP BT64 chunk scores: PASS")
