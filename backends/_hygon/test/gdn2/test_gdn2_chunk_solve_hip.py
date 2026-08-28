"""Validate the BT64 unit-lower solve used by GDN2 WY auxiliaries."""

import torch

from gdn2_hip import load_extension


def pad_chunks(x: torch.Tensor, bt: int = 64) -> torch.Tensor:
    b, t, h, d = x.shape
    nt = (t + bt - 1) // bt
    tp = nt * bt
    if tp != t:
        x = torch.cat((x, torch.zeros(
            b, tp - t, h, d, device=x.device, dtype=x.dtype
        )), dim=1)
    return x.view(b, nt, bt, h, d).permute(0, 3, 1, 2, 4).contiguous()


def unchunk(x: torch.Tensor, tokens: int) -> torch.Tensor:
    b, h, nt, bt, d = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(b, nt * bt, h, d)[:, :tokens]


def check(ref: torch.Tensor, got: torch.Tensor, name: str) -> None:
    diff = got.float() - ref.float()
    rel = diff.square().mean().sqrt() / (
        ref.float().square().mean().sqrt() + 1e-8
    )
    max_abs = diff.abs().max()
    print(name, "relative_rmse", float(rel), "max_abs", float(max_abs))
    assert float(rel) < 2e-5


torch.cuda.set_device(0)
torch.manual_seed(2026)
ext = load_extension(verbose=False)

for dtype in (torch.float16, torch.bfloat16):
    for ksize in (64, 128, 256):
        for tokens in (13, 64, 130):
            shape = (1, tokens, 2, ksize)
            q = (torch.randn(shape, device="cuda:0", dtype=dtype) /
                 ksize**0.5).contiguous()
            k = (torch.randn(shape, device="cuda:0", dtype=dtype) /
                 ksize**0.5).contiguous()
            b = (0.1 * torch.rand(
                shape, device="cuda:0", dtype=dtype
            )).contiguous()
            g = (-0.01 * torch.rand(
                shape, device="cuda:0", dtype=dtype
            )).contiguous()
            G = ext.chunk_cumsum(g, 64)
            qg, kn, ke = ext.chunk_factors(q, k, b, G)
            _, Akk = ext.chunk_scores(qg, kn, ke, ksize**-0.5, 64)

            eye = torch.eye(64, device="cuda:0", dtype=torch.float32)
            L = Akk + eye.view(1, 1, 1, 64, 64)
            for label, rhs in (
                ("wy", ke),
                ("u", torch.randn(
                    1, tokens, 2, 37, device="cuda:0", dtype=torch.float32
                )),
            ):
                got = ext.chunk_solve(Akk, rhs.contiguous(), 64)
                rhs_chunks = pad_chunks(rhs.contiguous())
                ref_chunks = torch.linalg.solve_triangular(
                    L, rhs_chunks, upper=False
                )
                ref = unchunk(ref_chunks, tokens)
                torch.cuda.synchronize()
                check(ref, got, f"{dtype}/K{ksize}/T{tokens}/{label}")

print("GDN2 HIP BT64 chunk solve: PASS")
