import torch

from kda_hip import chunk_kda_dtk


def rel_rmse(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() /
            (b.square().mean().sqrt() + 1e-6)).item()


torch.manual_seed(123)
for dtype in (torch.float16, torch.bfloat16):
    B, T, H, K, V = 1, 37, 2, 128, 128
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    beta = torch.randn(B, T, H, device="cuda", dtype=dtype)
    ref, ref_state = chunk_kda_dtk(
        q, k, v, g, beta, output_final_state=True,
        hip_backend="recurrent")
    for chunk_size in (16, 32, 64):
        out, state = chunk_kda_dtk(
            q, k, v, g, beta, output_final_state=True,
            chunk_size=chunk_size, gemm_dtype=dtype)
        eo, es = rel_rmse(out, ref), rel_rmse(state, ref_state)
        print(dtype, "chunk_size", chunk_size, "output", eo, "state", es)
        assert torch.isfinite(out).all() and torch.isfinite(state).all()
        assert eo < (3e-2 if dtype == torch.float16 else 5e-2)
        assert es < (3e-2 if dtype == torch.float16 else 5e-2)
print("KDA DTK GEMM correctness: PASS")
