"""Regression tests for the forward-only NSA E2E composition."""

import pytest
import torch

from tileops.ops import NSAForwardVarlenOp


@pytest.mark.smoke
def test_nsa_e2e_native_metadata_matches_precomputed() -> None:
    """Native metadata and precomputed metadata must be exactly equivalent.

    Mathematical correctness of each component is covered by the existing
    pooling, compression, top-k, selected-attention, sliding-window, and
    fusion tests. This test protects their E2E orchestration and public API.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    total_tokens = 128
    heads = 32
    heads_kv = 2
    dim = 128
    block_size = 32
    selected_blocks = 16
    window_size = 128
    dtype = torch.float16

    q = (
        torch.randn(
            total_tokens, heads, dim, dtype=dtype, device="cuda"
        ) * 0.1
    ).contiguous()
    k = (
        torch.randn(
            total_tokens, heads_kv, dim, dtype=dtype, device="cuda"
        ) * 0.1
    ).contiguous()
    v = (
        torch.randn(
            total_tokens, heads_kv, dim, dtype=dtype, device="cuda"
        ) * 0.1
    ).contiguous()

    gates = torch.softmax(
        torch.randn(
            total_tokens, heads, 3,
            dtype=torch.float32,
            device="cuda",
        ),
        dim=-1,
    ).to(dtype)
    g_cmp = gates[:, :, 0].contiguous()
    g_slc = gates[:, :, 1].contiguous()
    g_swa = gates[:, :, 2].contiguous()
    offsets = torch.tensor(
        [0, total_tokens], dtype=torch.int32, device="cuda"
    )

    named_inputs = {
        "q": q,
        "k": k,
        "v": v,
        "g_cmp": g_cmp,
        "g_slc": g_slc,
        "g_swa": g_swa,
        "offsets": offsets,
    }
    snapshots = {
        name: tensor.clone()
        for name, tensor in named_inputs.items()
    }

    op = NSAForwardVarlenOp(
        seq_num=1,
        c_seq_len=total_tokens,
        max_seqlen=total_tokens,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        chunk_num=total_tokens // block_size,
        block_size=block_size,
        selected_blocks=selected_blocks,
        window_size=window_size,
        scale=dim**-0.5,
        accum_dtype=torch.float32,
        tune=False,
    )

    with torch.no_grad():
        native_output = op(
            q, k, v, g_cmp, g_slc, g_swa, offsets
        )
        metadata = op._build_metadata(offsets)
        precomputed_output = op(
            q, k, v, g_cmp, g_slc, g_swa, offsets, *metadata
        )
    torch.cuda.synchronize()

    chunk_offsets, chunk_indices, token_indices, block_counts = metadata

    assert tuple(native_output.shape) == (total_tokens, heads, dim)
    assert native_output.dtype == dtype
    assert torch.isfinite(native_output).all()
    difference = (
        native_output.float() - precomputed_output.float()
    ).abs()
    relative_l2 = (
        difference.norm()
        / precomputed_output.float().norm().clamp_min(1e-12)
    )

    print("native_precomputed_max_abs_err:", difference.max().item())
    print("native_precomputed_relative_l2_err:", relative_l2.item())

    torch.testing.assert_close(
        native_output,
        precomputed_output,
        atol=5e-4,
        rtol=1e-3,
    )

    assert chunk_offsets.tolist() == [0, total_tokens // block_size]
    assert tuple(chunk_indices.shape) == (total_tokens // block_size, 2)
    assert tuple(token_indices.shape) == (total_tokens, 2)
    assert tuple(block_counts.shape) == (total_tokens, heads_kv)
    assert block_counts.min().item() == 1
    assert block_counts.max().item() == total_tokens // block_size

    for name, tensor in named_inputs.items():
        assert torch.equal(tensor, snapshots[name]), f"{name} was modified"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
