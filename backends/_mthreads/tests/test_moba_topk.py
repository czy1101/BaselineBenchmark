"""Independent correctness checks for K=1, K=2, and K=64.

Run in lgh_flagtree with MUSA_VISIBLE_DEVICES, MATE_WORKSPACE_BASE and project
cache paths configured. Correctness remains separate from the performance-only benchmark.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import pytest
import torch
import torch_musa


@lru_cache(None)
def implementation(provider):
    workspace = Path(__file__).resolve().parents[4]
    path = (workspace / "src/flag_attn/runtime/backend/_mthreads/FLA/moba" if provider == "triton"
            else workspace / "BaselineBenchmark/backends/_mthreads/ops/moba")
    name = "_topk1_" + provider
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                               submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.parallel_moba


@pytest.mark.parametrize("provider", ["triton", "mate"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [False, True])
def test_topk1_chunk_local_not_full_sequence(provider, dtype, training):
    lengths, chunk = (192, 129), 64
    total = sum(lengths)
    q = torch.zeros(total, 4, 64, device="musa", dtype=dtype, requires_grad=training)
    k = torch.zeros(total, 2, 64, device="musa", dtype=dtype, requires_grad=training)
    v = torch.empty_like(k, requires_grad=False)
    expected = torch.empty_like(q, requires_grad=False)
    full_sequence = torch.empty_like(q, requires_grad=False)
    start = 0
    for batch, length in enumerate(lengths):
        for pos in range(length):
            # Constant within each chunk, different across chunks and sequences.
            value = (pos // chunk + 1) * (1 if batch == 0 else -1)
            v[start + pos].fill_(value)
            expected[start + pos].fill_(value)
        for pos in range(length):
            full_sequence[start + pos] = v[start:start + pos + 1].float().mean(0).repeat_interleave(2, dim=0).to(dtype)
        start += length
    v.requires_grad_(training)
    cu = torch.tensor([0, lengths[0], total], device="musa", dtype=torch.int32)
    with torch.set_grad_enabled(training):
        actual = implementation(provider)(q, k, v, cu, max(lengths), chunk, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual - full_sequence).abs().max().item() > 0.5
    assert actual.requires_grad == training


@pytest.mark.parametrize("provider", ["triton", "mate"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("lengths,topk", [
    pytest.param((257, 193), 2, id="k2-varlen-gqa"),
    pytest.param((257, 193), 64, id="k64-saturated"),
    pytest.param((4161,), 64, id="k64-more-than64-chunks"),
])
def test_topk_edges_against_sampled_float64(provider, dtype, training, lengths, topk):
    # Well-separated scores avoid unspecified tie-breaking. Opposite query heads
    # sharing KV must independently select opposite ends of the historical list.
    chunk, hq, hkv, dim = 64, 4, 2, 64
    total = sum(lengths)
    qc = torch.full((total, hq, dim), 0.125, dtype=dtype)
    qc[:, 1::2] *= -1
    kc = torch.empty(total, hkv, dim, dtype=dtype)
    start = 0
    for length in lengths:
        for pos in range(length):
            kc[start + pos].fill_((pos // chunk + 1) / 128)
        start += length
    generator = torch.Generator().manual_seed(42)
    vc = torch.randn(total, hkv, dim, generator=generator, dtype=torch.float32).to(dtype)
    q, k, v = [x.to("musa").requires_grad_(training) for x in (qc, kc, vc)]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu = torch.tensor(boundaries, device="musa", dtype=torch.int32)
    with torch.set_grad_enabled(training):
        actual = implementation(provider)(q, k, v, cu, max(lengths), chunk, topk)
    actual = actual.detach().cpu().double()
    errors = []
    for batch, length in enumerate(lengths):
        start = boundaries[batch]
        keys, values = kc[start:start + length], vc[start:start + length].double()
        means = torch.stack([keys[t:t + chunk].double().mean(0).to(dtype).double()
                             for t in range(0, length, chunk)])
        # Boundaries, first/last tokens and a deterministic spread of interior rows.
        positions = {0, length - 1}
        positions.update(range(0, length, max(1, length // 32)))
        for edge in range(chunk, length, chunk):
            positions.update((edge - 1, edge))
        for pos in sorted(positions):
            current = pos // chunk
            for head in range(hq):
                kvhead = head // (hq // hkv)
                query = qc[start + pos, head].double()
                scores = means[:current, kvhead] @ query
                selected = torch.topk(scores, min(topk - 1, current)).indices.tolist()
                indices = [i for block in selected for i in range(block * chunk, (block + 1) * chunk)]
                indices += list(range(current * chunk, pos + 1))
                logits = (keys[indices, kvhead].double() @ query) * (dim ** -0.5)
                expected = logits.softmax(0) @ values[indices, kvhead]
                errors.append((actual[start + pos, head] - expected).abs())
    diff = torch.stack(errors)
    assert torch.isfinite(diff).all()
    # Same max/mean limits as the existing project correctness tests.
    assert diff.max().item() < 0.04
    assert diff.mean().item() < 4e-4
