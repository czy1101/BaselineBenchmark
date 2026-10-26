from pathlib import Path
from typing import Optional
import torch
from torch.utils.cpp_extension import load

_EXT = None

@torch.no_grad()
def _prefill_chunked(q, table, cache, lengths, max_pad, block_size,
                     h_q, h_kv, d, dv, causal, qtile=64, ktile=256):
    """DTK GEMM-backed prefill path with online softmax over KV tiles."""
    B, SQ = q.shape[:2]
    ids = table[:, : max_pad // block_size].long()
    flat = cache.index_select(0, ids.reshape(-1)).view(B, -1, h_kv, d)
    kv = flat[:, :, :, :].float().transpose(1, 2)
    # MLA's common decode/prefill case has Hkv=1.  Expand without materializing
    # 64/128 copies of the same K/V tensor; only GQA inputs need replication.
    if h_kv == 1:
        kv = kv.expand(B, h_q, kv.shape[2], kv.shape[3])
    else:
        kv = kv.repeat_interleave(h_q // h_kv, dim=1)
    k = kv[..., :d]
    v = kv[..., :dv]
    qf = q.float().transpose(1, 2)
    out = torch.empty(B, h_q, SQ, dv, device=q.device, dtype=torch.float32)
    lse_out = torch.empty(B, h_q, SQ, device=q.device, dtype=torch.float32)
    scale = d ** -0.5
    for qs in range(0, SQ, qtile):
        qe = min(qs + qtile, SQ)
        qq = qf[:, :, qs:qe]
        m = torch.full((B, h_q, qe-qs), -float('inf'), device=q.device)
        z = torch.zeros((B, h_q, qe-qs), device=q.device)
        acc = torch.zeros((B, h_q, qe-qs, dv), device=q.device)
        for ks in range(0, max_pad, ktile):
            ke = min(ks + ktile, max_pad)
            scores = torch.matmul(qq, k[:, :, ks:ke].transpose(-1, -2)) * scale
            valid = torch.arange(ks, ke, device=q.device)[None, None, None, :] < lengths[:, None, None, None]
            if causal:
                qi = torch.arange(qs, qe, device=q.device)[None, None, :, None]
                ki = torch.arange(ks, ke, device=q.device)[None, None, None, :]
                valid = valid & (ki <= lengths[:, None, None, None] - SQ + qi)
            scores = scores.masked_fill(~valid, -float('inf'))
            nm = torch.maximum(m.unsqueeze(-1), scores.max(-1, keepdim=True).values).squeeze(-1)
            a = torch.exp(m - nm)
            w = torch.exp(scores - nm.unsqueeze(-1))
            z = z * a + w.sum(-1)
            acc = acc * a.unsqueeze(-1) + torch.matmul(w, v[:, :, ks:ke])
            m = nm
        out[:, :, qs:qe] = acc / z.unsqueeze(-1)
        lse_out[:, :, qs:qe] = torch.log(z) + m
    return out.transpose(1, 2).to(q.dtype), lse_out

@torch.no_grad()
def _prefill_head_grouped(q, table, cache, lengths, max_pad, block_size,
                          h_q, h_kv, d, dv, causal, head_tile=16):
    """Run prefill in small head groups to bound GEMM workspace."""
    out = torch.empty(q.shape[0], q.shape[1], h_q, dv,
                      device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[0], h_q, q.shape[1],
                      device=q.device, dtype=torch.float32)
    for hs in range(0, h_q, head_tile):
        he = min(hs + head_tile, h_q)
        og, lg = _prefill_chunked(
            q[:, :, hs:he], table, cache, lengths, max_pad, block_size,
            he - hs, h_kv, d, dv, causal)
        out[:, :, hs:he] = og
        lse[:, hs:he] = lg
    return out, lse

def load_extension(verbose=False):
    global _EXT
    if _EXT is None:
        _EXT = load(name="mla_hygon_hip_ext_v14_qreg", sources=[str(Path(__file__).with_name("mla_hip_kernel.cu"))],
                    extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-ffast-math"], verbose=verbose)
    return _EXT

@torch.no_grad()
def flash_mla_hygon(q, block_table, blocked_k, max_seqlen_pad, block_size,
                    cache_seqlens, h_q, h_kv, d, dv, causal=True):
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("only float16/bfloat16 are supported")
    if q.shape[1] > 1:
        # Process all query heads in one tiled GEMM stream. Splitting Hq into
        # groups repeats the complete KV traversal for every group and causes
        # a severe prefill regression for Hq=128; the tile sizes already bound
        # score/accumulator workspace.
        return _prefill_chunked(q, block_table, blocked_k, cache_seqlens,
                                max_seqlen_pad, block_size, h_q, h_kv, d, dv, causal)
    ext = load_extension()
    return ext.forward(q.contiguous(), block_table.contiguous(), blocked_k.contiguous(),
                       cache_seqlens.contiguous(), block_table.shape[1], block_size,
                       h_kv, dv, causal)
