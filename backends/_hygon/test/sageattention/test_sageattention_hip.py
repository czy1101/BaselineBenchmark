import math

import torch

from sageattention_hip import backend_name, forward, per_block_int8


def expand_scale(scale, block, length):
    return scale.repeat_interleave(block, dim=-1)[..., :length, None]


def reference(q, k, v, qs, ks, layout, mask=None):
    if layout == "NHD":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    q = q.float() * expand_scale(qs, 128, q.shape[-2])
    k = k.float() * expand_scale(ks, 64, k.shape[-2])
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2))
    if mask is not None:
        logits = logits.masked_fill(~mask, -float("inf")) if mask.dtype == torch.bool else logits + mask
    probs = torch.softmax(logits * math.log(2.0), dim=-1)
    out = torch.matmul(probs, v.float())
    lse = torch.logsumexp(logits * math.log(2.0), dim=-1) / math.log(2.0)
    if layout == "NHD":
        out = out.transpose(1, 2)
    return out, lse


def run_case(layout, kv_heads, dim, q_len, kv_len, mask_kind=None):
    torch.manual_seed(2026 + dim + q_len + kv_len)
    q_heads = 4
    q = torch.randn((1, q_heads, q_len, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((1, kv_heads, kv_len, dim), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    mask = None
    if mask_kind == "bool":
        mask = torch.ones((1, q_heads, q_len, kv_len), device="cuda", dtype=torch.bool)
        mask[..., ::3] = False
    elif mask_kind == "additive":
        mask = torch.zeros((1, q_heads, q_len, kv_len), device="cuda")
        mask[..., ::3] = -2.0
    qi, qs, ki, ks = per_block_int8(q, k, tensor_layout=layout)
    out, lse = forward(
        qi, ki, v, qs, ks, tensor_layout=layout, attn_mask=mask,
        return_lse=True,
    )
    ref, ref_lse = reference(qi, ki, v, qs, ks, layout, mask)
    out_err = ((out.float() - ref).square().mean().sqrt() /
               (ref.square().mean().sqrt() + 1e-6)).item()
    lse_err = (lse - ref_lse).abs().max().item()
    print(layout, "kv_heads", kv_heads, "D", dim, "out_rel_rmse", out_err,
          "lse_max_abs", lse_err)
    assert torch.isfinite(out).all() and out_err < 3e-2
    assert torch.isfinite(lse).all() and lse_err < 3e-2


if __name__ == "__main__":
    print("SageAttention backend:", backend_name())
    run_case("HND", 1, 64, 128, 128)
    run_case("NHD", 2, 64, 128, 128)
    run_case("HND", 2, 128, 129, 70, "bool")
    run_case("HND", 2, 128, 129, 70, "additive")
    print("SageAttention Hygon correctness: PASS")
