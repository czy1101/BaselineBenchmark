import torch

from gdn2_hip import chunk_gdn2_hip, load_extension
from gdn2_hygon_reference import chunk_gdn2_hygon


def stats(ref, got):
    d = got.float() - ref.float()
    return d.abs().max().item(), (
        d.square().mean().sqrt() / (ref.float().square().mean().sqrt() + 1e-8)
    ).item()


load_extension(verbose=True)
for dtype in (torch.float16, torch.bfloat16):
    torch.manual_seed(42)
    B,T,H,K,V = 1,32,2,64,64
    kw = dict(device="cuda:0", dtype=dtype)
    q = torch.randn(B,T,H,K,**kw) / K**0.5
    k = torch.randn(B,T,H,K,**kw) / K**0.5
    v = torch.randn(B,T,H,V,**kw)
    g = (-torch.rand(B,T,H,K,device="cuda:0") * .1).to(dtype)
    b = torch.rand(B,T,H,K,**kw)
    w = torch.rand(B,T,H,V,**kw)
    ref, ref_ht = chunk_gdn2_hygon(q,k,v,g,b,w,output_final_state=True)
    got, got_ht = chunk_gdn2_hip(q,k,v,g,b,w,output_final_state=True)
    torch.cuda.synchronize()
    o_max, o_rel = stats(ref, got)
    h_max, h_rel = stats(ref_ht, got_ht)
    print(dtype, {"o_max":o_max,"o_rel_rmse":o_rel,"h_max":h_max,"h_rel_rmse":h_rel})
    assert o_rel < 0.01 and h_rel < 0.01
print("GDN2 HIP correctness: PASS")
