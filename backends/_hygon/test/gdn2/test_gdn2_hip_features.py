import torch

from gdn2_hip import chunk_gdn2_hip, load_extension
from gdn2_hygon_reference import chunk_gdn2_hygon


def close(ref, got, name):
    d = got.float() - ref.float()
    rel = (d.square().mean().sqrt() /
           (ref.float().square().mean().sqrt() + 1e-8)).item()
    print(name, "relative_rmse=", rel)
    assert rel < 0.01


def inputs(B=1,T=16,H=2,K=64,V=32,dtype=torch.bfloat16):
    torch.manual_seed(123)
    kw = dict(device="cuda:0",dtype=dtype)
    return (
        torch.randn(B,T,H,K,**kw)/K**.5,
        torch.randn(B,T,H,K,**kw)/K**.5,
        torch.randn(B,T,H,V,**kw),
        torch.randn(B,T,H,K,**kw),
        torch.rand(B,T,H,K,**kw),
        torch.rand(B,T,H,V,**kw),
    )


load_extension(verbose=False)

# Raw gate + bias.
x = inputs()
A_log = torch.randn(2,device="cuda:0")
dt_bias = torch.randn(2*64,device="cuda:0")
kw = dict(A_log=A_log,dt_bias=dt_bias,use_gate_in_kernel=True,
          output_final_state=True,chunk_size=16)
r, rh = chunk_gdn2_hygon(*x,**kw)
y, yh = chunk_gdn2_hip(*x,**kw)
close(r,y,"raw_gate/output"); close(rh,yh,"raw_gate/state")

# Safe gate and Q/K normalization.
kw.update(safe_gate=True,lower_bound=-5.0,use_qk_l2norm_in_kernel=True)
r, rh = chunk_gdn2_hygon(*x,**kw)
y, yh = chunk_gdn2_hip(*x,**kw)
close(r,y,"safe_l2/output"); close(rh,yh,"safe_l2/state")

# V-first initial/final state.
x = inputs(V=16)
h0 = torch.randn(1,2,16,64,device="cuda:0",dtype=torch.float32)
kw = dict(initial_state=h0,state_v_first=True,output_final_state=True)
r, rh = chunk_gdn2_hygon(*x,**kw)
y, yh = chunk_gdn2_hip(*x,**kw)
close(r,y,"v_first/output"); close(rh,yh,"v_first/state")

# Packed variable-length sequences and chunk-start intermediates.
x = inputs(B=1,T=10,V=16)
cu = torch.tensor([0,4,10],device="cuda:0",dtype=torch.int64)
h0 = torch.randn(2,2,64,16,device="cuda:0",dtype=torch.float32)
kw = dict(initial_state=h0,cu_seqlens=cu,output_final_state=True,chunk_size=4)
r, rh = chunk_gdn2_hygon(*x,**kw)
y, yh, hs = chunk_gdn2_hip(*x,**kw,return_intermediate_states=True)
close(r,y,"varlen/output"); close(rh,yh,"varlen/state")
assert hs.shape == (2,2,2,64,16)
print("intermediate shape:",tuple(hs.shape))
print("GDN2 HIP public feature tests: PASS")
