import torch
import torch.nn.functional as F
from gla_hip import chunk_gla_hip,load_extension
from gla_hygon_reference import chunk_gla_hygon

def rel(a,b):
    d=a.float()-b.float()
    return (d.square().mean().sqrt()/(a.float().square().mean().sqrt()+1e-8)).item()

assert getattr(torch.version,"hip",None) and torch.cuda.device_count()==1
torch.cuda.set_device(0); load_extension(False)
for dt in (torch.float16,torch.bfloat16):
  for K in (64,128,256,512):
    torch.manual_seed(1000+K)
    B,T,H,V=1,13,2,32
    kw=dict(device="cuda:0",dtype=dt)
    q=torch.randn(B,T,H,K,**kw)/K**.5; k=torch.randn(B,T,H,K,**kw)/K**.5
    v=torch.randn(B,T,H,V,**kw); g=F.logsigmoid(torch.randn(B,T,H,K,**kw))
    h0=torch.randn(B,H,K,V,device="cuda:0",dtype=torch.float32)
    r,rh=chunk_gla_hygon(q,k,v,g,initial_state=h0,output_final_state=True)
    with torch.no_grad(): y,yh=chunk_gla_hip(q,k,v,g,initial_state=h0,output_final_state=True)
    oe,he=rel(r,y),rel(rh,yh); print(dt,"K",K,"output",oe,"state",he)
    assert oe<.01 and he<.01
print("GLA HIP v1 forward correctness: PASS")
