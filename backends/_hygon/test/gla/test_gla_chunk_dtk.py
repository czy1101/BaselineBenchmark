import torch
import torch.nn.functional as F
from gla_chunk_dtk import chunk_gla_dtk
from gla_hygon_reference import chunk_gla_hygon

def rel(a,b):
 d=a.float()-b.float(); return (d.square().mean().sqrt()/(a.float().square().mean().sqrt()+1e-8)).item()
assert getattr(torch.version,"hip",None) and torch.cuda.device_count()==1
torch.cuda.set_device(0)
for dt in (torch.float16,torch.bfloat16):
 for D in (128,256,512):
  for T in (13,64,130):
   torch.manual_seed(D+T); kw=dict(device="cuda:0",dtype=dt); B,H=1,2
   q=torch.randn(B,T,H,D,**kw)/D**.5; k=torch.randn(B,T,H,D,**kw)/D**.5
   v=torch.randn(B,T,H,D,**kw); g=F.logsigmoid(torch.randn(B,T,H,D,**kw))*.1
   h0=torch.randn(B,H,D,D,device="cuda:0",dtype=torch.float32)*.01
   r,rh=chunk_gla_hygon(q,k,v,g,initial_state=h0,output_final_state=True)
   y,yh=chunk_gla_dtk(q,k,v,g,initial_state=h0,output_final_state=True)
   oe,he=rel(r,y),rel(rh,yh); print(dt,"D",D,"T",T,"output",oe,"state",he)
   assert oe<.01 and he<.01
print("GLA Hygon BT64 Chunk/WY correctness: PASS")
