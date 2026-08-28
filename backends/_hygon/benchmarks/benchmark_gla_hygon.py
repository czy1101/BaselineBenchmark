"""Hygon GLA forward benchmark aligned with the supplied official table."""
from __future__ import annotations
import argparse, math, statistics, time
import torch
import torch.nn.functional as F
from gla_hip import chunk_gla_hip
from gla_chunk_dtk import chunk_gla_dtk
from gla_hygon_reference import chunk_gla_hygon

SHAPES=[
 (1,8192,96,128),
 (2,16384,16,128),
 (4,2048,16,128),
 (4,4096,64,128),
 (8,2048,32,256),
 (2,2048,16,512),
 (4,1024,8,512),
 (8,1024,8,64),
]
# (official GEMS CUDA, Hygon before TLE, Hygon after TLE), milliseconds.
TARGETS={
 (1,8192,96,128):(math.nan,9.090,8.811),
 (2,16384,16,128):(math.nan,6.754,6.549),
 (4,2048,16,128):(math.nan,1.276,1.352),
 (4,4096,64,128):(3.305,11.668,11.715),
 (8,2048,32,256):(4.358,14.525,14.329),
 (2,2048,16,512):(1.780,5.694,5.970),
 (4,1024,8,512):(math.nan,2.887,3.142),
 (8,1024,8,64):(math.nan,0.510,0.517),
}

def inputs(s,dt):
 B,T,H,D=s; kw=dict(device="cuda:0",dtype=dt)
 torch.manual_seed(2026+D)
 q=torch.randn(B,T,H,D,**kw)/math.sqrt(D)
 k=torch.randn(B,T,H,D,**kw)/math.sqrt(D)
 v=torch.randn(B,T,H,D,**kw)
 g=F.logsigmoid(torch.randn(B,T,H,D,**kw))
 return q,k,v,g

def bench(shape,dt,warmup,iters,impl):
 x=inputs(shape,dt)
 op=chunk_gla_hip if impl=="hip" else chunk_gla_dtk if impl=="chunk" else chunk_gla_hygon
 torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for _ in range(warmup): op(*x,output_final_state=False)
  torch.cuda.synchronize(); vals=[]
  for _ in range(iters):
   st=time.perf_counter(); o,_=op(*x,output_final_state=False)
   torch.cuda.synchronize(); vals.append((time.perf_counter()-st)*1000)
 p50=statistics.median(vals); cuda,pre,post=TARGETS[shape]
 return dict(B=shape[0],T=shape[1],H=shape[2],D=shape[3],dtype=str(dt).split('.')[-1],
  official_gems_ms=cuda,hy_tle_before_ms=pre,hy_tle_after_ms=post,
  current_hip_ms=p50,hy_tle_before_speedup=pre/p50,
  current_over_hy_tle_before=p50/pre,hy_tle_after_speedup=post/p50,
  current_over_hy_tle_after=p50/post,
  current_over_cuda=p50/cuda if math.isfinite(cuda) else math.nan,
  current_vs_cuda=f"slow {p50/cuda:.2f}x" if math.isfinite(cuda) else "—",
  mean_ms=statistics.mean(vals),min_ms=min(vals),
  peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
  output_nan=bool(torch.isnan(o.float()).any()))

def show(rows,impl):
 impl_name={"hip":"HIP shared-K ms","chunk":"HIP hybrid ms",
            "reference":"PyTorch reference ms"}[impl]
 print(f"B\tT\tH\tD\tOfficial GEMS ms\tHY TLE pre ms\tHY TLE post ms\t{impl_name}\tCurrent/HY TLE pre\tCurrent/HY TLE post\tCurrent/Cuda")
 for r in rows:
  official=(f"{r['official_gems_ms']:.3f}"
            if math.isfinite(r['official_gems_ms']) else "—")
  over_cuda=(f"{r['current_over_cuda']:.2f}x"
             if math.isfinite(r['current_over_cuda']) else "—")
  print(f"{r['B']}\t{r['T']}\t{r['H']}\t{r['D']}\t"
        f"{official}\t{r['hy_tle_before_ms']:.3f}\t{r['hy_tle_after_ms']:.3f}\t"
        f"{r['current_hip_ms']:.3f}\t{r['current_over_hy_tle_before']:.2f}x\t"
        f"{r['current_over_hy_tle_after']:.2f}x\t{over_cuda}")
 print("details:")
 for r in rows: print(r,flush=True)

ap=argparse.ArgumentParser()
ap.add_argument("--implementation",choices=("hip","chunk","reference"),default="hip")
ap.add_argument("--dtype",choices=("fp16","bf16","both"),default="bf16")
ap.add_argument("--shape",nargs=4,type=int,action="append")
ap.add_argument("--warmup",type=int,default=5); ap.add_argument("--iterations",type=int,default=20)
a=ap.parse_args()
assert getattr(torch.version,"hip",None),"Hygon HIP PyTorch required"
assert torch.cuda.is_available() and torch.cuda.device_count()==1,"exactly one masked HCU required"
torch.cuda.set_device(0)
shapes=[tuple(x) for x in a.shape] if a.shape else SHAPES
for s in shapes:
 if s not in TARGETS: raise ValueError(f"no supplied target for shape {s}")
dts={"fp16":[torch.float16],"bf16":[torch.bfloat16],"both":[torch.float16,torch.bfloat16]}[a.dtype]
rows=[bench(s,dt,a.warmup,a.iterations,a.implementation) for dt in dts for s in shapes]
show(rows,a.implementation)
