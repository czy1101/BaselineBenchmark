"""Hygon GLA v1: fused HIP inference with full semantic fallback."""
from pathlib import Path
from typing import Optional
import torch
from torch.utils.cpp_extension import load
from gla_hygon_reference import chunk_gla_hygon

_EXT=None
def load_extension(verbose=False):
    global _EXT
    if _EXT is None:
        _EXT=load(name="gla_hygon_hip_ext_v5_chunk_prepare",
                  sources=[str(Path(__file__).with_name("gla_hip_kernel.cu"))],
                  extra_cflags=["-O3"],extra_cuda_cflags=["-O3","-ffast-math"],
                  verbose=verbose)
    return _EXT

def _fast_ok(q,k,v,g,initial_state,state_v_first,cu_seqlens):
    return (not any(x.requires_grad for x in (q,k,v,g))
            and q.dtype in (torch.float16,torch.bfloat16)
            and q.shape==k.shape==g.shape and q.shape[:3]==v.shape[:3]
            and q.shape[-1] in (64,128,256,512)
            and not state_v_first and cu_seqlens is None
            and (initial_state is None or initial_state.dtype==torch.float32))

def chunk_gla_hip(q,k,v,g,scale:Optional[float]=None,initial_state=None,
                  output_final_state=False,state_v_first=False,cu_seqlens=None,
                  cu_seqlens_cpu=None):
    scale=q.shape[-1]**-0.5 if scale is None else float(scale)
    if _fast_ok(q,k,v,g,initial_state,state_v_first,cu_seqlens):
        o,ht=load_extension(False).forward(q.contiguous(),k.contiguous(),v.contiguous(),
                                           g.contiguous(),initial_state,scale)
        return o, ht if output_final_state else None
    return chunk_gla_hygon(q,k,v,g,scale=scale,initial_state=initial_state,
                           output_final_state=output_final_state,
                           state_v_first=state_v_first,cu_seqlens=cu_seqlens,
                           cu_seqlens_cpu=cu_seqlens_cpu)

chunk_gla=chunk_gla_hip
