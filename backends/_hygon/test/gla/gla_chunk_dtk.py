"""Hygon GLA hybrid dispatcher: D512 Chunk/WY, otherwise shared-K HIP."""
from __future__ import annotations
from typing import Optional
import torch
from gla_hip import chunk_gla_hip, load_extension

_CAUSAL_MASK = {}

def _causal_mask(device):
    key = (device.type, device.index)
    mask = _CAUSAL_MASK.get(key)
    if mask is None:
        mask = torch.tril(torch.ones(64, 64, device=device, dtype=torch.bool))
        _CAUSAL_MASK[key] = mask
    return mask


def _eligible(q,k,v,g,state_v_first,cu_seqlens):
    return (q.shape==k.shape==g.shape and q.shape[:3]==v.shape[:3]
            and q.dtype in (torch.float16,torch.bfloat16)
            # Measured crossover on BW: the DTK Chunk/WY path only wins for
            # D512. D128/D256 stay on the v4 shared-K HIP fallback.
            and q.shape[-1]==512 and v.shape[-1]==512
            and not state_v_first and cu_seqlens is None
            and not any(x.requires_grad for x in (q,k,v,g)))


@torch.no_grad()
def chunk_gla_dtk(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,g:torch.Tensor,
                  scale:Optional[float]=None,initial_state=None,
                  output_final_state=False,state_v_first=False,cu_seqlens=None,
                  cu_seqlens_cpu=None,chunk_size:int=64):
    if not _eligible(q,k,v,g,state_v_first,cu_seqlens) or chunk_size!=64:
        return chunk_gla_hip(q,k,v,g,scale=scale,initial_state=initial_state,
                             output_final_state=output_final_state,
                             state_v_first=state_v_first,cu_seqlens=cu_seqlens,
                             cu_seqlens_cpu=cu_seqlens_cpu)
    B,T,H,D=q.shape; scale=D**-0.5 if scale is None else float(scale)
    NT=(T+63)//64; TP=NT*64; BH=B*H
    # One HIP launch fuses BTHD->BHNT64D packing, FP32 gate cumsum, both
    # exponentials, Q/K scaling, V casting, padding, and end-decay extraction.
    ext=load_extension(False)
    qg,kg,vc_work,end_decay=ext.chunk_prepare(
        q.contiguous(),k.contiguous(),v.contiguous(),g.contiguous())
    # BF16 keeps FP32-like exponent range for the subsequent DTK GEMMs.
    work_dtype=torch.bfloat16
    scores=torch.matmul(qg,kg.transpose(-1,-2))*scale
    scores=scores.masked_fill(~_causal_mask(q.device),0)
    local=torch.matmul(scores,vc_work)
    state=(torch.zeros(BH,D,D,device=q.device,dtype=torch.float32)
           if initial_state is None else initial_state.reshape(BH,D,D).float())
    outs=[]
    # Window the KxV chunk injections instead of retaining all NT matrices.
    # Sixteen chunks amortizes DTK GEMM launch overhead while keeping the
    # D512 workspace bounded on the masked Hygon device.
    WINDOW=16
    for ws in range(0,NT,WINDOW):
        we=min(ws+WINDOW,NT)
        raw_B=torch.matmul(kg[:,ws:we].transpose(-1,-2),vc_work[:,ws:we]).float()
        Bwin=raw_B*end_decay[:,ws:we].unsqueeze(-1)
        for ci in range(ws,we):
            boundary=torch.matmul(qg[:,ci],state.to(work_dtype)).float()*scale
            oc=local[:,ci].float()+boundary
            outs.append(oc)
            state=state*end_decay[:,ci].unsqueeze(-1)+Bwin[:,ci-ws]
    out=torch.stack(outs,dim=1).reshape(B,H,NT,64,D).permute(0,2,3,1,4)
    out=out.reshape(B,TP,H,D)[:,:T].to(v.dtype)
    ht=state.reshape(B,H,D,D)
    return out, ht if output_final_state else None

chunk_gla=chunk_gla_dtk
