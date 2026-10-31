"""Per-query-head MoBA route kernel using validated MP31 MATE collectives.

No replacement of the MoBA router, no shared GQA routing, no host route counts.
MATE headers and templates remain unmodified in their installation.
"""
import hashlib, functools, importlib.util, importlib.metadata
from pathlib import Path
import torch
import triton
import triton.language as tl

@functools.lru_cache(None)
def _compatible_mate():
    # Private collective/template API: enable only the version validated here.
    if importlib.util.find_spec("mate") is None:
        return False
    try:
        return importlib.metadata.version("mate") == "0.2.0+mu437torch2.7"
    except importlib.metadata.PackageNotFoundError:
        return False


@functools.lru_cache(None)
def _is_mp31(device):
    properties = torch.musa.get_device_properties(device)
    return (properties.major, properties.minor) == (3, 1)


def _supported(q):
    return (q.device.type == "musa" and q.shape[-1] in (64, 128)
            and q.dtype in (torch.float16, torch.bfloat16)
            and _is_mp31(q.device) and _compatible_mate())


SCHEDULER=r'''
// One descriptor identifies one query tile of exactly one (chunk, query-head).
struct MoBARouteScheduler {
  static constexpr bool HasMetadata = false;
  struct Arguments { int count; int const* experts; int const* starts; int const* offsets; };
  using Params = Arguments;
  static Params to_underlying_arguments(Arguments const& a) { return a; }
  static dim3 get_grid_shape(Params const& p, int) { return dim3(p.count,1,1); }
  struct WorkTileInfo {
    int expert; int tile;
    MUTLASS_DEVICE bool is_valid(Params const&) const { return expert>=0; }
    MUTLASS_DEVICE mute::tuple<int,int,int,int> get_block_coord(Params const&) const {
      return {tile,0,expert,0};
    }
  };
  MUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& p) const {
    int e=p.experts[blockIdx.x];
    if(e<0) return {-1,0};
    return {e,(p.starts[blockIdx.x]-p.offsets[e])/32};
  }
  MUTLASS_DEVICE WorkTileInfo get_next_work(Params const&, WorkTileInfo const&) const { return {-1,0}; }
};
'''
@functools.lru_cache(None)
def kernel(dtype,d):
    from mate.jit.core import gen_jit_spec
    from mate.jit.attention.fmha.fmha_fwd import _render_fmha_fwd_source
    from mate.jit.attention.fmha.fmha_utils import FMHA_EXTRA_CUDA_CFLAGS,FMHA_EXTRA_LDFLAGS,fmha_extra_include_paths
    from mate.jit import env as jit_env
    cfg=dict(has_cu_seqlens_q=True,has_cu_seqlens_k=True,has_seqused_q=False,
      has_seqused_k=False,has_cu_seqlens_k_new=False,has_leftpad_k=False,paged_kv=False,
      has_softcap=False,has_learnable_sink=False,is_causal=False,is_local=False,
      head_ratio=1,force_lsu_kv=False,tile_m=32,tile_n=64,head_dim_qk=d,head_dim_vo=d,
      stages_k=1,stages_v=1,consumers_qk=1,consumers_pv=1,is_packgqa=False,
      is_even_headdim=True,has_metadata=False,enable_cp=False,
      element='mutlass::bfloat16_t' if dtype==torch.bfloat16 else 'mutlass::half_t',func_name='moba_routes')
    source=_render_fmha_fwd_source(cfg)
    source=source.replace('ffi::Array<ffi::Any> moba_routes',SCHEDULER+'\nffi::Array<ffi::Any> moba_routes',1)
    source=source.replace('ffi::TensorView q,','ffi::TensorView descriptor_experts,\n  ffi::TensorView descriptor_starts,\n  ffi::TensorView q,',1)
    begin=source.index('  using TileScheduler = std::conditional_t')
    end=source.index('  using Kernel =',begin)
    source=source[:begin]+'  using TileScheduler = MoBARouteScheduler;\n\n'+source[end:]
    begin=source.index('  typename TileScheduler::Arguments scheduler_args')
    end=source.index('  typename Kernel::Arguments kernel_arguments',begin)
    source=source[:begin]+'''  typename TileScheduler::Arguments scheduler_args {
      static_cast<int>(descriptor_experts.numel()),
      static_cast<int const*>(descriptor_experts.data_ptr()),
      static_cast<int const*>(descriptor_starts.data_ptr()),
      static_cast<int const*>(cu_seqlens_q.value().data_ptr())};
\n'''+source[end:]
    # This no-split specialization never reads partial outputs: avoid hidden allocations.
    begin=source.index('  ffi::Tensor o_accum')
    end=source.index('  // Set Mainloop Args',begin)
    source=source[:begin]+source[end:]
    source=source.replace('return ffi::Array<ffi::Any>{ffi::Array<ffi::Tensor>{o_accum, lse_accum}, params.num_splits};','return ffi::Array<ffi::Any>{};')
    name='moba_route_'+hashlib.sha256(source.encode()).hexdigest()[:16]
    root=jit_env.MATE_GEN_SRC_DIR / "moba"
    path=root/(name+'.mu')
    spec=gen_jit_spec(name=name,sources=[path],generated_sources={path:source},
      extra_cuda_cflags=list(FMHA_EXTRA_CUDA_CFLAGS),extra_ldflags=list(FMHA_EXTRA_LDFLAGS),
      extra_include_paths=fmha_extra_include_paths())
    return spec.build_and_load().get_function('moba_routes')

@triton.jit
def pack_q(Q,I,OFF,O,N:tl.constexpr,E:tl.constexpr,H:tl.constexpr,D:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,B:tl.constexpr):
    rows=tl.program_id(0)*16+tl.arange(0,16)
    dims=tl.arange(0,B)
    count=tl.load(OFF+E)
    indices=tl.load(I+rows,rows<tl.minimum(count,N),other=0)
    vals=tl.load(Q+(indices//H)[:,None]*S0+(indices%H)[:,None]*S1+dims[None,:]*S2,
                 (rows<tl.minimum(count,N))[:,None]&(dims<D)[None,:],other=0)
    tl.store(O+rows[:,None]*D+dims[None,:],vals,(rows<N)[:,None]&(dims<D)[None,:])
@triton.jit
def pack_kv(K,V,CU,KO,VO,H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,C:tl.constexpr,MC:tl.constexpr,
            K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0);row=tl.program_id(1)*16+tl.arange(0,16);dim=tl.arange(0,B)
    chunk=e//H;head=(e%H)//G;batch=chunk//MC;start=tl.load(CU+batch);end=tl.load(CU+batch+1)
    token=start+(chunk%MC)*C+row
    mask=(row<C)[:,None]&(token<end)[:,None]&(dim<D)[None,:]
    kval=tl.load(K+token[:,None]*K0+head*K1+dim[None,:]*K2,mask,other=0)
    vval=tl.load(V+token[:,None]*V0+head*V1+dim[None,:]*V2,mask,other=0)
    offset=(e*C+row[:,None])*D+dim[None,:]
    storemask=(row<C)[:,None]&(dim<D)[None,:]
    tl.store(KO+offset,kval,storemask);tl.store(VO+offset,vval,storemask)

def routed(q,k,v,indices,offsets,experts,starts,cu,max_chunks,chunk_size):
    d=q.shape[-1];h=q.shape[1];e=offsets.numel()-1;n=indices.numel()
    if not _supported(q):
        from .attention import _routed_sparse_attention_triton
        return _routed_sparse_attention_triton(q,k,v,indices,offsets,experts,starts,cu,max_chunks,chunk_size)
    fn=kernel(q.dtype,d)
    qp=torch.empty((n,1,d),device=q.device,dtype=q.dtype)
    kp=torch.empty((e*chunk_size,1,d),device=q.device,dtype=q.dtype);vp=torch.empty_like(kp)
    pack_q[(triton.cdiv(n,16),)](q,indices,offsets,qp,n,e,h,d,*q.stride(),triton.next_power_of_2(d))
    pack_kv[(e,triton.cdiv(chunk_size,16))](k,v,cu,kp,vp,h,h//k.shape[1],d,chunk_size,max_chunks,*k.stride(),*v.stride(),triton.next_power_of_2(d))
    cuk=torch.arange(e+1,device=q.device,dtype=torch.int32)*chunk_size
    out=torch.empty_like(qp);lse=torch.empty((1,n),device=q.device,dtype=torch.float32)
    fn(experts,starts,qp,kp,vp,None,None,None,offsets,cuk,None,None,None,q.shape[0],chunk_size,
       None,None,None,None,None,None,None,None,None,d**-.5,False,-1,-1,0,0.0,True,0,1,
       None,None,None,None,out,lse,1,0,None)
    return out.view(n,d),lse.view(n)

@triton.jit
def descriptors(CU,EX,ST,E:tl.constexpr,B:tl.constexpr):
    e=tl.program_id(0);first=tl.load(CU+e);last=tl.load(CU+e+1)
    i=tl.arange(0,B);count=tl.cdiv(last-first,32);slot=first//32+e+i
    tl.store(EX+slot,e,i<count);tl.store(ST+slot,first+32*i,i<count)

def varlen(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,softmax_scale=None,causal=False):
    from .attention import _triton_varlen_forward
    if causal or q.shape[1]!=1 or not _supported(q):
        return _triton_varlen_forward(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,softmax_scale,causal)
    d=q.shape[-1];n=q.shape[0];e=cu_seqlens_q.numel()-1
    ex=torch.full((triton.cdiv(n,32)+e,),-1,device=q.device,dtype=torch.int32);st=torch.empty_like(ex)
    descriptors[(e,)](cu_seqlens_q,ex,st,e,triton.next_power_of_2(max(1,triton.cdiv(max_seqlen_q,32))))
    out=torch.empty_like(q);lse=torch.empty((1,n),device=q.device,dtype=torch.float32)
    fn=kernel(q.dtype,d)
    fn(ex,st,q.contiguous(),k.contiguous(),v.contiguous(),None,None,None,cu_seqlens_q,cu_seqlens_k,None,None,None,max_seqlen_q,max_seqlen_k,
       None,None,None,None,None,None,None,None,None,softmax_scale if softmax_scale is not None else d**-.5,False,-1,-1,0,0.0,True,0,1,
       None,None,None,None,out,lse,1,0,None)
    return out,lse
