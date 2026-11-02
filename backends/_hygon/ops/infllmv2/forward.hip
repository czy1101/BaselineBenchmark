// SPDX-License-Identifier: Apache-2.0
#include "common.hpp"
// SPDX-License-Identifier: Apache-2.0

template<int D,bool BF,int Window,int Stride>
__global__ void compress_kernel(const W* k,W* out,const int* meta,int which) {
    const int* m=meta+blockIdx.z*8;
    int c=blockIdx.x,hk=blockIdx.y,d=threadIdx.x;
    int start=which==1?m[C1S]:m[C2S],count=which==1?m[C1N]:m[C2N];
    if(c>=count) return;
    float sum=0;
    #pragma unroll
    for(int w=0;w<Window;++w) sum+=unpack<BF>(k[(int64_t(m[KS]+c*Stride+w)*2+hk)*D+d]);
    out[(int64_t(start+c)*2+hk)*D+d]=pack<BF>(sum/Window);
}
template<int D,bool BF>
__device__ __forceinline__ F4 qk(const B4 (&qfrag)[D/16],const W* k,int hk,int t,int start,int count) {
    int lane=threadIdx.x&63,row=lane&15,group=lane>>4;
    F4 c={0,0,0,0};
    #pragma unroll
    for(int kt=0;kt<D/16;++kt) {
        B4 b={0,0,0,0};
        if(t+row<count) b=*reinterpret_cast<const B4*>(k+(int64_t(start+t+row)*2+hk)*D+kt*16+group*4);
        c=mma<BF>(qfrag[kt],b,c);
    }
    return c;
}
template<int D,bool BF,int G>
__global__ __launch_bounds__(64)
void stage1_kernel(const W* q,const W* k1,const W* k2,W* score,const int* meta,
                   int totalq,int maxn1,float scale) {
    const int* m=meta+blockIdx.z*8;
    int qi=blockIdx.x,hk=blockIdx.y,lane=threadIdx.x,row=lane&15,group=lane>>4;
    if(qi>=m[QL]) return;
    int index=m[QS]+qi,n1=m[C1N],n2=m[C2N];
    B4 qfrag[D/16];
    #pragma unroll
    for(int kt=0;kt<D/16;++kt) {
        qfrag[kt]=B4{0,0,0,0};
        if(row<G) qfrag[kt]=*reinterpret_cast<const B4*>(q+(int64_t(index)*(2*G)+hk*G+row)*D+kt*16+group*4);
    }
    int stride=n1==n2?16:64;
    int cr=max(0,(qi+1)/stride-1+n2-max(0,m[QL]-stride+1)/stride);
    int fr=max(0,(qi+1)/16-1+n1-max(0,m[QL]-16+1)/16);
    float peak=-INFINITY,sum=0;
    for(int t=0;t<min(cr,n2);t+=16) {
        F4 c=qk<D,BF>(qfrag,k2,hk,t,m[C2S],n2);
        float tm=-INFINITY;
        #pragma unroll
        for(int r=0;r<4;++r) {c[r]=(t+group+4*r<min(cr,n2))?c[r]*scale:-INFINITY;tm=fmaxf(tm,c[r]);}
        tm=fmaxf(tm,__shfl_xor(tm,16,64));tm=fmaxf(tm,__shfl_xor(tm,32,64));
        float nm=fmaxf(peak,tm),a=peak==-INFINITY?0.f:__expf(peak-nm),s=0;
        #pragma unroll
        for(int r=0;r<4;++r) s+=c[r]==-INFINITY?0.f:__expf(c[r]-nm);
        s+=__shfl_xor(s,16,64);s+=__shfl_xor(s,32,64);
        sum=sum*a+s;peak=nm;
    }
    float lse=peak+logf(sum);
    for(int t=0;t<maxn1;t+=16) {
        F4 c=qk<D,BF>(qfrag,k1,hk,t,m[C1S],n1);
        #pragma unroll
        for(int r=0;r<4;++r) {
            int token=t+group+4*r;
            float p=(row<G && token<min(fr,n1) && sum>0)?__expf(c[r]*scale-lse):0.f;
            p+=__shfl_xor(p,1,64);p+=__shfl_xor(p,2,64);p+=__shfl_xor(p,4,64);p+=__shfl_xor(p,8,64);
            if(row==0 && token<maxn1) score[(int64_t(hk)*totalq+index)*maxn1+token]=pack<BF>(p);
        }
    }
}
template<bool BF>
__global__ void pool_kernel(const W* score,W* out,const int* meta,int totalq,int maxn1,int nb,int initial,int local) {
    const int* m=meta+blockIdx.z*8;
    int qi=blockIdx.x,hk=blockIdx.y;
    if(qi>=m[QL]) return;
    int index=m[QS]+qi,qb=(m[KL]-m[QL]+qi)/64;
    for(int b=threadIdx.x;b<nb;b+=blockDim.x) {
        float x=-INFINITY;
        if(b<initial || (b<=qb && qb<=b+local)) x=INFINITY;
        else {
            #pragma unroll
            for(int j=0;j<5;++j) {
                int f=b*4-1+j;
                if(f>=0 && f<m[C1N]) x=fmaxf(x,unpack<BF>(score[(int64_t(hk)*totalq+index)*maxn1+f]));
            }
        }
        out[(int64_t(hk)*totalq+index)*nb+b]=pack<BF>(x);
    }
}
template<int N,bool BF>
__global__ void topk_kernel(const W* score,int* out,const int* meta,int totalq,int nb,int topk) {
    __shared__ uint64_t keys[N];
    __shared__ int ids[64];
    const int* m=meta+blockIdx.z*8;
    int qi=blockIdx.x,hk=blockIdx.y,tid=threadIdx.x;
    if(qi>=m[QL]) return;
    int index=m[QS]+qi,valid=min(nb,(m[KL]-m[QL]+qi+64)/64);
    int64_t dst=(int64_t(hk)*totalq+index)*topk;
    if(valid<=topk) {if(tid<topk) out[dst+tid]=tid<valid?tid:-1;return;}
    for(int b=tid;b<N;b+=blockDim.x) {
        uint64_t key=0;
        if(b<valid) {
            float x=unpack<BF>(score[(int64_t(hk)*totalq+index)*nb+b]);if(isnan(x)) x=-INFINITY;
            uint32_t bits=__float_as_uint(x),ordered=bits^((bits&0x80000000U)?0xffffffffU:0x80000000U);
            key=(uint64_t(ordered)<<16)|uint64_t(0xffff-b);
        }
        keys[b]=key;
    }
    __syncthreads();
    #pragma unroll
    for(int k=2;k<=N;k<<=1) {
        #pragma unroll
        for(int j=k>>1;j;j>>=1) {
            for(int b=tid;b<N;b+=blockDim.x) {
                int other=b^j;
                if(other>b) {uint64_t a=keys[b],c=keys[other];bool desc=(b&k)==0;
                    if((desc&&a<c)||(!desc&&a>c)) {keys[b]=c;keys[other]=a;}}
            }
            __syncthreads();
        }
    }
    if(tid<topk) ids[tid]=0xffff-int(keys[tid]&0xffff);
    __syncthreads();
    if(tid<topk) {int b=ids[tid],rank=0;for(int j=0;j<topk;++j) rank+=ids[j]<b;out[dst+rank]=b;}
}

// SPDX-License-Identifier: Apache-2.0
// Generalizes the frozen wave64 MMAC Stage2 fragment mapping from native v1.
template<int D,bool BF,int G,bool Dense,bool Partial>
__global__ __launch_bounds__(64)
void attention_kernel(const W* q,const W* k,const W* v,const int* selected,W* out,float* lse,
                      const int* meta,int totalq,int topk,float scale,float* partial,int splits) {
    const int* m=meta+blockIdx.z*8;
    int qi=Partial?0:blockIdx.x,hk=blockIdx.y,lane=threadIdx.x,row=lane&15,group=lane>>4;
    if(qi>=m[QL]) return;
    int index=m[QS]+qi,limit=m[KL]-m[QL]+qi+1;
    int ranks=Dense?(limit+63)/64:min(topk,(limit+63)/64);
    int begin=0;
    if constexpr(Partial) {int chunk=(topk+splits-1)/splits;begin=blockIdx.x*chunk;ranks=min(ranks,begin+chunk);}
    B4 qfrag[D/16];F4 acc[D/16];
    #pragma unroll
    for(int t=0;t<D/16;++t) {
        qfrag[t]=B4{0,0,0,0};acc[t]=F4{0,0,0,0};
        if(row<G) qfrag[t]=*reinterpret_cast<const B4*>(q+(int64_t(index)*2*G+hk*G+row)*D+t*16+group*4);
    }
    float peak=-INFINITY,sum=0;
    for(int rank=begin;rank<ranks;++rank) {
        int b=Dense?rank:selected[(int64_t(hk)*totalq+index)*topk+rank];
        int base=max(0,b)*64;
        #pragma unroll
        for(int part=0;part<2;++part) {
            F4 logits[2];
            #pragma unroll
            for(int n=0;n<2;++n) {
                int token=base+part*32+n*16+row;F4 c={0,0,0,0};
                #pragma unroll
                for(int t=0;t<D/16;++t) {
                    B4 x={0,0,0,0};
                    if(b>=0 && token<min(m[KL],limit)) x=*reinterpret_cast<const B4*>(k+(int64_t(m[KS]+token)*2+hk)*D+t*16+group*4);
                    c=mma<BF>(qfrag[t],x,c);
                }
                #pragma unroll
                for(int r=0;r<4;++r) {int pos=base+part*32+n*16+group+4*r;
                    logits[n][r]=(b>=0 && pos<min(m[KL],limit))?c[r]*scale:-INFINITY;}
            }
            float tm=-INFINITY;
            #pragma unroll
            for(int n=0;n<2;++n) for(int r=0;r<4;++r) tm=fmaxf(tm,logits[n][r]);
            tm=fmaxf(tm,__shfl_xor(tm,16,64));tm=fmaxf(tm,__shfl_xor(tm,32,64));
            float nm=fmaxf(peak,tm),alpha=peak==-INFINITY?0.f:__expf(peak-nm),s=0;
            #pragma unroll
            for(int n=0;n<2;++n) for(int r=0;r<4;++r) {float x=logits[n][r];logits[n][r]=x==-INFINITY?0.f:__expf(x-nm);s+=logits[n][r];}
            s+=__shfl_xor(s,16,64);s+=__shfl_xor(s,32,64);
            #pragma unroll
            for(int t=0;t<D/16;++t) acc[t]*=alpha;
            #pragma unroll
            for(int n=0;n<2;++n) {
                B4 a=as_a<BF>(logits[n],group);
                #pragma unroll
                for(int t=0;t<D/16;++t) {
                    B4 x;
                    #pragma unroll
                    for(int i=0;i<4;++i) {int token=base+part*32+n*16+group*4+i;
                        x[i]=(b>=0 && token<min(m[KL],limit))?v[(int64_t(m[KS]+token)*2+hk)*D+t*16+row]:0;}
                    acc[t]=mma<BF>(a,x,acc[t]);
                }
            }
            sum=sum*alpha+s;peak=nm;
        }
    }
    float inverse=sum>0?1.f/sum:0.f;
    #pragma unroll
    for(int t=0;t<D/16;++t) for(int r=0;r<4;++r) {
        int d=t*16+group+4*r;
        if(row<G) {
            int64_t h=int64_t(index)*(2*G)+hk*G+row;
            if constexpr(Partial) partial[(int64_t(blockIdx.x)*totalq*(2*G)+h)*D+d]=acc[t][r];
            else out[h*D+d]=pack<BF>(acc[t][r]*inverse);
        }
    }
    if(group==0 && row<G) {
        int64_t h=int64_t(index)*(2*G)+hk*G+row;
        if constexpr(Partial) {float* stats=partial+int64_t(splits)*totalq*(2*G)*D;
            int64_t p=(int64_t(blockIdx.x)*totalq*(2*G)+h)*2;stats[p]=peak;stats[p+1]=sum;}
        else lse[h]=sum>0?peak+logf(sum):-INFINITY;
    }
}
template<bool BF>
__global__ __launch_bounds__(64)
void merge_kernel(const float* partial,W* out,float* lse,int totalq,int hq,int d,int splits) {
    int lane=threadIdx.x,head=blockIdx.x;
    int64_t heads=int64_t(totalq)*hq;
    const float* stats=partial+int64_t(splits)*heads*d;
    float peak=-INFINITY,l=0;
    if(lane<splits) {peak=stats[(int64_t(lane)*heads+head)*2];l=stats[(int64_t(lane)*heads+head)*2+1];}
    float global=peak;
    #pragma unroll
    for(int shift=32;shift;shift>>=1) global=fmaxf(global,__shfl_xor(global,shift,64));
    float w=l>0?__expf(peak-global):0.f,den=w*l;
    #pragma unroll
    for(int shift=32;shift;shift>>=1) den+=__shfl_xor(den,shift,64);
    float a=0,b=0;
    for(int split=0;split<splits;++split) {float alpha=__shfl(w,split,64);const float* p=partial+(int64_t(split)*heads+head)*d;
        a+=alpha*p[lane];if(d==128) b+=alpha*p[lane+64];}
    float inv=den>0?1.f/den:0;
    out[int64_t(head)*d+lane]=pack<BF>(a*inv);
    if(d==128) out[int64_t(head)*d+lane+64]=pack<BF>(b*inv);
    if(lane==0) lse[head]=den>0?global+logf(den):-INFINITY;
}

#include "backward.hpp"
extern "C" const char* native_error(int code) {return hipGetErrorString((hipError_t)code);}

extern "C" int native_compress(const void* k,void* out,const void* meta,int maxn,int batches,int d,int bf,int which,void* stream) {
    if(maxn==0) return hipSuccess;
    return dispatch(d,bf,16,[&](auto spec) {
        using T=decltype(spec);
        if(which==1) hipLaunchKernelGGL((compress_kernel<T::D,T::BF,32,16>),dim3(maxn,2,batches),dim3(T::D),0,(hipStream_t)stream,(const W*)k,(W*)out,(const int*)meta,which);
        else hipLaunchKernelGGL((compress_kernel<T::D,T::BF,128,64>),dim3(maxn,2,batches),dim3(T::D),0,(hipStream_t)stream,(const W*)k,(W*)out,(const int*)meta,which);
    });
}
extern "C" int native_stage1(const void* q,const void* k1,const void* k2,void* scores,const void* meta,
                             int maxq,int batches,int totalq,int maxn,int d,int bf,int g,float scale,void* stream) {
    if(maxn==0) return hipSuccess;
    return dispatch(d,bf,g,[&](auto spec) {using T=decltype(spec);
        hipLaunchKernelGGL((stage1_kernel<T::D,T::BF,T::G>),dim3(maxq,2,batches),dim3(64),0,(hipStream_t)stream,
                          (const W*)q,(const W*)k1,(const W*)k2,(W*)scores,(const int*)meta,totalq,maxn,scale);
    });
}
extern "C" int native_pool(const void* score,void* out,const void* meta,int maxq,int batches,int totalq,int maxn,int nb,int initial,int local,int bf,void* stream) {
    #define POOL(B) hipLaunchKernelGGL((pool_kernel<B>),dim3(maxq,2,batches),dim3(128),0,(hipStream_t)stream,(const W*)score,(W*)out,(const int*)meta,totalq,maxn,nb,initial,local)
    if(bf) {POOL(true);} else {POOL(false);}
    #undef POOL
    return hipGetLastError();
}
template<bool BF>
void launch_topk(const void* scores,void* out,const void* meta,int maxq,int batches,int totalq,int nb,int topk,void* stream) {
    #define SORT(N) hipLaunchKernelGGL((topk_kernel<N,BF>),dim3(maxq,2,batches),dim3(128),0,(hipStream_t)stream,(const W*)scores,(int*)out,(const int*)meta,totalq,nb,topk)
    if(nb<=64) {SORT(64);} else if(nb<=128) {SORT(128);} else if(nb<=256) {SORT(256);}
    else if(nb<=512) {SORT(512);} else {SORT(1024);}
    #undef SORT
}
extern "C" int native_topk(const void* score,void* out,const void* meta,int maxq,int batches,int totalq,int nb,int topk,int bf,void* stream) {
    if(nb<1 || nb>1024 || topk<1 || topk>64) return hipErrorInvalidValue;
    if(bf) launch_topk<true>(score,out,meta,maxq,batches,totalq,nb,topk,stream);
    else launch_topk<false>(score,out,meta,maxq,batches,totalq,nb,topk,stream);
    return hipGetLastError();
}
extern "C" int native_attention(const void* q,const void* k,const void* v,const void* selected,void* out,void* lse,
    const void* meta,void* partial,int maxq,int batches,int totalq,int topk,int d,int bf,int g,int dense,int splits,float scale,void* stream) {
    return dispatch(d,bf,g,[&](auto spec) {using T=decltype(spec);
        #define ATTN(DENSE,PART) hipLaunchKernelGGL((attention_kernel<T::D,T::BF,T::G,DENSE,PART>),grid,dim3(64),0,(hipStream_t)stream, \
            (const W*)q,(const W*)k,(const W*)v,(const int*)selected,(W*)out,(float*)lse,(const int*)meta,totalq,topk,scale,(float*)partial,splits)
        dim3 grid(splits>1?splits:maxq,2,batches);
        if(dense) {ATTN(true,false);} else if(splits>1) {ATTN(false,true);} else {ATTN(false,false);}
        #undef ATTN
        if(splits>1 && hipPeekAtLastError()==hipSuccess) {
            hipLaunchKernelGGL((merge_kernel<T::BF>),dim3(totalq*2*g),dim3(64),0,(hipStream_t)stream,
                              (const float*)partial,(W*)out,(float*)lse,totalq,2*g,d,splits);
        }
    });
}
extern "C" int native_backward(const void* q,const void* k,const void* v,const void* out,const void* dout,
    const void* lse,const void* selected,const void* meta,void* dq,void* dk,void* dv,void* delta,void* membership,
    int maxq,int batches,int totalq,int nb,int topk,int d,int bf,int g,int dense,float scale,void* stream) {
    if(!dense) {
        hipError_t rc=hipMemsetAsync(membership,0,int64_t(batches)*2*nb*maxq,(hipStream_t)stream);
        if(rc!=hipSuccess) return rc;
        hipLaunchKernelGGL(reverse_kernel,dim3(maxq,2,batches),dim3(64),0,(hipStream_t)stream,
                          (const int*)selected,(unsigned char*)membership,(const int*)meta,totalq,topk,maxq,nb);
        rc=hipGetLastError();if(rc!=hipSuccess) return rc;
    }
    return dispatch(d,bf,g,[&](auto spec) {using T=decltype(spec);
        #define DQ(DENSE) hipLaunchKernelGGL((dq_kernel<T::D,T::BF,T::G,DENSE>),dim3(maxq,2,batches),dim3(64),0,(hipStream_t)stream, \
            (const W*)q,(const W*)k,(const W*)v,(const W*)out,(const W*)dout,(const float*)lse,(const int*)selected,(W*)dq,(float*)delta,(const int*)meta,totalq,topk,scale)
        #define DKDV(DENSE) hipLaunchKernelGGL((dkdv_kernel<T::D,T::BF,T::G,DENSE>),dim3(nb,2,batches),dim3(256),0,(hipStream_t)stream, \
            (const W*)q,(const W*)k,(const W*)v,(const W*)dout,(const float*)lse,(const float*)delta,(const unsigned char*)membership,(W*)dk,(W*)dv,(const int*)meta,maxq,nb,scale)
        if(dense) {DQ(true);} else {DQ(false);}
        if(hipPeekAtLastError()==hipSuccess) {if(dense) {DKDV(true);} else {DKDV(false);}}
        #undef DQ
        #undef DKDV
    });
}
