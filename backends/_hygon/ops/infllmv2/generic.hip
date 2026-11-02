// SPDX-License-Identifier: Apache-2.0
// Native HIP correctness path for configurations outside the MMAC fast path.
// No Triton, libtorch, host tensor computation or matrix-library dependency.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
struct Params {
    int d,hq,hkv,type,topk,bs,nb,maxn1,kw1,st1,kw2,st2,initial,local,dense,batch,tq,tk,n1,n2;
    float scale;
};
using I=int64_t;
enum {QS=0,QL=1,KS=2,KL=3,C1S=4,C1N=5,C2S=6,C2N=7};
__device__ __forceinline__ float get(const void* x,I i,int type) {
    if(type==2)return ((const float*)x)[i];
    uint16_t b=((const uint16_t*)x)[i];
    return type==0?__uint_as_float(uint32_t(b)<<16):float(__builtin_bit_cast(__fp16,b));
}
__device__ __forceinline__ void put(void* x,I i,float value,int type) {
    if(type==2){((float*)x)[i]=value;return;}
    uint16_t b;
    if(type==0){uint32_t u=__float_as_uint(value);b=((u&0x7fffffffU)>0x7f800000U)?((u>>16)|0x40U):((u+0x7fffU+((u>>16)&1U))>>16);}
    else b=__builtin_bit_cast(uint16_t,(__fp16)value);
    ((uint16_t*)x)[i]=b;
}
__device__ __forceinline__ float sum64(float x) {
    #pragma unroll
    for(int s=32;s;s>>=1)x+=__shfl_xor(x,s,64);
    return x;
}
__device__ __forceinline__ int owner(const int* meta,int batches,int index,int start,int count) {
    int lo=0,hi=batches;
    while(lo<hi){int mid=(lo+hi)/2;if(meta[mid*8+start]+meta[mid*8+count]<=index)lo=mid+1;else hi=mid;}
    return lo;
}
__device__ __forceinline__ float dot(const void* q,const void* k,I qo,I ko,Params p) {
    float x=0;int lane=threadIdx.x&63;
    for(int d=lane;d<p.d;d+=64)x=fmaf(get(q,qo+d,p.type),get(k,ko+d,p.type),x);
    return sum64(x);
}
__global__ void compress(const void* k,void* out,const int* meta,Params p,int which) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    I count=I(which==1?p.n1:p.n2)*p.hkv*p.d;
    if(i>=count)return;
    int chunk=i/(p.hkv*p.d),hk=i/p.d%p.hkv,d=i%p.d;
    int cs=which==1?C1S:C2S,cn=which==1?C1N:C2N;
    const int* m=meta+owner(meta,p.batch,chunk,cs,cn)*8;
    int window=which==1?p.kw1:p.kw2,stride=which==1?p.st1:p.st2;
    int base=m[KS]+(chunk-m[cs])*stride;
    float x=0;
    for(int w=0;w<window;++w)x+=get(k,(I(base+w)*p.hkv+hk)*p.d+d,p.type);
    put(out,i,x/window,p.type);
}
__global__ __launch_bounds__(64)
void coarse_lse(const void* q,const void* k2,float* lse,const int* meta,Params p) {
    int i=blockIdx.x,qi=i/p.hq,head=i%p.hq,lane=threadIdx.x;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int local=qi-m[QS],hk=head/(p.hq/p.hkv);
    int stride=m[C1N]==m[C2N]?p.st1:p.st2;
    int right=max(0,(local+1)/stride-1+m[C2N]-max(0,m[QL]-stride+1)/stride);
    float peak=-INFINITY,total=0;
    for(int t=0;t<min(right,m[C2N]);++t){
        float x=dot(q,k2,(I(qi)*p.hq+head)*p.d,(I(m[C2S]+t)*p.hkv+hk)*p.d,p)*p.scale;
        float nm=fmaxf(peak,x),a=peak==-INFINITY?0.f:expf(peak-nm);
        total=total*a+expf(x-nm);peak=nm;
    }
    if(lane==0)lse[i]=total>0?peak+logf(total):-INFINITY;
}
__global__ __launch_bounds__(256)
void fine_scores(const void* q,const void* k1,const float* lse,void* scores,const int* meta,Params p) {
    I i=I(blockIdx.x)*4+(threadIdx.x>>6);
    if(i>=I(p.hkv)*p.tq*p.maxn1)return;
    int f=i%p.maxn1,qi=i/p.maxn1%p.tq,hk=i/(I(p.maxn1)*p.tq),lane=threadIdx.x&63;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int local=qi-m[QS];
    int right=max(0,(local+1)/p.st1-1+m[C1N]-max(0,m[QL]-p.st1+1)/p.st1);
    float score=0;
    if(f<min(right,m[C1N])){
        int group=p.hq/p.hkv;
        for(int g=0;g<group;++g){int head=hk*group+g;float l=lse[I(qi)*p.hq+head];
            if(l!=-INFINITY){float x=dot(q,k1,(I(qi)*p.hq+head)*p.d,(I(m[C1S]+f)*p.hkv+hk)*p.d,p);score+=expf(x*p.scale-l);}}
    }
    if(lane==0)put(scores,i,score,p.type);
}
__global__ void pool(const void* scores,void* output,const int* meta,Params p) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=I(p.hkv)*p.tq*p.nb)return;
    int b=i%p.nb,qi=i/p.nb%p.tq,hk=i/(I(p.nb)*p.tq);
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int qb=(m[KL]-m[QL]+qi-m[QS])/p.bs;
    float value=-INFINITY;
    if(b<p.initial||(b<=qb&&qb<=b+p.local))value=INFINITY;
    else{
        int window=p.kw1/p.st1+p.bs/p.st1-1,first=b*(p.bs/p.st1)-(p.kw1/p.st1-1);
        for(int j=0;j<window;++j){int f=first+j;if(f>=0&&f<m[C1N])value=fmaxf(value,get(scores,(I(hk)*p.tq+qi)*p.maxn1+f,p.type));}
    }
    put(output,i,value,p.type);
}
template<int N>
__global__ void select(const void* score,int* out,const int* meta,Params p) {
    __shared__ uint64_t keys[N];
    int qi=blockIdx.x%p.tq,hk=blockIdx.x/p.tq,tid=threadIdx.x;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int valid=min(p.nb,min((m[KL]+p.bs-1)/p.bs,(m[KL]-m[QL]+qi-m[QS]+p.bs)/p.bs));
    I dst=(I(hk)*p.tq+qi)*p.topk;
    if(valid<=p.topk){for(int t=tid;t<p.topk;t+=blockDim.x)out[dst+t]=t<valid?t:-1;return;}
    for(int b=tid;b<N;b+=blockDim.x){uint64_t key=0;
        if(b<valid){float x=get(score,(I(hk)*p.tq+qi)*p.nb+b,p.type);if(isnan(x))x=-INFINITY;
            uint32_t bits=__float_as_uint(x),ordered=bits^((bits&0x80000000U)?0xffffffffU:0x80000000U);
            key=(uint64_t(ordered)<<16)|uint64_t(0xffff-b);}
        keys[b]=key;
    }
    __syncthreads();
    for(int size=2;size<=N;size<<=1)for(int step=size>>1;step;step>>=1){
        for(int i=tid;i<N;i+=blockDim.x){int j=i^step;if(j>i){uint64_t a=keys[i],b=keys[j];
            if(((i&size)==0&&a<b)||((i&size)!=0&&a>b)){keys[i]=b;keys[j]=a;}}}
        __syncthreads();
    }
    for(int i=tid;i<p.topk;i+=blockDim.x)keys[i]&=0xffffU;
    __syncthreads();
    for(int size=2;size<=p.topk;size<<=1)for(int step=size>>1;step;step>>=1){
        for(int i=tid;i<p.topk;i+=blockDim.x){int j=i^step;if(j>i){uint64_t a=keys[i],b=keys[j];
            if(((i&size)==0&&a<b)||((i&size)!=0&&a>b)){keys[i]=b;keys[j]=a;}}}
        __syncthreads();
    }
    for(int i=tid;i<p.topk;i+=blockDim.x)out[dst+i]=0xffff-int(keys[i]);
}
__device__ __forceinline__ int rank_block(const int* selected,Params p,int qi,int head,int rank) {
    return p.dense?rank:selected[(I(head/(p.hq/p.hkv))*p.tq+qi)*p.topk+rank];
}
__global__ __launch_bounds__(64)
void attention_lse(const void* q,const void* k,const int* selected,float* lse,const int* meta,Params p) {
    int qi=blockIdx.x/p.hq,head=blockIdx.x%p.hq,lane=threadIdx.x;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int limit=m[KL]-m[QL]+qi-m[QS]+1,hk=head/(p.hq/p.hkv);
    int ranks=p.dense?(limit+p.bs-1)/p.bs:p.topk;
    float peak=-INFINITY,total=0;
    for(int r=0;r<ranks;++r){int b=rank_block(selected,p,qi,head,r);if(b<0)continue;
        for(int t=0;t<p.bs;++t){int token=b*p.bs+t;if(token>=min(m[KL],limit))break;
            float x=dot(q,k,(I(qi)*p.hq+head)*p.d,(I(m[KS]+token)*p.hkv+hk)*p.d,p)*p.scale;
            float nm=fmaxf(peak,x),a=peak==-INFINITY?0.f:expf(peak-nm);
            total=total*a+expf(x-nm);peak=nm;
        }}
    if(lane==0)lse[I(qi)*p.hq+head]=total>0?peak+logf(total):-INFINITY;
}
__global__ __launch_bounds__(64)
void attention_output(const void* q,const void* k,const void* v,const int* selected,const float* lse,
                      void* out,const int* meta,Params p) {
    int tiles=(p.d+63)/64,tile=blockIdx.x%tiles,head=blockIdx.x/tiles%p.hq,qi=blockIdx.x/(I(tiles)*p.hq);
    int lane=threadIdx.x,d=tile*64+lane;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int limit=m[KL]-m[QL]+qi-m[QS]+1,hk=head/(p.hq/p.hkv);
    int ranks=p.dense?(limit+p.bs-1)/p.bs:p.topk;
    float norm=lse[I(qi)*p.hq+head],acc=0;
    if(norm!=-INFINITY)for(int r=0;r<ranks;++r){int b=rank_block(selected,p,qi,head,r);if(b<0)continue;
        for(int t=0;t<p.bs;++t){int token=b*p.bs+t;if(token>=min(m[KL],limit))break;
            float x=dot(q,k,(I(qi)*p.hq+head)*p.d,(I(m[KS]+token)*p.hkv+hk)*p.d,p);
            float weight=expf(x*p.scale-norm);
            if(d<p.d)acc=fmaf(weight,get(v,(I(m[KS]+token)*p.hkv+hk)*p.d+d,p.type),acc);
        }}
    if(d<p.d)put(out,(I(qi)*p.hq+head)*p.d+d,acc,p.type);
}
__global__ __launch_bounds__(64)
void gradient(const void* q,const void* k,const void* v,const void* out,const void* dout,
              const float* lse,const int* selected,float* dq,float* dk,float* dv,const int* meta,Params p) {
    int qi=blockIdx.x/p.hq,head=blockIdx.x%p.hq,lane=threadIdx.x;
    const int* m=meta+owner(meta,p.batch,qi,QS,QL)*8;
    int limit=m[KL]-m[QL]+qi-m[QS]+1,hk=head/(p.hq/p.hkv);
    I qo=(I(qi)*p.hq+head)*p.d;
    float del=0;for(int d=lane;d<p.d;d+=64)del=fmaf(get(out,qo+d,p.type),get(dout,qo+d,p.type),del);
    del=sum64(del);float norm=lse[I(qi)*p.hq+head];
    int ranks=p.dense?(limit+p.bs-1)/p.bs:p.topk;
    if(norm!=-INFINITY)for(int r=0;r<ranks;++r){int b=rank_block(selected,p,qi,head,r);if(b<0)continue;
        for(int t=0;t<p.bs;++t){int token=b*p.bs+t;if(token>=min(m[KL],limit))break;
            I ko=(I(m[KS]+token)*p.hkv+hk)*p.d;
            float prob=expf(dot(q,k,qo,ko,p)*p.scale-norm);
            float dp=dot(dout,v,qo,ko,p),ds=prob*(dp-del)*p.scale;
            for(int d=lane;d<p.d;d+=64){
                dq[qo+d]+=ds*get(k,ko+d,p.type);
                atomicAdd(dk+ko+d,ds*get(q,qo+d,p.type));
                atomicAdd(dv+ko+d,prob*get(dout,qo+d,p.type));
            }
        }}
}
__global__ void cast_grads(const float* dq,const float* dk,const float* dv,void* oq,void* ok,void* ov,Params p) {
    I i=I(blockIdx.x)*blockDim.x+threadIdx.x,nq=I(p.tq)*p.hq*p.d,nk=I(p.tk)*p.hkv*p.d;
    if(i<nq)put(oq,i,dq[i],p.type);
    if(i<nk){put(ok,i,dk[i],p.type);put(ov,i,dv[i],p.type);}
}
extern "C" const char* generic_error(int code){return hipGetErrorString((hipError_t)code);}
extern "C" int generic_compress(const void* k,void* out,const int* meta,const Params* host,int which,void* stream) {
    Params p=*host;I n=I(which==1?p.n1:p.n2)*p.hkv*p.d;if(!n)return hipSuccess;
    hipLaunchKernelGGL(compress,dim3((n+255)/256),dim3(256),0,(hipStream_t)stream,k,out,meta,p,which);
    return hipGetLastError();
}
extern "C" int generic_stage(int stage,const void* q,const void* k,const void* v,void* k1,void* k2,
    void* score,void* pooled,int* selected,void* out,float* lse,const int* meta,const Params* host,void* stream) {
    Params p=*host;hipStream_t s=(hipStream_t)stream;
    if(stage==0){int rc=generic_compress(k,k1,meta,host,1,stream);if(rc)return rc;return generic_compress(k,k2,meta,host,2,stream);}
    if(stage==1 && p.maxn1){
        hipLaunchKernelGGL(coarse_lse,dim3(I(p.tq)*p.hq),dim3(64),0,s,q,k2,lse,meta,p);
        if(hipPeekAtLastError()==hipSuccess)hipLaunchKernelGGL(fine_scores,dim3((I(p.hkv)*p.tq*p.maxn1+3)/4),dim3(256),0,s,q,k1,lse,score,meta,p);
    }else if(stage==2){
        hipLaunchKernelGGL(pool,dim3((I(p.hkv)*p.tq*p.nb+255)/256),dim3(256),0,s,score,pooled,meta,p);
    }else if(stage==3){
        #define SORT(N) hipLaunchKernelGGL((select<N>),dim3(I(p.hkv)*p.tq),dim3(256),0,s,pooled,selected,meta,p)
        if(p.nb<=64){SORT(64);}else if(p.nb<=128){SORT(128);}else if(p.nb<=256){SORT(256);}else if(p.nb<=512){SORT(512);}
        else if(p.nb<=1024){SORT(1024);}else if(p.nb<=2048){SORT(2048);}else if(p.nb<=4096){SORT(4096);}else return hipErrorInvalidValue;
        #undef SORT
    }else if(stage==4){
        hipLaunchKernelGGL(attention_lse,dim3(I(p.tq)*p.hq),dim3(64),0,s,q,k,selected,lse,meta,p);
        if(hipPeekAtLastError()==hipSuccess)hipLaunchKernelGGL(attention_output,dim3(I(p.tq)*p.hq*((p.d+63)/64)),dim3(64),0,s,q,k,v,selected,lse,out,meta,p);
    }
    return hipGetLastError();
}
extern "C" int generic_backward(const void* q,const void* k,const void* v,const void* out,const void* dout,
    const float* lse,const int* selected,const int* meta,float* dq,float* dk,float* dv,void* oq,void* ok,void* ov,
    const Params* host,void* stream) {
    Params p=*host;hipStream_t s=(hipStream_t)stream;I nq=I(p.tq)*p.hq*p.d,nk=I(p.tk)*p.hkv*p.d;
    hipError_t rc=hipMemsetAsync(dq,0,nq*sizeof(float),s);if(rc!=hipSuccess)return rc;
    rc=hipMemsetAsync(dk,0,nk*sizeof(float),s);if(rc!=hipSuccess)return rc;
    rc=hipMemsetAsync(dv,0,nk*sizeof(float),s);if(rc!=hipSuccess)return rc;
    hipLaunchKernelGGL(gradient,dim3(I(p.tq)*p.hq),dim3(64),0,s,q,k,v,out,dout,lse,selected,dq,dk,dv,meta,p);
    if(p.type!=2 && hipPeekAtLastError()==hipSuccess)hipLaunchKernelGGL(cast_grads,dim3(((nq>nk?nq:nk)+255)/256),dim3(256),0,s,dq,dk,dv,oq,ok,ov,p);
    return hipGetLastError();
}
