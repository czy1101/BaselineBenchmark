// SPDX-License-Identifier: Apache-2.0
// MMAC dQ and KV-block-owned dK/dV. Each output has exactly one writer.
#pragma once
#include "common.hpp"
__global__ void reverse_kernel(const int* selected,unsigned char* membership,const int* meta,
                              int totalq,int topk,int maxq,int nb) {
    const int* m=meta+blockIdx.z*8;
    int qi=blockIdx.x,hk=blockIdx.y,r=threadIdx.x;
    if(qi>=m[QL] || r>=topk) return;
    int b=selected[(int64_t(hk)*totalq+m[QS]+qi)*topk+r];
    if(b>=0 && b<nb) membership[((int64_t(blockIdx.z)*2+hk)*nb+b)*maxq+qi]=1;
}
template<int D,bool BF,int G,bool Dense>
__global__ __launch_bounds__(64)
void dq_kernel(const W* q,const W* k,const W* v,const W* out,const W* dout,const float* lse,
               const int* selected,W* dq,float* delta,const int* meta,int totalq,int topk,float scale) {
    const int* m=meta+blockIdx.z*8;
    int qi=blockIdx.x,hk=blockIdx.y,lane=threadIdx.x,row=lane&15,group=lane>>4;
    if(qi>=m[QL]) return;
    int index=m[QS]+qi,limit=m[KL]-m[QL]+qi+1;
    int64_t head=int64_t(index)*(2*G)+hk*G+row;
    B4 a[D/16],da[D/16];F4 accum[D/16];float del=0;
    #pragma unroll
    for(int t=0;t<D/16;++t) {
        a[t]=B4{0,0,0,0};da[t]=B4{0,0,0,0};accum[t]=F4{0,0,0,0};
        if(row<G) {
            a[t]=*reinterpret_cast<const B4*>(q+head*D+t*16+group*4);
            da[t]=*reinterpret_cast<const B4*>(dout+head*D+t*16+group*4);
            #pragma unroll
            for(int i=0;i<4;++i) del+=unpack<BF>(out[head*D+t*16+group*4+i])*unpack<BF>((W)da[t][i]);
        }
    }
    del+=__shfl_xor(del,16,64);del+=__shfl_xor(del,32,64);
    float norm=row<G?lse[head]:-INFINITY;
    if(group==0 && row<G) delta[head]=del;
    int ranks=Dense?(limit+63)/64:min(topk,(limit+63)/64);
    for(int rank=0;rank<ranks;++rank) {
        int b=Dense?rank:selected[(int64_t(hk)*totalq+index)*topk+rank];
        if(b<0) continue;
        #pragma unroll
        for(int part=0;part<4;++part) {
            int base=b*64+part*16,token=base+row;
            F4 logits={0,0,0,0},dp={0,0,0,0};
            #pragma unroll
            for(int t=0;t<D/16;++t) {
                B4 kb={0,0,0,0},vb={0,0,0,0};
                if(token<min(m[KL],limit)) {
                    int64_t pos=(int64_t(m[KS]+token)*2+hk)*D+t*16+group*4;
                    kb=*reinterpret_cast<const B4*>(k+pos);vb=*reinterpret_cast<const B4*>(v+pos);
                }
                logits=mma<BF>(a[t],kb,logits);dp=mma<BF>(da[t],vb,dp);
            }
            F4 ds;
            #pragma unroll
            for(int r=0;r<4;++r) {
                bool valid=row<G && base+group+4*r<min(m[KL],limit) && norm!=-INFINITY;
                float p=valid?__expf(logits[r]*scale-norm):0.f;
                ds[r]=p*(dp[r]-del)*scale;
            }
            B4 ds_a=as_a<BF>(ds,group);
            #pragma unroll
            for(int t=0;t<D/16;++t) {
                B4 kb;
                #pragma unroll
                for(int i=0;i<4;++i) {int p=base+group*4+i;
                    kb[i]=p<min(m[KL],limit)?k[(int64_t(m[KS]+p)*2+hk)*D+t*16+row]:0;}
                accum[t]=mma<BF>(ds_a,kb,accum[t]);
            }
        }
    }
    if(row<G) {
        #pragma unroll
        for(int t=0;t<D/16;++t) for(int r=0;r<4;++r) dq[head*D+t*16+group+4*r]=pack<BF>(accum[t][r]);
    }
}
template<int D,bool BF,int G,bool Dense>
__global__ __launch_bounds__(256)
void dkdv_kernel(const W* q,const W* k,const W* v,const W* dout,const float* lse,const float* delta,
                 const unsigned char* membership,W* dk,W* dv,const int* meta,
                 int maxq,int nb,float scale) {
    const int* m=meta+blockIdx.z*8;
    int b=blockIdx.x,hk=blockIdx.y,lane=threadIdx.x&63,wave=threadIdx.x>>6;
    int row=lane&15,group=lane>>4,base=b*64+wave*16,token=base+row;
    B4 kfrag[D/16],vfrag[D/16];F4 ak[D/16],av[D/16];
    #pragma unroll
    for(int t=0;t<D/16;++t) {
        kfrag[t]=B4{0,0,0,0};vfrag[t]=B4{0,0,0,0};ak[t]=F4{0,0,0,0};av[t]=F4{0,0,0,0};
        if(token<m[KL]) {
            int64_t pos=(int64_t(m[KS]+token)*2+hk)*D+t*16+group*4;
            kfrag[t]=*reinterpret_cast<const B4*>(k+pos);vfrag[t]=*reinterpret_cast<const B4*>(v+pos);
        }
    }
    for(int qi=0;qi<m[QL];++qi) {
        int limit=m[KL]-m[QL]+qi+1;
        bool active=Dense?b*64<limit:membership[((int64_t(blockIdx.z)*2+hk)*nb+b)*maxq+qi]!=0;
        if(!active) continue;  // Uniform inside every independent wave64.
        int index=m[QS]+qi;
        F4 logits={0,0,0,0},dp={0,0,0,0};
        #pragma unroll
        for(int t=0;t<D/16;++t) {
            B4 qb={0,0,0,0},dob={0,0,0,0};
            if(row<G) {
                int64_t pos=(int64_t(index)*(2*G)+hk*G+row)*D+t*16+group*4;
                qb=*reinterpret_cast<const B4*>(q+pos);dob=*reinterpret_cast<const B4*>(dout+pos);
            }
            logits=mma<BF>(kfrag[t],qb,logits);dp=mma<BF>(vfrag[t],dob,dp);
        }
        F4 p,ds;
        #pragma unroll
        for(int r=0;r<4;++r) {
            int head=group+4*r;
            int64_t pos=int64_t(index)*(2*G)+hk*G+head;
            float norm=head<G?lse[pos]:-INFINITY,del=head<G?delta[pos]:0;
            bool valid=head<G && token<min(m[KL],limit) && norm!=-INFINITY;
            p[r]=valid?__expf(logits[r]*scale-norm):0.f;ds[r]=p[r]*(dp[r]-del)*scale;
        }
        B4 pa=as_a<BF>(p,group),sa=as_a<BF>(ds,group);
        #pragma unroll
        for(int t=0;t<D/16;++t) {
            B4 qb,dob;
            #pragma unroll
            for(int i=0;i<4;++i) {
                int h=group*4+i;
                int64_t pos=(int64_t(index)*(2*G)+hk*G+h)*D+t*16+row;
                qb[i]=h<G?q[pos]:0;dob[i]=h<G?dout[pos]:0;
            }
            ak[t]=mma<BF>(sa,qb,ak[t]);av[t]=mma<BF>(pa,dob,av[t]);
        }
    }
    if(token<m[KL]) {
        #pragma unroll
        for(int t=0;t<D/16;++t) for(int r=0;r<4;++r) {
            int64_t pos=(int64_t(m[KS]+token)*2+hk)*D+t*16+group+4*r;
            dk[pos]=pack<BF>(ak[t][r]);dv[pos]=pack<BF>(av[t][r]);
        }
    }
}
