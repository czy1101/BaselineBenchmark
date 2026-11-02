// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
#include <hytlass/arch/mmac_gfx9.h>
using W=uint16_t;
using B4=short __attribute__((ext_vector_type(4)));
using H4=__fp16 __attribute__((ext_vector_type(4)));
using F4=float __attribute__((ext_vector_type(4)));

template<bool BF> __device__ __forceinline__ float unpack(W x) {
    if constexpr(BF) return __uint_as_float(uint32_t(x)<<16);
    else return float(__builtin_bit_cast(__fp16,x));
}
template<bool BF> __device__ __forceinline__ W pack(float x) {
    if constexpr(BF) {
        uint32_t b=__float_as_uint(x);
        if((b&0x7fffffffU)>0x7f800000U) return (b>>16)|0x40U;
        return (b+0x7fffU+((b>>16)&1U))>>16;
    } else return __builtin_bit_cast(W,(__fp16)x);
}
template<bool BF> __device__ __forceinline__ F4 mma(B4 a,B4 b,F4 c) {
#if defined(__gfx936__)
    if constexpr(BF) return hytlass::arch::detail::mmac_f32_16x16x16_bf16(a,b,c);
    else return hytlass::arch::detail::mmac_f32_16x16x16_f16(__builtin_bit_cast(H4,a),__builtin_bit_cast(H4,b),c);
#else
    return c;
#endif
}
__device__ __forceinline__ F4 c_to_a(F4 x,int group) {
    F4 first,result;
    #pragma unroll
    for(int r=0;r<4;++r) {float y=__shfl_xor(x[r^1],16,64);first[r]=((group^r)&1)?y:x[r];}
    #pragma unroll
    for(int r=0;r<4;++r) {float y=__shfl_xor(first[r^2],32,64);result[r]=((group^r)&2)?y:first[r];}
    return result;
}
template<bool BF> __device__ __forceinline__ B4 as_a(F4 x,int group) {
    F4 y=c_to_a(x,group); B4 a;
    #pragma unroll
    for(int i=0;i<4;++i) a[i]=pack<BF>(y[i]);
    return a;
}
// Eight metadata integers per sequence: Q start/length, K start/length,
// compressed K1 start/count, compressed K2 start/count.
enum {QS=0,QL=1,KS=2,KL=3,C1S=4,C1N=5,C2S=6,C2N=7};
template<int Dim,bool BFloat,int Group> struct Spec {
    static constexpr int D=Dim,G=Group;
    static constexpr bool BF=BFloat;
};
template<class F> int dispatch(int d,int bf,int g,F fn) {
    #define RUN(D,B,G) fn(Spec<D,B,G>{})
    if(d==64 && bf && g==16) {RUN(64,true,16);}
    else if(d==128 && bf && g==16) {RUN(128,true,16);}
    else if(d==64 && !bf && g==16) {RUN(64,false,16);}
    else if(d==128 && !bf && g==16) {RUN(128,false,16);}
    else if(d==64 && bf && g==4) {RUN(64,true,4);}
    else if(d==128 && bf && g==4) {RUN(128,true,4);}
    else if(d==64 && !bf && g==4) {RUN(64,false,4);}
    else if(d==128 && !bf && g==4) {RUN(128,false,4);}
    else return hipErrorInvalidValue;
    #undef RUN
    return hipGetLastError();
}
