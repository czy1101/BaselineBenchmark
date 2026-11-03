#pragma once
#include <cstdint>
#include <hggc_runtime.h>
void launch_16_4_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_16_4_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_16_4_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_16_4_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_8_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_8_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_8_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_8_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_4_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_4_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_4_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_32_4_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_4_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_4_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_4_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_4_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_128_4_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_128_4_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_128_4_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_128_4_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_8_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_8_4_1_0_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_8_4_1_1_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
void launch_64_8_4_1_1_1(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
using Launcher=void(*)(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream);
inline Launcher select_launcher(int bv,int wn,int sn,int hp,bool bf16,bool mtp) {
if(bv==16 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_16_4_4_1_0_0;
if(bv==16 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_16_4_4_1_0_1;
if(bv==16 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_16_4_4_1_1_0;
if(bv==16 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_16_4_4_1_1_1;
if(bv==32 && wn==8 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_32_8_4_1_0_0;
if(bv==32 && wn==8 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_32_8_4_1_0_1;
if(bv==32 && wn==8 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_32_8_4_1_1_0;
if(bv==32 && wn==8 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_32_8_4_1_1_1;
if(bv==32 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_32_4_4_1_0_0;
if(bv==32 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_32_4_4_1_0_1;
if(bv==32 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_32_4_4_1_1_0;
if(bv==32 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_32_4_4_1_1_1;
if(bv==64 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_64_4_4_1_0_0;
if(bv==64 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_64_4_4_1_0_1;
if(bv==64 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_64_4_4_1_1_0;
if(bv==64 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_64_4_4_1_1_1;
if(bv==128 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_128_4_4_1_0_0;
if(bv==128 && wn==4 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_128_4_4_1_0_1;
if(bv==128 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_128_4_4_1_1_0;
if(bv==128 && wn==4 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_128_4_4_1_1_1;
if(bv==64 && wn==8 && sn==4 && hp==1 && bf16==false && mtp==false) return &launch_64_8_4_1_0_0;
if(bv==64 && wn==8 && sn==4 && hp==1 && bf16==false && mtp==true) return &launch_64_8_4_1_0_1;
if(bv==64 && wn==8 && sn==4 && hp==1 && bf16==true && mtp==false) return &launch_64_8_4_1_1_0;
if(bv==64 && wn==8 && sn==4 && hp==1 && bf16==true && mtp==true) return &launch_64_8_4_1_1_1;
return nullptr;
}
