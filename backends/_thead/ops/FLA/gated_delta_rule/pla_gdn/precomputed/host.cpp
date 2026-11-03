#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "src/ppu_torch_stream.h"
#include "src/common.cuh"
#include "src/decode/k-last/k_last_shared_layout.h"
#include "src/decode/k-last/generated/tuned_dispatch.h"
void pre_bv16_wn4_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv32_wn4_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv32_wn4_sn5_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv32_wn4_sn6_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv32_wn4_sn8_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv32_wn8_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn4_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn4_sn5_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn4_sn6_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn4_sn8_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn8_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn8_sn5_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn8_sn6_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv64_wn8_sn8_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn4_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn4_sn5_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn4_sn6_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn4_sn8_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn8_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn8_sn5_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn8_sn6_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
void pre_bv128_wn8_sn8_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream);
using Launcher=void(*)(const void*,const void*,const void*,const float*,const void*,void*,void*,
 const int32_t*,int,int,int,int64_t,float,bool,int,hggcStream_t);
std::vector<int> config(int B,int H,int HV) {
 auto p=get_tuned_k_last(B,H,HV,128,128);return {p.bv,p.wn,p.sn,p.hv_per_cta};
}
std::vector<int> packed_config(int B,bool bf16) {auto p=get_tuned_k_last_packed(B,4,12,128,128,bf16);return {p.bv,p.wn,p.sn,p.hv_per_cta};}
torch::Tensor run(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,
 torch::Tensor beta,torch::Tensor state,torch::Tensor indices,float scale,bool norm) {
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && k.scalar_type()==q.scalar_type()
  && v.scalar_type()==q.scalar_type() && beta.scalar_type()==q.scalar_type()
  && state.scalar_type()==q.scalar_type() && g.scalar_type()==at::kFloat
  && indices.scalar_type()==at::kInt,"dtype/device mismatch");
 TORCH_CHECK(q.dim()==4 && q.size(1)==1 && q.size(3)==128 && v.size(3)==128,
  "only benchmark T1 K=V128 is supported");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
  && g.is_contiguous() && beta.is_contiguous() && state.is_contiguous()
  && indices.is_contiguous(),"contiguous inputs required");
 int B=q.size(0),H=q.size(2),HV=v.size(2);
 TORCH_CHECK(HV%H==0 && g.numel()==B*HV && beta.numel()==B*HV
  && indices.numel()==B,"shape mismatch");
 auto tp=get_tuned_k_last(B,H,HV,128,128);Launcher launch=nullptr;
if (tp.bv==16 && tp.wn==4 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv16_wn4_sn4_hv1;
if (tp.bv==32 && tp.wn==4 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv32_wn4_sn4_hv1;
if (tp.bv==32 && tp.wn==4 && tp.sn==5 && tp.hv_per_cta==1) launch=&pre_bv32_wn4_sn5_hv1;
if (tp.bv==32 && tp.wn==4 && tp.sn==6 && tp.hv_per_cta==1) launch=&pre_bv32_wn4_sn6_hv1;
if (tp.bv==32 && tp.wn==4 && tp.sn==8 && tp.hv_per_cta==1) launch=&pre_bv32_wn4_sn8_hv1;
if (tp.bv==32 && tp.wn==8 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv32_wn8_sn4_hv1;
if (tp.bv==64 && tp.wn==4 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv64_wn4_sn4_hv1;
if (tp.bv==64 && tp.wn==4 && tp.sn==5 && tp.hv_per_cta==1) launch=&pre_bv64_wn4_sn5_hv1;
if (tp.bv==64 && tp.wn==4 && tp.sn==6 && tp.hv_per_cta==1) launch=&pre_bv64_wn4_sn6_hv1;
if (tp.bv==64 && tp.wn==4 && tp.sn==8 && tp.hv_per_cta==1) launch=&pre_bv64_wn4_sn8_hv1;
if (tp.bv==64 && tp.wn==8 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv64_wn8_sn4_hv1;
if (tp.bv==64 && tp.wn==8 && tp.sn==5 && tp.hv_per_cta==1) launch=&pre_bv64_wn8_sn5_hv1;
if (tp.bv==64 && tp.wn==8 && tp.sn==6 && tp.hv_per_cta==1) launch=&pre_bv64_wn8_sn6_hv1;
if (tp.bv==64 && tp.wn==8 && tp.sn==8 && tp.hv_per_cta==1) launch=&pre_bv64_wn8_sn8_hv1;
if (tp.bv==128 && tp.wn==4 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv128_wn4_sn4_hv1;
if (tp.bv==128 && tp.wn==4 && tp.sn==5 && tp.hv_per_cta==1) launch=&pre_bv128_wn4_sn5_hv1;
if (tp.bv==128 && tp.wn==4 && tp.sn==6 && tp.hv_per_cta==1) launch=&pre_bv128_wn4_sn6_hv1;
if (tp.bv==128 && tp.wn==4 && tp.sn==8 && tp.hv_per_cta==1) launch=&pre_bv128_wn4_sn8_hv1;
if (tp.bv==128 && tp.wn==8 && tp.sn==4 && tp.hv_per_cta==1) launch=&pre_bv128_wn8_sn4_hv1;
if (tp.bv==128 && tp.wn==8 && tp.sn==5 && tp.hv_per_cta==1) launch=&pre_bv128_wn8_sn5_hv1;
if (tp.bv==128 && tp.wn==8 && tp.sn==6 && tp.hv_per_cta==1) launch=&pre_bv128_wn8_sn6_hv1;
if (tp.bv==128 && tp.wn==8 && tp.sn==8 && tp.hv_per_cta==1) launch=&pre_bv128_wn8_sn8_hv1;
 TORCH_CHECK(launch,"uncompiled official config");
 const pla::KLastSharedLayout layout(tp.sn,128,tp.wn,tp.bv,tp.hv_per_cta,false,false,1);
 auto out=torch::empty_like(v);
 launch(q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr<float>(),beta.data_ptr(),
  out.data_ptr(),state.data_ptr(),indices.data_ptr<int32_t>(),B,H,HV,state.stride(0),
  scale,norm,layout.total_bytes,pla::current_stream());
 return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("config",&config);m.def("packed_config",&packed_config);}
