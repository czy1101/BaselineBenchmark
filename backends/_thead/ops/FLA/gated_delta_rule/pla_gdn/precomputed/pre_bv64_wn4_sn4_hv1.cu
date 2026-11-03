#include "src/decode/k-last/fused_sigmoid_gating_k_last_kernel.cuh"
void pre_bv64_wn4_sn4_hv1(const void* q, const void* k, const void* v, const float* g,
 const void* beta, void* out, void* state, const int32_t* indices,
 int B, int H, int HV, int64_t state_stride, float scale, bool norm,
 int shared_bytes, hggcStream_t stream) {
  using S=__ppu_bfloat16;
  dim3 grid(128/64, HV/H/1, B*H), block(32,4);
  fused_sigmoid_gating_delta_kernel<float,S,4,64,1,4,false,false,false,true,S>
  <<<grid,block,shared_bytes,stream>>>(
    nullptr,g,nullptr,1.0f,20.0f,
    (const S*)q,(const S*)k,(const S*)v,(const S*)beta,(S*)out,
    (S*)state,nullptr,nullptr,indices,nullptr,0,nullptr,0,0,scale,1,
    HV,H*128,H*128,HV*128,HV,state_stride,B,H,HV,128,128,
    false,false,norm,false,nullptr,0,false);
}
