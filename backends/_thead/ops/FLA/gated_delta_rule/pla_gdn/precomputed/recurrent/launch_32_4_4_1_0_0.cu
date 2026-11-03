#include "kernel.cuh"
void launch_32_4_4_1_0_0(const void* q,const void* k,const void* v,const float* g,const void* beta,
 void* out,float* state,const int32_t* indices,const int32_t* cu,const int32_t* accepted,
 int n,int h,int hv,int max_tokens,int sq,int sk,int sv,int sg,int sb,
 int64_t state_stride,int index_stride,float scale,bool norm,int smem,hggcStream_t stream) {
 using S=__half;
 dim3 grid(128/32,hv/h/1,n*h),block(32,4);
 fused_sigmoid_gating_delta_kernel<float,S,4,32,1,4,false,false,false,true,float>
 <<<grid,block,smem,stream>>>(nullptr,g,nullptr,1.0f,20.0f,
 (const S*)q,(const S*)k,(const S*)v,(const S*)beta,(S*)out,state,cu,
 nullptr,indices,nullptr,0,nullptr,0,0,scale,max_tokens,sg,sq,sk,sv,sb,state_stride,
 n,h,hv,128,128,false,false,norm,false,accepted,index_stride,false);
}
