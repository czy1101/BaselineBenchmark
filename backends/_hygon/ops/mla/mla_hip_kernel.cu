#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <cmath>

constexpr int THREADS = 256;

template <typename scalar_t, int GROUP=4>
__global__ void mla_decode_group_kernel(const scalar_t* __restrict__ q,
    const int* __restrict__ table, const scalar_t* __restrict__ cache,
    const int* __restrict__ lengths, scalar_t* __restrict__ out,
    float* __restrict__ lse, int B, int SQ, int HQ, int D, int DV,
    int max_blocks, int block_size, float scale, bool causal) {
  constexpr int LANES=32;
  int g=blockIdx.x, group=g%((HQ+GROUP-1)/GROUP); g/=((HQ+GROUP-1)/GROUP);
  int sq=g%SQ, b=g/SQ, lane=threadIdx.x&31, wi=threadIdx.x>>5;
  if(b>=B || wi>=GROUP) return;
  int h=group*GROUP+wi; if(h>=HQ) return;
  int len=lengths[b], qbase=((b*SQ+sq)*HQ+h)*D;
  float acc[16] = {0.f};
  float m=-INFINITY,z=0.f;
  __shared__ float sm[GROUP], sz[GROUP], sa[GROUP], sw[GROUP];
  // HKV==1: all query heads in this block read the same cache row. Stage it
  // once in LDS so the 4/8 warps reuse K and V instead of reloading globally.
  // Keep the tile in input precision (FP16/BF16); arithmetic below remains
  // FP32. This halves LDS traffic versus a float32 staging tile.
  __shared__ scalar_t sk[576], sv[512];
  // Q is invariant across the KV scan.  Cache the per-head Q vector in
  // registers once instead of rereading it from global memory for every KV
  // token.  D is 512 or 576 on the MLA target shapes (18 float lanes max).
  float qreg[18];
  #pragma unroll
  for (int j=0; j<18; ++j) {
    int qd = lane + 32*j;
    qreg[j] = (qd < D) ? (float)q[qbase + qd] : 0.0f;
  }
  for(int t=0;t<len;++t){
    if(causal && t>len-SQ+sq) continue;
    int blk=table[b*max_blocks+t/block_size], off=t%block_size;
    int cbase=(blk*block_size+off)*D;
    for (int i=threadIdx.x; i<D; i+=blockDim.x) sk[i]=cache[cbase+i];
    for (int i=threadIdx.x; i<DV; i+=blockDim.x) sv[i]=cache[cbase+i];
    __syncthreads();
    float dot=0.f;
    #pragma unroll
    for (int j=0; j<18; ++j) {
      int kd = lane + 32*j;
      if (kd < D) dot += qreg[j] * (float)sk[kd];
    }
    for(int sh=16;sh;sh>>=1) dot+=__shfl_down(dot,sh);
    if(lane==0){
      float score=dot*scale, nm=fmaxf(m,score);
      sa[wi]=__expf(m-nm); sw[wi]=__expf(score-nm);
      z=z*sa[wi]+sw[wi]; m=nm; sm[wi]=m; sz[wi]=z;
    }
    __syncwarp();
    float al=sa[wi], wt=sw[wi];
    for(int j=0;j<16;++j){
      int dv=lane+32*j;
      if(dv<DV) acc[j]=acc[j]*al+wt*(float)sv[dv];
    }
    __syncwarp();
    __syncthreads();
  }
  int obase=((b*SQ+sq)*HQ+h)*DV;
  for(int j=0;j<16;++j){
    int dv=lane+32*j;
    if(dv<DV) out[obase+dv]=(scalar_t)(acc[j]/sz[wi]);
  }
  if(lane==0) lse[(b*HQ+h)*SQ+sq]=logf(sz[wi])+sm[wi];
}

template <typename scalar_t>
__global__ void mla_decode_v2_kernel(const scalar_t* __restrict__ q,
    const int* __restrict__ table, const scalar_t* __restrict__ cache,
    const int* __restrict__ lengths, scalar_t* __restrict__ out,
    float* __restrict__ lse, int B, int SQ, int HQ, int HKV, int D, int DV,
    int max_blocks, int block_size, float scale, bool causal) {
  int pid=blockIdx.x, hq=pid%HQ; pid/=HQ; int sq=pid%SQ, b=pid/SQ;
  if (b>=B) return;
  int len=lengths[b], hkv=hq/(HQ/HKV), qbase=((b*SQ+sq)*HQ+hq)*D;
  float acc0=0.f, acc1=0.f, m=-INFINITY, z=0.f;
  __shared__ float red[THREADS], score, alpha, weight, new_m, final_z, final_m;
  for (int t=0;t<len;++t) {
    if (causal && t>len-SQ+sq) continue;
    int blk=table[b*max_blocks+t/block_size], off=t%block_size;
    int cbase=((blk*block_size+off)*HKV+hkv)*D;
    float dot=0.f;
    for (int d=threadIdx.x;d<D;d+=blockDim.x)
      dot+=(float)q[qbase+d]*(float)cache[cbase+d];
    red[threadIdx.x]=dot; __syncthreads();
    for(int s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)red[threadIdx.x]+=red[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0){
      score=red[0]*scale; new_m=fmaxf(m,score);
      alpha=expf(m-new_m); weight=expf(score-new_m);
      z=z*alpha+weight; m=new_m;
    }
    __syncthreads();
    int dv=threadIdx.x;
    if(dv<DV) acc0=acc0*alpha+weight*(float)cache[cbase+dv];
    if(dv+blockDim.x<DV) acc1=acc1*alpha+weight*(float)cache[cbase+dv+blockDim.x];
    __syncthreads();
  }
  int obase=((b*SQ+sq)*HQ+hq)*DV;
  if(threadIdx.x==0) { final_z=z; final_m=m; }
  __syncthreads();
  if(threadIdx.x<DV) out[obase+threadIdx.x]=(scalar_t)(acc0/final_z);
  if(threadIdx.x+blockDim.x<DV) out[obase+threadIdx.x+blockDim.x]=(scalar_t)(acc1/final_z);
  if(threadIdx.x==0) lse[(b*HQ+hq)*SQ+sq]=logf(final_z)+final_m;
}

template <typename scalar_t>
__global__ void mla_split_kernel(const scalar_t* __restrict__ q,
    const int* __restrict__ table, const scalar_t* __restrict__ cache,
    const int* __restrict__ lengths, float* __restrict__ partial,
    float* __restrict__ stats, int B, int SQ, int HQ, int D, int DV,
    int max_blocks, int block_size, int splits, float scale, bool causal) {
  int pid=blockIdx.x, split=pid%splits; pid/=splits;
  int h=pid%HQ; pid/=HQ; int sq=pid%SQ, b=pid/SQ;
  if (b>=B) return;
  int len=lengths[b], start=(len*split)/splits, end=(len*(split+1))/splits;
  int qbase=((b*SQ+sq)*HQ+h)*D;
  float m=-INFINITY, z=0.f;
  float acc0=0.f, acc1=0.f;
  __shared__ float red[THREADS], sm, sz;
  for (int t=start;t<end;++t) {
    if (causal && t>len-SQ+sq) continue;
    int blk=table[b*max_blocks+t/block_size], off=t%block_size;
    int cbase=(blk*block_size+off)*D;
    float dot=0.f;
    for(int d=threadIdx.x;d<D;d+=blockDim.x)
      dot+=(float)q[qbase+d]*(float)cache[cbase+d];
    // Warp-local reduction avoids the multi-step block tree and its repeated
    // barriers. Only one value per warp reaches LDS; warp 0 folds the warp
    // sums into the dot product used by online softmax.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off; off >>= 1)
      dot += __shfl_down(dot, off);
    if (lane == 0) red[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float block_dot = (lane < (THREADS / 32)) ? red[lane] : 0.0f;
      for (int off = 16; off; off >>= 1)
        block_dot += __shfl_down(block_dot, off);
      if (lane == 0) red[0] = block_dot;
    }
    __syncthreads();
    if(threadIdx.x==0){
      float score=red[0]*scale, nm=fmaxf(m,score);
      float a=expf(m-nm), w=expf(score-nm);
      z=z*a+w; m=nm; sm=a; sz=w;
    }
    __syncthreads();
    float a=sm,w=sz; int dv_i=threadIdx.x;
    if(dv_i<DV) acc0=acc0*a+w*(float)cache[cbase+dv_i];
    if(dv_i+blockDim.x<DV) acc1=acc1*a+w*(float)cache[cbase+dv_i+blockDim.x];
    __syncthreads();
  }
  int row=((b*SQ+sq)*HQ+h)*splits+split;
  if(threadIdx.x==0){ stats[row*2]=m; stats[row*2+1]=z; }
  int ob=row*DV;
  if(threadIdx.x<DV) partial[ob+threadIdx.x]=acc0;
  if(threadIdx.x+blockDim.x<DV) partial[ob+threadIdx.x+blockDim.x]=acc1;
}

template <typename scalar_t>
__global__ void mla_split_combine_kernel(const float* __restrict__ partial,
    const float* __restrict__ stats, scalar_t* __restrict__ out,
    float* __restrict__ lse, int rows, int DV, int splits) {
  int row=blockIdx.x; if(row>=rows) return;
  __shared__ float gm, gz;
  if(threadIdx.x==0){
    float m=-INFINITY;
    for(int s=0;s<splits;++s) m=fmaxf(m,stats[(row*splits+s)*2]);
    float z=0.f;
    for(int s=0;s<splits;++s){
      float ms=stats[(row*splits+s)*2], zs=stats[(row*splits+s)*2+1];
      z+=zs*expf(ms-m);
    }
    gm=m; gz=z; lse[row]=logf(z)+m;
  }
  __syncthreads();
  for(int d=threadIdx.x;d<DV;d+=blockDim.x){
    float a=0.f;
    for(int s=0;s<splits;++s){
      float ms=stats[(row*splits+s)*2], zs=stats[(row*splits+s)*2+1];
      a+=partial[(row*splits+s)*DV+d]*expf(ms-gm);
    }
    out[row*DV+d]=(scalar_t)(a/gz);
  }
}

template <typename scalar_t>
__global__ void mla_kernel(const scalar_t* __restrict__ q,
                           const int* __restrict__ table,
                           const scalar_t* __restrict__ cache,
                           const int* __restrict__ lengths,
                           scalar_t* __restrict__ out,
                           float* __restrict__ lse,
                           int B, int SQ, int HQ, int HKV, int D, int DV,
                           int max_blocks, int block_size, float scale,
                           bool causal) {
  int pid = blockIdx.x;
  int hq = pid % HQ; pid /= HQ;
  int sq = pid % SQ; int b = pid / SQ;
  if (b >= B) return;
  int len = lengths[b];
  int hkv = hq / (HQ / HKV);
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (int t = 0; t < len; ++t) {
    if (causal && t > len - SQ + sq) continue;
    int blk = table[b * max_blocks + t / block_size];
    int off = t % block_size;
    int cbase = ((blk * block_size + off) * HKV + hkv) * D;
    int qbase = ((b * SQ + sq) * HQ + hq) * D;
    float dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
      dot += (float)q[qbase + d] * (float)cache[cbase + d];
    __shared__ float red[THREADS];
    red[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
      if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float score = red[0] * scale;
      float new_max = fmaxf(local_max, score);
      local_sum = local_sum * expf(local_max - new_max) + expf(score - new_max);
      local_max = new_max;
      red[0] = score;
    }
    __syncthreads();
    if (threadIdx.x == 0) red[1] = local_max;
    __syncthreads();
  }
  __shared__ float max_s, sum_s;
  if (threadIdx.x == 0) { max_s = local_max; sum_s = local_sum; }
  __syncthreads();
  int obase = ((b * SQ + sq) * HQ + hq) * DV;
  for (int dv = threadIdx.x; dv < DV; dv += blockDim.x) {
    float acc = 0.0f;
    for (int t = 0; t < len; ++t) {
      if (causal && t > len - SQ + sq) continue;
      int blk = table[b * max_blocks + t / block_size];
      int off = t % block_size;
      int cbase = ((blk * block_size + off) * HKV + hkv) * D;
      int qbase = ((b * SQ + sq) * HQ + hq) * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += (float)q[qbase+d] * (float)cache[cbase+d];
      acc += expf(dot * scale - max_s) * (float)cache[cbase + dv];
    }
    out[obase + dv] = (scalar_t)(acc / sum_s);
  }
  if (threadIdx.x == 0) lse[(b * HQ + hq) * SQ + sq] = logf(sum_s) + max_s;
}

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor q, torch::Tensor table, torch::Tensor cache,
    torch::Tensor lengths, int64_t max_blocks, int64_t block_size,
    int64_t h_kv, int64_t dv, bool causal) {
  TORCH_CHECK(q.is_cuda() && table.is_cuda() && cache.is_cuda() && lengths.is_cuda());
  TORCH_CHECK(q.is_contiguous() && table.is_contiguous() && cache.is_contiguous());
  int B=q.size(0), SQ=q.size(1), HQ=q.size(2), D=q.size(3);
  auto out = torch::empty({B,SQ,HQ,dv}, q.options());
  auto lse = torch::empty({B,HQ,SQ}, q.options().dtype(torch::kFloat));
  // Split-KV is useful when a request has too few independent head CTAs.
  // For large decode batches (e.g. B=128,HQ=128), the regular group8 path
  // already launches thousands of CTAs; splitting there only adds partial
  // buffers and a combine launch.
  const int head_ctas = B * SQ * HQ;
  const bool use_split = (SQ == 1 && h_kv == 1 && HQ % 8 == 0 &&
                          dv == 512 && max_blocks * block_size >= 8192 &&
                          head_ctas < 4096);
  // More shards only help when there are few independent head CTAs.
  // Keep 8-way Split-KV for B=1/2 and 4-way for the B=8 boundary;
  // measurements show that 2-way splitting underutilizes the KV stream.
  const int splits = (head_ctas < 1024) ? 8 : 4;
  auto partial = use_split ? torch::empty({B*SQ*HQ*splits, dv},
                                          q.options().dtype(torch::kFloat)) : torch::Tensor();
  auto stats = use_split ? torch::empty({B*SQ*HQ*splits, 2},
                                        q.options().dtype(torch::kFloat)) : torch::Tensor();
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "mla_forward", [&] {
    if (use_split) {
      dim3 grid(B*SQ*HQ*splits), block(THREADS);
      hipLaunchKernelGGL((mla_split_kernel<scalar_t>), grid, block, 0, stream,
        q.data_ptr<scalar_t>(), table.data_ptr<int>(), cache.data_ptr<scalar_t>(),
        lengths.data_ptr<int>(), partial.data_ptr<float>(), stats.data_ptr<float>(),
        B,SQ,HQ,D,(int)dv,(int)max_blocks,(int)block_size,splits,
        1.0f/sqrtf((float)D),causal);
      dim3 cgrid(B*SQ*HQ), cblock(THREADS);
      hipLaunchKernelGGL((mla_split_combine_kernel<scalar_t>), cgrid, cblock, 0, stream,
        partial.data_ptr<float>(), stats.data_ptr<float>(), out.data_ptr<scalar_t>(),
        lse.data_ptr<float>(), B*SQ*HQ, (int)dv, splits);
    } else if (SQ == 1 && h_kv == 1 && HQ % 8 == 0) {
      dim3 grid(B*SQ*(HQ/8)), block(256);
      hipLaunchKernelGGL((mla_decode_group_kernel<scalar_t,8>), grid, block, 0, stream,
        q.data_ptr<scalar_t>(), table.data_ptr<int>(), cache.data_ptr<scalar_t>(),
        lengths.data_ptr<int>(), out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
        B,SQ,HQ,D,(int)dv,(int)max_blocks,(int)block_size,
        1.0f/sqrtf((float)D),causal);
    } else if (SQ == 1 && h_kv == 1 && HQ % 4 == 0) {
      dim3 grid(B*SQ*(HQ/4)), block(128);
      hipLaunchKernelGGL((mla_decode_group_kernel<scalar_t,4>), grid, block, 0, stream,
        q.data_ptr<scalar_t>(), table.data_ptr<int>(), cache.data_ptr<scalar_t>(),
        lengths.data_ptr<int>(), out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
        B,SQ,HQ,D,(int)dv,(int)max_blocks,(int)block_size,
        1.0f/sqrtf((float)D),causal);
    } else {
      dim3 grid(B*SQ*HQ), block(THREADS);
      hipLaunchKernelGGL((mla_decode_v2_kernel<scalar_t>), grid, block, 0, stream,
        q.data_ptr<scalar_t>(), table.data_ptr<int>(), cache.data_ptr<scalar_t>(),
        lengths.data_ptr<int>(), out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
        B,SQ,HQ,(int)h_kv,D,(int)dv,(int)max_blocks,(int)block_size,
        1.0f/sqrtf((float)D),causal);
    }
  });
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {out,lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
