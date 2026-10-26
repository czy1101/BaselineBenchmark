#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <tuple>

namespace {
constexpr int WAVE = 64;
constexpr int THREADS = 128;
constexpr int WAVES = THREADS / WAVE;
constexpr int PREP_THREADS = 256;

template <int WIDTH>
__device__ __forceinline__ float wave_sum(float x) {
  #pragma unroll
  for (int d = WIDTH / 2; d > 0; d >>= 1) x += __shfl_down(x, d, WIDTH);
  return x;
}

template <typename scalar_t, int KSIZE>
__global__ __launch_bounds__(THREADS)
void gla_forward_wave(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const float* __restrict__ h0,
    scalar_t* __restrict__ o,
    float* __restrict__ ht,
    int B, int T, int H, int V, float scale, bool has_h0) {
  constexpr int ITEMS = KSIZE / WAVE;
  const int lane = threadIdx.x & (WAVE - 1);
  const int wave = threadIdx.x / WAVE;
  const int vi = blockIdx.x * WAVES + wave;
  const int hi = blockIdx.y;
  const int bi = blockIdx.z;
  const bool active = vi < V;
  float state[ITEMS];

  #pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int ki = lane + item * WAVE;
    const int si = ((bi * H + hi) * KSIZE + ki) * V + vi;
    state[item] = (active && has_h0) ? h0[si] : 0.0f;
  }

  #pragma unroll 1
  for (int ti = 0; ti < T; ++ti) {
    const int bk = ((bi * T + ti) * H + hi) * KSIZE;
    const int bv = ((bi * T + ti) * H + hi) * V;
    const float vv = active ? static_cast<float>(v[bv + vi]) : 0.0f;
    float dot = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * WAVE;
      const float kval = static_cast<float>(k[bk + ki]);
      state[item] = fmaf(kval, vv,
          state[item] * expf(static_cast<float>(g[bk + ki])));
      dot = fmaf(static_cast<float>(q[bk + ki]), state[item], dot);
    }
    dot = wave_sum<WAVE>(dot);
    if (active && lane == 0) o[bv + vi] = static_cast<scalar_t>(dot * scale);
  }

  if (active) {
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * WAVE;
      const int si = ((bi * H + hi) * KSIZE + ki) * V + vi;
      ht[si] = state[item];
    }
  }
}

template <typename scalar_t, int KSIZE, int GROUP>
__global__ __launch_bounds__(THREADS)
void gla_forward_shared_k(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const scalar_t* __restrict__ g,
    const float* __restrict__ h0, scalar_t* __restrict__ o,
    float* __restrict__ ht, int B, int T, int H, int V,
    float scale, bool has_h0) {
  static_assert(KSIZE % GROUP == 0, "invalid subgroup mapping");
  constexpr int ITEMS=KSIZE/GROUP;
  constexpr int GROUPS=THREADS/GROUP;
  const int tid=threadIdx.x, subgroup=tid/GROUP, lane=tid&(GROUP-1);
  const int vi=blockIdx.x*GROUPS+subgroup, hi=blockIdx.y, bi=blockIdx.z;
  const bool active=vi<V;
  __shared__ float sq[KSIZE], sk[KSIZE], sd[KSIZE];
  float state[ITEMS];
  #pragma unroll
  for(int item=0;item<ITEMS;++item){
    int ki=lane+item*GROUP;
    int si=((bi*H+hi)*KSIZE+ki)*V+vi;
    state[item]=(active&&has_h0)?h0[si]:0.0f;
  }
  #pragma unroll 1
  for(int ti=0;ti<T;++ti){
    int bk=((bi*T+ti)*H+hi)*KSIZE;
    for(int ki=tid;ki<KSIZE;ki+=THREADS){
      sq[ki]=static_cast<float>(q[bk+ki]);
      sk[ki]=static_cast<float>(k[bk+ki]);
      sd[ki]=expf(static_cast<float>(g[bk+ki]));
    }
    __syncthreads();
    int bv=((bi*T+ti)*H+hi)*V;
    float vv=active?static_cast<float>(v[bv+vi]):0.0f;
    float dot=0.0f;
    #pragma unroll
    for(int item=0;item<ITEMS;++item){
      int ki=lane+item*GROUP;
      state[item]=fmaf(sk[ki],vv,state[item]*sd[ki]);
      dot=fmaf(sq[ki],state[item],dot);
    }
    dot=wave_sum<GROUP>(dot);
    if(active&&lane==0)o[bv+vi]=static_cast<scalar_t>(dot*scale);
    __syncthreads();
  }
  if(active){
    #pragma unroll
    for(int item=0;item<ITEMS;++item){
      int ki=lane+item*GROUP;
      int si=((bi*H+hi)*KSIZE+ki)*V+vi;
      ht[si]=state[item];
    }
  }
}

// Fuses the tensor-layout conversion and all elementwise work required by
// BT64 Chunk/WY. One block owns one (batch, head, chunk), while each lane
// independently scans one or more D columns through the 64-token chunk.
template <typename scalar_t>
__global__ __launch_bounds__(PREP_THREADS)
void gla_chunk_prepare_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const scalar_t* __restrict__ g,
    at::BFloat16* __restrict__ qg, at::BFloat16* __restrict__ kg,
    at::BFloat16* __restrict__ vc, float* __restrict__ end_decay,
    int B, int T, int H, int D, int NT) {
  const int ci=blockIdx.x, hi=blockIdx.y, bi=blockIdx.z;
  const int bh=bi*H+hi, t0=ci*64;
  for(int di=threadIdx.x;di<D;di+=PREP_THREADS){
    float G=0.0f;
    #pragma unroll
    for(int ri=0;ri<64;++ri){
      const int ti=t0+ri;
      const int dst=(((bh*NT+ci)*64+ri)*D+di);
      if(ti<T){
        const int src=(((bi*T+ti)*H+hi)*D+di);
        G+=static_cast<float>(g[src]);
        const float ep=expf(G), en=expf(-G);
        qg[dst]=static_cast<at::BFloat16>(static_cast<float>(q[src])*ep);
        kg[dst]=static_cast<at::BFloat16>(static_cast<float>(k[src])*en);
        vc[dst]=static_cast<at::BFloat16>(v[src]);
      }else{
        qg[dst]=static_cast<at::BFloat16>(0.0f);
        kg[dst]=static_cast<at::BFloat16>(0.0f);
        vc[dst]=static_cast<at::BFloat16>(0.0f);
      }
    }
    end_decay[(bh*NT+ci)*D+di]=expf(G);
  }
}


void check(const torch::Tensor& x, const char* n) {
  TORCH_CHECK(x.is_cuda(), n, " must be on Hygon HIP");
  TORCH_CHECK(x.is_contiguous(), n, " must be contiguous");
}
}

std::tuple<torch::Tensor, torch::Tensor> gla_forward_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    c10::optional<torch::Tensor> initial_state, double scale) {
  check(q,"q"); check(k,"k"); check(v,"v"); check(g,"g");
  TORCH_CHECK(q.dim()==4 && v.dim()==4, "inputs must be rank 4");
  TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==g.sizes(), "q/k/g mismatch");
  TORCH_CHECK(q.size(0)==v.size(0) && q.size(1)==v.size(1) && q.size(2)==v.size(2),
              "v B/T/H mismatch");
  TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==v.scalar_type() &&
              q.scalar_type()==g.scalar_type(), "input dtype mismatch");
  TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,
              "HIP v1 supports fp16/bf16");
  const int B=q.size(0), T=q.size(1), H=q.size(2), K=q.size(3), V=v.size(3);
  TORCH_CHECK(K==64 || K==128 || K==256 || K==512, "K must be 64/128/256/512");
  const float* h0=nullptr; bool has_h0=initial_state.has_value() && initial_state->defined();
  if (has_h0) {
    check(*initial_state,"initial_state");
    TORCH_CHECK(initial_state->scalar_type()==at::kFloat, "initial_state must be fp32");
    TORCH_CHECK(initial_state->dim()==4 && initial_state->size(0)==B &&
                initial_state->size(1)==H && initial_state->size(2)==K &&
                initial_state->size(3)==V, "h0 shape mismatch");
    h0=initial_state->data_ptr<float>();
  }
  auto o=torch::empty_like(v);
  auto ht=torch::empty({B,H,K,V},q.options().dtype(torch::kFloat));
  // Measured BW mapping: group8 for K<=256; group16 avoids excessive live
  // state and scratch pressure for K512 without sacrificing throughput.
  const int group=(K==512)?16:8;
  const int groups_per_block=THREADS/group;
  dim3 block(THREADS), grid((V+groups_per_block-1)/groups_per_block,H,B);
  hipStream_t stream=c10::hip::getCurrentHIPStream().stream();
#define ARGS grid,block,0,stream,q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),h0,o.data_ptr<scalar_t>(),ht.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_h0
#define LAUNCH64() hipLaunchKernelGGL((gla_forward_shared_k<scalar_t,64,8>),ARGS)
#define LAUNCH128() hipLaunchKernelGGL((gla_forward_shared_k<scalar_t,128,8>),ARGS)
#define LAUNCH256() hipLaunchKernelGGL((gla_forward_shared_k<scalar_t,256,8>),ARGS)
#define LAUNCH512() hipLaunchKernelGGL((gla_forward_shared_k<scalar_t,512,16>),ARGS)
  AT_DISPATCH_SWITCH(q.scalar_type(),"gla_forward_hip",
    AT_DISPATCH_CASE(at::ScalarType::Half,[&]{if(K==64){LAUNCH64();}else if(K==128){LAUNCH128();}else if(K==256){LAUNCH256();}else{LAUNCH512();}})
    AT_DISPATCH_CASE(at::ScalarType::BFloat16,[&]{if(K==64){LAUNCH64();}else if(K==128){LAUNCH128();}else if(K==256){LAUNCH256();}else{LAUNCH512();}}));
#undef LAUNCH64
#undef LAUNCH128
#undef LAUNCH256
#undef LAUNCH512
#undef ARGS
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {o,ht};
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
gla_chunk_prepare_hip(torch::Tensor q, torch::Tensor k,
                      torch::Tensor v, torch::Tensor g) {
  check(q,"q"); check(k,"k"); check(v,"v"); check(g,"g");
  TORCH_CHECK(q.dim()==4 && q.sizes()==k.sizes() && q.sizes()==v.sizes() &&
              q.sizes()==g.sizes(), "chunk_prepare expects equal BTHD tensors");
  TORCH_CHECK(q.scalar_type()==k.scalar_type() && q.scalar_type()==v.scalar_type() &&
              q.scalar_type()==g.scalar_type(), "chunk_prepare dtype mismatch");
  TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,
              "chunk_prepare supports fp16/bf16");
  const int B=q.size(0),T=q.size(1),H=q.size(2),D=q.size(3),NT=(T+63)/64;
  auto bf=q.options().dtype(torch::kBFloat16);
  auto qg=torch::empty({B*H,NT,64,D},bf);
  auto kg=torch::empty_like(qg),vc=torch::empty_like(qg);
  auto end_decay=torch::empty({B*H,NT,D},q.options().dtype(torch::kFloat));
  dim3 block(PREP_THREADS),grid(NT,H,B);
  hipStream_t stream=c10::hip::getCurrentHIPStream().stream();
#define PREP_ARGS grid,block,0,stream,q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),qg.data_ptr<at::BFloat16>(),kg.data_ptr<at::BFloat16>(),vc.data_ptr<at::BFloat16>(),end_decay.data_ptr<float>(),B,T,H,D,NT
  AT_DISPATCH_SWITCH(q.scalar_type(),"gla_chunk_prepare_hip",
    AT_DISPATCH_CASE(at::ScalarType::Half,[&]{hipLaunchKernelGGL((gla_chunk_prepare_kernel<scalar_t>),PREP_ARGS);})
    AT_DISPATCH_CASE(at::ScalarType::BFloat16,[&]{hipLaunchKernelGGL((gla_chunk_prepare_kernel<scalar_t>),PREP_ARGS);}));
#undef PREP_ARGS
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {qg,kg,vc,end_decay};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("forward",&gla_forward_hip,"GLA HIP forward");
  m.def("chunk_prepare",&gla_chunk_prepare_hip,"GLA BT64 HIP chunk preparation");
}
