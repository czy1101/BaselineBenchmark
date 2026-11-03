// SPDX-License-Identifier: Apache-2.0
// Benchmark-only launcher around T-Head PLA's BSD-licensed k-last kernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "ppu_torch_stream.h"
#include "decode/k-last/k_last_shared_layout.h"
#include "decode/k-last/generated/tuned_dispatch.h"
#include "launchers.h"

std::vector<int> config(int n, int h, int hv, int max_tokens) {
    auto p = max_tokens > 1 ? get_tuned_k_last_mtp(n,h,hv,128,128,false)
                            : get_tuned_k_last(n,h,hv,128,128);
    return {p.bv,p.wn,p.sn,p.hv_per_cta};
}

torch::Tensor run(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state,
    torch::Tensor indices, torch::Tensor cu, std::optional<torch::Tensor> accepted,
    float scale, bool norm, int max_tokens) {
    c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16,
                "PLA precomputed supports FP16/BF16 activations");
    for (const auto& x : {q,k,v,g,beta,state,indices,cu})
        TORCH_CHECK(x.is_cuda() && x.device()==q.device(),"inputs must share one PPU");
    TORCH_CHECK(q.dim()==4 && k.sizes()==q.sizes() && q.size(0)==1 && q.size(3)==128,
                "packed Q/K must be [1,total,H,128]");
    TORCH_CHECK(v.dim()==4 && v.size(0)==1 && v.size(1)==q.size(1) && v.size(3)==128,
                "V must be [1,total,HV,128]");
    TORCH_CHECK(k.scalar_type()==q.scalar_type() && v.scalar_type()==q.scalar_type()
                && beta.scalar_type()==q.scalar_type() && g.scalar_type()==at::kFloat
                && state.scalar_type()==at::kFloat,"gate/activation/state dtype mismatch");
    const int h=q.size(2), hv=v.size(2), n=cu.numel()-1;
    TORCH_CHECK(h>0 && hv%h==0 && n>0 && max_tokens>=1 && max_tokens<=8,
                "invalid benchmark geometry; supported maximum length is 8");
    TORCH_CHECK(g.dim()==3 && beta.dim()==3 && g.size(0)==1 && g.size(1)==q.size(1)
                && g.size(2)==hv && beta.sizes()==g.sizes(),"scalar g/beta shape mismatch");
    for (const auto& x : {q,k,v})
        TORCH_CHECK(x.stride(3)==1 && x.stride(2)==128,"dense head/feature axes required");
    TORCH_CHECK(g.is_contiguous() && beta.is_contiguous(),"dense g/beta required");
    TORCH_CHECK(state.dim()==4 && state.size(1)==hv && state.size(2)==128
                && state.size(3)==128 && state.is_contiguous(),"dense [pool,HV,V,K] FP32 state required");
    TORCH_CHECK(indices.scalar_type()==at::kInt && cu.scalar_type()==at::kInt
                && cu.dim()==1 && cu.is_contiguous() && indices.size(0)==n,
                "contiguous-column INT32 index metadata required");
    const bool mtp=max_tokens>1;
    TORCH_CHECK(indices.dim()==(mtp?2:1) && indices.is_contiguous(),"index layout mismatch");
    TORCH_CHECK(!mtp || indices.size(1)>=max_tokens,"state table too narrow");
    if (accepted.has_value()) {
        TORCH_CHECK(mtp && accepted->is_cuda() && accepted->device()==q.device()
                    && accepted->scalar_type()==at::kInt && accepted->is_contiguous()
                    && accepted->numel()==n,"invalid accepted-token metadata");
    }
    auto p=mtp?get_tuned_k_last_mtp(n,h,hv,128,128,false):get_tuned_k_last(n,h,hv,128,128);
    Launcher launch=select_launcher(p.bv,p.wn,p.sn,p.hv_per_cta,
                                    q.scalar_type()==at::kBFloat16,mtp);
    TORCH_CHECK(launch,"official LUT configuration not compiled in this benchmark extension");
    const pla::KLastSharedLayout layout(p.sn,128,p.wn,p.bv,p.hv_per_cta,false,mtp,max_tokens);
    auto out=torch::empty(v.sizes(),v.options());
    launch(q.data_ptr(),k.data_ptr(),v.data_ptr(),g.data_ptr<float>(),beta.data_ptr(),
           out.data_ptr(),state.data_ptr<float>(),indices.data_ptr<int32_t>(),
           cu.data_ptr<int32_t>(),accepted.has_value()?accepted->data_ptr<int32_t>():nullptr,
           n,h,hv,max_tokens,q.stride(1),k.stride(1),v.stride(1),g.stride(1),
           beta.stride(1),state.stride(0),mtp?indices.stride(0):0,scale,norm,
           layout.total_bytes,pla::current_stream());
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("run",&run);m.def("config",&config);}
