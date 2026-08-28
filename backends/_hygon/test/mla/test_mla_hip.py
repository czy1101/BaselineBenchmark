import torch
from mla_hygon_reference import flash_mla_hygon_reference
from mla_hip import flash_mla_hygon

def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one masked Hygon device is required")
    for dtype in (torch.float16, torch.bfloat16):
        B,SQ,HQ,HKV,D,DV = 2,1,16,1,576,512
        bs=64; lens=[65,127]; mp=128
        q=torch.randn(B,SQ,HQ,D,device='cuda',dtype=dtype)
        tab=torch.arange(B*(mp//bs),device='cuda',dtype=torch.int32).view(B,-1)
        cache=torch.randn(tab.numel(),bs,HKV,D,device='cuda',dtype=dtype)
        ls=torch.tensor(lens,device='cuda',dtype=torch.int32)
        r,_=flash_mla_hygon_reference(q,tab,cache,mp,bs,B,SQ,ls,HQ,HKV,D,DV,True)
        y,_=flash_mla_hygon(q,tab,cache,mp,bs,ls,HQ,HKV,D,DV,True)
        err=(y.float()-r).square().mean().sqrt().item()
        print(dtype, 'rmse', err)
        assert err < (3e-2 if dtype == torch.float16 else 5e-2)
    print('MLA HIP v1 correctness: PASS')

if __name__ == '__main__': main()
