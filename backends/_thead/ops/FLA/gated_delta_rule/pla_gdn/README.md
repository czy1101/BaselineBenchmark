# PLA GDN 对比实现

本目录保存与 FlagAttention 平头哥 GDN 算子对比使用的 PLA 源码、必要头文件及当前设备上的编译产物。运行和构建不依赖之前 `/tmp` 下的源码或 `/workspace/pla` 中的头文件。

## 三条对比路径

| 对比目标 | Python / 主机入口 | 核心源码 |
| --- | --- | --- |
| Chunk GDN | `official/pla/prefill/flashqla/ops.py` | `official/src/prefill/flashqla/kernels/` |
| 普通 forward，预计算 g/beta | `precomputed/host.cpp` | `precomputed/src/decode/k-last/fused_sigmoid_gating_k_last_kernel.cuh` |
| Packed decode，官方融合入口 | `official/pla/decode/__init__.py`；`official/src/decode/k-last/fused_sigmoid_gating_cuda_k_last_packed.cpp` | `official/src/decode/k-last/fused_sigmoid_gating_k_last_packed_kernel.cuh` |

`official` 的非生成内核源码保持官方版本。`precomputed` 替换了门控输入和计算部分，其余递推、访存、并行结构及启动参数保持原实现。源码版本为 `t-head/pla` 的 `ace8581717b080aafdd3c9604f582d35da8e6355`。许可证和 NOTICE 在 `official/` 下，原文件的版权声明均保留。

## 导入

```python
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.pla_gdn import (
    chunk_forward,
    recurrent_precomputed,
    packed_decode,
)
```

这三个名称延迟加载各自的本地扩展，不登记为 FlagAttention 后端算子，也不改变现有平头哥实现的分发。它们保留 PLA 的调用形式：

- `chunk_forward`：PLA 的五项返回值 `(g_cum, A, out, None, final_state)`；预计算 g/beta 要求 FP32。
- `recurrent_precomputed(q,k,v,g,beta,state,indices,scale,norm)`：返回输出 Tensor，并原地更新状态。当前适配版支持本次普通 forward benchmark 的 `T=1、K=V=128`，Q/K/V、beta 和状态均为 BF16，g 为 FP32；状态布局为 `[slots,HV,V,K]`，索引 0 为无效槽。
- `packed_decode`：保留官方融合入口的参数顺序和 Tensor 返回值，状态原地更新。对齐 vLLM 的无效槽约定时传入 `is_sglang=False`。

`benchmark/test_gated_delta_rule_benchmark.py` 从本目录加载 `chunk_forward`，无需设置额外 PYTHONPATH。原来本地 12 组和共同 72 组的顺序、输入及计时方法保持不变。

## 文件及扩展

```text
pla_gdn/
├── __init__.py
├── official/
│   ├── pla/                    # Python 入口及已编译的 decode、FlashQLA 扩展
│   ├── src/                    # 原生内核、主机绑定及公共头文件
│   ├── 3rdparty/actlize/        # 构建 FlashQLA 所需的头文件及许可证
│   ├── lut/                    # 官方调优表
│   ├── scripts/decode/generate_kernel_instantiations.py
│   ├── setup.py
│   ├── ppu_build.py
│   ├── LICENSE
│   └── NOTICE
├── precomputed/
│   ├── src/                    # 适配版的独立头文件与源码
│   ├── pre_*.cu                # 官方启动配置对应的适配版实例
│   ├── host.cpp
│   ├── setup.py
│   ├── ppu_build.py
│   └── pla_precomputed*.so
├── precomputed-gates.patch
├── ppu_build-compat.patch
├── manifest.json
└── README.md
```

当前 `.so` 对应 PPU-ZW810E、PyTorch 2.10.0、Python 3.12。依赖系统已安装的 PyTorch 和 PPU SDK。二进制和构建缓存在本目录的 `.gitignore` 中；源码和头文件保持可查看、可重建。没有指向临时目录的符号链接。

## 重新构建

在本目录下执行。下列环境变量只用于当前 shell，不安装或升级任何依赖：

```bash
export MAX_JOBS=2 HGCC_THREADS=2
export PPU_SDK_DIR=/usr/local/PPU_SDK
export PYTHONDONTWRITEBYTECODE=1

cd official
export PPU_ACTLIZE_DIR="$PWD/3rdparty/actlize"
python scripts/decode/generate_kernel_instantiations.py \
  --tuned-csv lut/tuned_result_PPU-ZW810E.csv
python setup.py build_ext --inplace --namespace=pla.decode
python setup.py build_ext --inplace --namespace=pla.prefill.flashqla

cd ../precomputed
python setup.py build_ext --inplace
```

普通 forward 适配版的实例和 dispatch 已随源码保存；它使用固定的上述官方 LUT，没有额外启动搜索。Packed decode 在共同 `(H,HV)=(4,12)` 的配置上仍按官方 LUT 未命中时的默认参数执行。
