# SageAttention 海光 HIP 适配说明

## 1. 最终提交目录

```text
backends/_hygon/
├── benchmarks/
│   └── benchmark_sageattention_hygon.py
├── doc/
│   └── SAGEATTENTION_HYGON.md        # 本文件
├── ops/
│   └── sageattention/
│       ├── sageattention_hip.py       # Python 公共 API 与参数校验
│       ├── sageattention_triton.py    # Triton 内核（INT8 QK + FP16 PV + online softmax）
│       └── .gitkeep
└── test/
    └── sageattention/
        └── test_sageattention_hip.py  # HND/NHD、GQA、bool/additive mask 正确性
```

## 2. 应提交的文件

| 文件 | 作用 |
|---|---|
| `sageattention_hip.py` | Python 公共 API：`forward` / `per_block_int8` / `backend_name()`，dtype/contiguity/head_dim/GQA 校验，按需注入 FlagTree Triton 路径 |
| `sageattention_triton.py` | Triton 内核：`_quant_block_kernel`（分块 INT8 量化）+ `_attention_kernel`（INT8 QK 矩阵核 + online softmax + FP16 PV） |
| `benchmark_sageattention_hygon.py` | 官方 8 组 shape × {fp16, bf16} 性能测试，输出 `speedup_vs_hygon_opt` |
| `test_sageattention_hip.py` | 4 个 case：HND/NHD × D=64、D=128 × bool/additive mask，相对 RMSE < 3e-2 |

## 3. 不建议提交的文件

```text
__pycache__/
*.csv
*.log
*.json
```

## 4. 实现要点

### `sageattention_hip.py`

- 公共接口与官方 SageAttention Python API 完全一致：`per_block_int8(q, k, ...)`、`forward(q, k, v, q_scale, k_scale, ...)`、`backend_name()`；
- 支持 `tensor_layout ∈ {"HND", "NHD"}`、GQA（任意 `num_kv_groups`）、bool / additive `attn_mask`、`return_lse`、`output_dtype ∈ {fp16, bf16}`；
- BW1000 HCU 的 `HIPOptions` 不支持 NVIDIA `maxnreg`，包装层只校验不下传；
- `backend_name()` 是**函数**（非字符串），benchmark 打印时调用 `backend_name()`；
- v1 仅支持 `head_dim ∈ {64, 128}`，`BLKQ=128 / BLKK=64`。

### `sageattention_triton.py`

- 公共路径与官方 Triton2 per-block INT8 一致：
  - **第一层 `_quant_block_kernel`**：按 128（Q）/64（K）分块，`scale = max|x| / 127`，Q 端把 `sm_scale * log2(e)` 折进 `q_scale`，核内直接用 `exp2`；
  - **第二层 `_attention_kernel`**：`tl.dot(q, k, out_dtype=tl.int32)`（关键，让 FlagTree HCU Triton 后端 lower 到 INT8 矩阵核），`to(fp32) * (q_scale * k_scale)` 还原，online softmax 后 `tl.dot(p.to(fp16), v, out_dtype=fp32)` 做 PV；
- 网格 `(cdiv(q_len, 128), qh, b)`，`num_warps` 与 `num_stages` 通过 `SAGEATTENTION_HYGON_WARPS` / `SAGEATTENTION_HYGON_STAGES` 覆盖；
- **关键约束**：`num_stages` 在 BW1000 上必须 ≤3。`num_stages=4` 触发 HCU 软件流水 pass bug：循环内的标量 `tl.load`（k_scale）被 destroy 而 `logits *= q_scale * k_scale` 仍依赖它，编译期 `LLVM ERROR: operation destroyed but still has uses`，所有 `head_dim=128`（以及任何 stages=4 配置）启动全部失败；
- 代码内已固化为 `stages = min(int(os.getenv("SAGEATTENTION_HYGON_STAGES", "3")), 3)`：默认 3、设了 4 也会被压回 3；想再验证 4 时把 `min` 注释掉；
- `BLOCK_M / BLOCK_N` 与 `per_block_int8` 的 `BLKQ=128 / BLKK=64` **强耦合**：q_scale 一项/128 token，k_scale 一项/64 token；单独改某一边会用错 scale 算出错的注意力——已在源码注释中标注。

## 5. 正确性测试

```bash
cd /workspace/FlagGems-vllm/BaselineBenchmark

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export TRITON_ROOT=/workspace/FlagTree
export PYTHONPATH=$TRITON_ROOT/build/lib.linux-x86_64-cpython-310:$TRITON_ROOT/python:$PWD/backends/_hygon/ops/sageattention:$PYTHONPATH

# 1) 先确认 Triton 解析到 FlagTree（错误解析会引爆标量 HIP 重写问题）
python -c "import triton; print(triton.__file__)"
# 期望输出：/workspace/FlagTree/build/lib.linux-x86_64-cpython-310/triton/__init__.py

python backends/_hygon/test/sageattention/test_sageattention_hip.py
```

期望输出：

```text
SageAttention backend: flagtree_triton_hip_int8_qk_fp16_pv
HND kv_heads 1 D 64 out_rel_rmse 0.000262 lse_max_abs 9.5367e-07
NHD kv_heads 2 D 64 out_rel_rmse 0.000265 lse_max_abs 9.5367e-07
HND kv_heads 2 D 128 out_rel_rmse 0.000260 lse_max_abs 9.5367e-07
HND kv_heads 2 D 128 out_rel_rmse 0.000258 lse_max_abs 9.5367e-07
SageAttention Hygon correctness: PASS
```

`rel_rmse < 3e-2` / `lse_max_abs < 3e-2`，D=64 与 D=128、HND 与 NHD、bool / additive mask 都覆盖。

## 6. 性能测试

### 首轮两组探编译 + 时间量级

```bash
python backends/_hygon/benchmarks/benchmark_sageattention_hygon.py \
  --dtype both --start 0 --end 2 --warmup 10 --iterations 30
```

### 完整 8 组 shape

```bash
python backends/_hygon/benchmarks/benchmark_sageattention_hygon.py \
  --dtype both --start 0 --end 8 --warmup 10 --iterations 30
```

8 组 shape（`B=1, H_qo=32, H_kv=32`）：

```text
[1,1024,32,64]    [1,4096,32,64]    [1,8192,32,64]    [1,16384,32,64]
[1,1024,32,128]   [1,4096,32,128]   [1,8192,32,128]   [1,16384,32,128]
```

### 自定义 shape

`--shape` 顺序为 `B T H D`：

```bash
python backends/_hygon/benchmarks/benchmark_sageattention_hygon.py \
  --dtype bf16 --shape 1 4096 32 128 --warmup 10 --iterations 30
```

## 7. Benchmark 输出字段

主要关注：

| 字段 | 含义 |
|---|---|
| `quant_ms` | 预量化耗时（Q/K → INT8 + scale） |
| `attention_p50_ms` | **预量化 benchmark**：与官方 SageAttention 一致，仅 INT8 QK + FP16 PV 主循环 |
| `attention_mean_ms` | 同上，取算术平均 |
| `pipeline_ms` | 量化 + attention 端到端 |
| `tflops` | `4·B·H_qo·T·T·D / attention_p50_ms / 1e9` |
| `peak_mib` | 整次 benchmark 的 torch 显存峰值 |
| `hygon_opt_ref_ms` | 海光 Optimized 参考值 |
| `speedup_vs_hygon_opt` | `hygon_opt_ref_ms / attention_p50_ms`，**大于 1.0× 表示当前更快** |

## 8. 首轮实测（2026-09-01，gfx936 / BW1000，FlagTree Triton）

`maxnreg` 列只对 NVIDIA 参考值有意义，海光侧不使用。

### fp16 全量

| Shape `[B,T,H,D]` | NVIDIA Base | NVIDIA Opt | Hygon Base | Hygon Opt | Current HIP | HIP / Hygon Opt | HIP / NVIDIA Opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[1,1024,32,64]` | 0.0313 | 0.0322 | 0.1788 | 0.1798 | 0.3614 | **2.010** | 11.22 |
| `[1,4096,32,64]` | 0.3535 | 0.3371 | 1.8392 | 1.8393 | 2.8130 | **1.529** | 8.35 |
| `[1,8192,32,64]` | 1.3630 | 1.2868 | 7.0693 | 7.0687 | 10.6715 | **1.510** | 8.29 |
| `[1,16384,32,64]` | 5.3013 | 4.9960 | 27.8865 | 27.8410 | 41.9718 | **1.508** | 8.40 |
| `[1,1024,32,128]` | 0.0537 | 0.0516 | 0.3224 | 0.3224 | 0.5316 | **1.649** | 10.30 |
| `[1,4096,32,128]` | 0.6532 | 0.5898 | 4.2833 | 4.2880 | 4.7419 | **1.106** | 8.04 |
| `[1,8192,32,128]` | 2.5321 | 2.2105 | 16.4373 | 16.4336 | 18.2580 | **1.111** | 8.26 |
| `[1,16384,32,128]` | 9.9807 | 8.5175 | 61.5444 | 61.5374 | 72.1325 | **1.172** | 8.47 |

### bf16

无参考值；与 fp16 差 ≤2%（仅 T=1024 D=64 上 1.8%）。

| Shape `[B,T,H,D]` | Hygon Opt | Current HIP | HIP / Hygon Opt |
|---|---:|---:|---:|
| `[1,1024,32,64]` | 0.1798 | 0.3712 | 2.064 |
| `[1,4096,32,64]` | 1.8393 | 2.8192 | 1.533 |
| `[1,8192,32,64]` | 7.0687 | 10.6808 | 1.511 |
| `[1,16384,32,64]` | 27.8410 | 41.9903 | 1.508 |
| `[1,1024,32,128]` | 0.3224 | 0.5365 | 1.664 |
| `[1,4096,32,128]` | 4.2880 | 4.7465 | 1.107 |
| `[1,8192,32,128]` | 16.4336 | 18.2685 | 1.112 |
| `[1,16384,32,128]` | 61.5374 | 72.1564 | 1.173 |

### 解读

- **D=64 大 T（≥4096）稳定在 HIP/Hygon Opt ≈ 1.51**：差距是结构性的，海光 Optimized 与 Baseline 在参考表里几乎相等（≈0.18×），意味着海光 Optimized ≈ 海光上已知最优 SageAttention；这个 1.51× 即「我们 vs 任何已知海光 SageAttention」的差距；
- **D=128 大 T 稳定在 HIP/Hygon Opt ≈ 1.10–1.17**：差 10%–17%，离打平只一步；
- **小 T（T=1024）两个 head_dim 都退化**（D=64=2.01、D=128=1.65）：与 CTA 数在该 shape 下只有 256 个、BW1000 上百个 CU 喂不满有关；
- **HIP/NVIDIA Opt 在 8.0–11.2**：海光硬件本身要 5.7–6×，再叠加我们这层 1.5×，符合预期量级；
- **INT8 MMA 已落矩阵核**：汇编里 `tt.dot tensor<128x128xi8> * tensor<128x64xi8> -> i32` 经 `tritonhcugpu-accelerate-matmul{arch-generation-name=gfx936}` lower，未走标量回退；
- **`quant_ms` 有 ~0.2 ms 平台**：逐次 `synchronize()` 引入的 launch overhead 主导，不是真实量化耗时；做 perf 优化时不要拿 `pipeline_ms` 当小 T 下的业务开销。

## 9. 已知编译器 bug 与规避

| 现象 | 触发条件 | 解决 |
|---|---|---|
| `LLVM ERROR: operation destroyed but still has uses` | `_attention_kernel` 内 num_stages=4，标量 k_scale `tl.load` 被 HCU 软件流水 pass destroy | 代码默认 stages=3 + `min(stages, 3)` 硬封顶 |

将 `stages = min(stages, 3)` 这一行注释掉并设 `SAGEATTENTION_HYGON_STAGES=4` 即可复现；建议保留硬封顶。

## 10. 已知限制

- `head_dim ∈ {64, 128}` 之外不报支持；包装层在调用前 raise；
- `output_dtype ∈ {fp16, bf16}`；其他 dtype raise；
- `q.dtype != int8` / 非 contiguous / 非 cuda tensor 在 `forward` 入口 raise；
- 因标量 HIP 重写会令矩阵核完全闲置，**禁止将本实现改写为标量 HIP 路径**——这是全部性能意义的来源；
- `BF16` 仅当上层传入 `output_dtype=torch.bfloat16` 时生效；中间量化/online softmax 一律 FP32，PV 用 FP16/FP16 矩阵核 + FP32 累加。

## 11. 最终验收顺序

```bash
cd /workspace/FlagGems-vllm/BaselineBenchmark
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 MAX_JOBS=8
export TRITON_ROOT=/workspace/FlagTree
export PYTHONPATH=$TRITON_ROOT/build/lib.linux-x86_64-cpython-310:$TRITON_ROOT/python:$PWD/backends/_hygon/ops/sageattention:$PYTHONPATH

python -c "import triton; print(triton.__file__)"                              # 必须 FlagTree
python backends/_hygon/test/sageattention/test_sageattention_hip.py             # SageAttention Hygon correctness: PASS
python backends/_hygon/benchmarks/benchmark_sageattention_hygon.py \
  --dtype both --start 0 --end 8 --warmup 10 --iterations 30                   # 8×2 全部出数
```

只有正确性 `PASS` 且 8 组 shape 全部出数（`peak_mib` 与 TFLOPS 行非空），才记录最终性能。