# GDN2 海光 HIP 适配说明

## 1. 项目概述

本版本将官方 `chunk_gdn2` 的核心功能适配到海光 DCU/HIP 环境，使用 PyTorch C++ Extension 加载纯 HIP C++ Kernel，支持 FP16 和 BF16。

主要目标：

- 在海光 HIP PyTorch 上提供 GDN2 公共调用接口；
- 保持官方 chunk GDN2 的输出和状态语义；
- 支持固定长度、chunk 扫描、状态输出和 packed varlen；
- 提供可重复的正确性与性能测试流程。

## 2. 最终提交文件

```text
gdn2_hygon/
├── benchmarks/
│   └── benchmark_gdn2_hygon.py
├── ops/
│   └── gdn2/
│       ├── gdn2_hip.py
│       └── gdn2_hip_kernel.cu
└── test/
    ├── gdn2_hygon_reference.py
    ├── test_gdn2_hip.py
    ├── test_gdn2_hip_features.py
    ├── test_gdn2_chunk_cumsum_hip.py
    ├── test_gdn2_chunk_factors_hip.py
    ├── test_gdn2_chunk_scores_hip.py
    └── test_gdn2_chunk_solve_hip.py
```

实验性文件、PyTorch 原型、性能日志、CSV 和缓存文件不属于最终算子提交内容。参考实现属于测试 oracle，应放在 `test/gdn2/`，不放在生产 `ops/gdn2/`。

## 3. 实现组成

### 3.1 Python API

`ops/gdn2/gdn2_hip.py` 负责：

- 编译和缓存 HIP Extension；
- 输入 shape、dtype、device 和 contiguous 检查；
- Q/K L2 normalization；
- gate、safe gate、`A_log`、`dt_bias` 处理；
- `initial_state` 和 `state_v_first` 转换；
- 固定长度和 packed varlen dispatch；
- `output_final_state` 和 intermediate state 返回。

公共入口：

```python
from gdn2_hip import chunk_gdn2_hip

output, final_state = chunk_gdn2_hip(
    q, k, v, g, b, w,
    output_final_state=True,
)
```

兼容别名：

```python
chunk_gdn2 = chunk_gdn2_hip
```

### 3.2 HIP Kernel

`ops/gdn2/gdn2_hip_kernel.cu` 使用 `hipcc`/PyTorch HIP Extension 编译，核心计算包括：

1. gate decay；
2. state 与 key 的 correction；
3. beta/write/value 更新；
4. query 与 state 的输出计算；
5. FP32 final state 写回。

支持的数据类型：

```text
torch.float16
torch.bfloat16
```

状态使用 FP32，输出保持输入 dtype。

## 4. 功能覆盖

| 功能 | 支持情况 |
|---|---|
| FP16 | 支持 |
| BF16 | 支持 |
| 固定长度序列 | 支持 |
| packed varlen / `cu_seqlens` | 支持 |
| Q/K L2 normalization | 支持 |
| raw gate | 支持 |
| safe gate | 支持 |
| `A_log` | 支持 |
| `dt_bias` | 支持 |
| initial state | 支持 |
| final state | 支持 |
| `state_v_first` | 支持 |
| intermediate states | 支持 |
| GQA/MQA | 支持，要求 `HV % H == 0` |

## 5. 环境要求

```text
PyTorch: 2.9.0 或兼容 HIP PyTorch
HIP: 6.3 或兼容版本
设备: 海光 DCU/BW
```

运行前设置单卡环境：

```bash
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
```

## 6. 正确性测试

进入提交目录：

```bash
cd /workspace/hy/compare/gdn2_hygon
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export PYTHONPATH=$PWD/ops/gdn2:$PWD/test:$PYTHONPATH
```

测试文件依赖参考 oracle，请确认文件存在：

```bash
test -f test/gdn2/gdn2_hygon_reference.py
```

清理旧 Extension 缓存：

```bash
rm -rf /root/.cache/torch_extensions/py310_cpu/gdn2_hygon_hip_ext*
```

基础测试：

```bash
python test/test_gdn2_hip.py
```

公共功能测试：

```bash
python test/test_gdn2_hip_features.py
```

Chunk 原语测试：

```bash
python test/test_gdn2_chunk_cumsum_hip.py
python test/test_gdn2_chunk_factors_hip.py
python test/test_gdn2_chunk_scores_hip.py
python test/test_gdn2_chunk_solve_hip.py
```

验收标准：

```text
GDN2 HIP correctness: PASS
GDN2 HIP public feature tests: PASS
GDN2 HIP BT64 chunk cumsum: PASS
GDN2 HIP BT64 chunk factors: PASS
GDN2 HIP BT64 chunk scores: PASS
GDN2 HIP BT64 chunk solve: PASS
```

## 7. 性能测试

完整默认 shape：

```bash
python benchmarks/benchmark_gdn2_hygon.py \
  --implementation hip \
  --dtype both \
  --warmup 10 \
  --iterations 30
```

重点大 shape：

```bash
python benchmarks/benchmark_gdn2_hygon.py \
  --implementation hip \
  --dtype both \
  --shape 1 8192 96 128 128 \
  --shape 2 2048 16 256 512 \
  --shape 8 2048 32 256 256 \
  --warmup 10 \
  --iterations 30
```

单独测试 FP16：

```bash
python benchmarks/benchmark_gdn2_hygon.py \
  --implementation hip \
  --dtype fp16 \
  --warmup 10 \
  --iterations 50
```

Benchmark 输出字段包括：

- `mean_ms`、`p50_ms`、`p95_ms`、`min_ms`；
- `tokens_per_second`；
- `peak_memory_mib`；
- `output_nan`、`state_nan`；
- 与 HY FLA/HY optimized 的时间比。

## 8. 性能结果解读

GDN2 主要包含沿序列方向的状态递推。简单 recurrent Kernel 的复杂度接近：

```text
O(B × T × H × K × V)
```

官方 chunk/WY 实现会把多个 token 组织成 chunk，通过矩阵运算、三角求解和 chunk state update 提高并行度。因此：

- 短序列和小 batch 可能受到 Kernel launch 和 occupancy 影响；
- 大 `K/V` shape 主要受状态读写和矩阵计算吞吐影响；
- FP16/BF16 输出精度不同，但状态保持 FP32；
- 性能比较必须使用相同 warmup、iterations、shape 和单卡环境。

## 9. 不应提交的文件

以下内容只用于本地开发或分析：

```text
__pycache__/
*.csv
*.log
*.json
gdn2_affine_scan.py
gdn2_chunk14.py
gdn2_chunked_torch.py
profile_gdn2_chunked.py
```

`gdn2_hygon_reference.py` 不属于生产算子，但为了让基础正确性和公共功能测试可独立复现，应保留在 `test/gdn2/`。

## 10. 最终验收顺序

```bash
python test/test_gdn2_hip.py
python test/test_gdn2_hip_features.py
python test/test_gdn2_chunk_cumsum_hip.py
python test/test_gdn2_chunk_factors_hip.py
python test/test_gdn2_chunk_scores_hip.py
python test/test_gdn2_chunk_solve_hip.py
python benchmarks/benchmark_gdn2_hygon.py --implementation hip --dtype both --warmup 10 --iterations 30
```

只有在所有正确性测试通过、且 `output_nan=False`、`state_nan=False` 时，才记录最终性能结果。
