# GLA 海光 HIP 适配说明

## 1. 最终提交目录

```text
gla_hygon/
├── benchmarks/
│   └── benchmark_gla_hygon.py
├── ops/
│   └── gla/
│       ├── gla_hip.py
│       ├── gla_hip_kernel.cu
│       └── gla_hygon_reference.py
└── test/
    └── gla/
        ├── gla_chunk_dtk.py
        ├── test_gla_hip.py
        └── test_gla_chunk_dtk.py
```

## 2. 应提交的文件

| 文件 | 作用 |
|---|---|
| `gla_hip.py` | GLA Python 公共 API、输入处理和 HIP Extension 加载 |
| `gla_hip_kernel.cu` | 正式纯 HIP C++ Kernel |
| `benchmark_gla_hygon.py` | 官方 shape 性能测试和 HY/H800 对比 |
| `test_gla_hip.py` | HIP 版本基础正确性测试 |
| `test_gla_chunk_dtk.py` | chunk/DTK 路径功能测试 |

## 3. 不建议提交的文件

```text
gla_hip_kernel.hip          # 编译中间文件/备份文件
__pycache__/
*.csv
*.log
*.json
```

当前 `gla_hip.py` 顶层依赖 `gla_hygon_reference.py`，因此该文件必须保留在 `ops/gla/`。`test_gla_chunk_dtk.py` 直接依赖 `gla_chunk_dtk.py`，因此测试目录也必须保留该原型文件。

## 4. 实现职责

### `gla_hip.py`

- GLA 输入 shape 和 dtype 检查；
- FP16/BF16 dispatch；
- initial/final state 处理；
- `state_v_first` 转换；
- fixed-length 和 varlen 调度；
- HIP Extension JIT 编译和缓存。

### `gla_hip_kernel.cu`

- gate 衰减；
- Q/K/V 状态递推；
- FP32 状态累加；
- FP16/BF16 输出；
- GLA output 和 final state 计算。

## 5. 正确性测试

```bash
cd /workspace/hy/compare/gla_hygon

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export PYTHONPATH=$PWD/ops/gla:$PWD/test:$PYTHONPATH
```

确认兼容依赖存在：

```bash
test -f ops/gla/gla_hygon_reference.py
test -f test/gla/gla_chunk_dtk.py
```

清理旧扩展：

```bash
rm -rf /root/.cache/torch_extensions/py310_cpu/gla_hygon_hip_ext*
```

基础 HIP 正确性：

```bash
python test/test_gla_hip.py
```

Chunk/DTK 功能测试：

```bash
python test/test_gla_chunk_dtk.py
```

期望输出：

```text
GLA HIP correctness: PASS
GLA chunk/DTK correctness: PASS
```

## 6. 性能测试

### 官方三组 shape

```bash
python benchmarks/benchmark_gla_hygon.py \
  --implementation chunk \
  --dtype both \
  --warmup 10 \
  --iterations 30
```

官方主要 shape：

```text
B=4, T=4096, H=64, D=128
B=2, T=2048, H=16, D=512
B=8, T=2048, H=32, D=256
```

### 扩展 shape

```bash
python benchmarks/benchmark_gla_hygon.py \
  --implementation chunk \
  --dtype both \
  --shape 1 8192 96 128 \
  --shape 2 16384 16 128 \
  --shape 4 2048 16 128 \
  --shape 4 4096 64 128 \
  --shape 4 1024 8 512 \
  --warmup 10 \
  --iterations 30
```

## 7. Benchmark 输出字段

主要关注：

- `HIP hybrid ms`；
- `Current/HY TLE pre`；
- `Current/HY TLE post`；
- `Current/Cuda`；
- `peak_memory_mib`；
- `output_nan`。

计算方式：

```text
当前/HY TLE 前 = 当前 HIP 时间 / HY TLE 前时间
当前/HY TLE 后 = 当前 HIP 时间 / HY TLE 后时间
当前/Cuda      = 当前 HIP 时间 / 官方 GEMS 时间
```

数值越小越好；`1.00×` 表示与对比实现相同，低于 `1.00×` 表示当前实现更快。

## 8. 最终验收顺序

```bash
python test/test_gla_hip.py
python test/test_gla_chunk_dtk.py
python benchmarks/benchmark_gla_hygon.py \
  --implementation chunk \
  --dtype both \
  --warmup 10 \
  --iterations 30
```

只有在两个正确性测试通过且 `output_nan=False` 后，才记录最终性能。
