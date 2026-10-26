# KDA 海光最终提交与测试

## 最终目录

```text
BaselineBenchmark/backends/_hygon/
├── benchmarks/benchmark_kda_hygon.py
├── ops/kda/kda_hip.py
├── ops/kda/kda_hip_kernel.cu
└── test/kda/test_kda_hip.py
```

## 环境设置

```bash
cd /workspace/FlagGems-vllm/BaselineBenchmark

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export PYTHONPATH=$PWD/backends/_hygon/ops/kda:$PYTHONPATH
```

## 正确性测试

```bash
python backends/_hygon/test/kda/test_kda_hip.py
```

正确输出：

```text
KDA DTK GEMM correctness: PASS
```

## 最终 Benchmark

当前 Benchmark 内置两组生产 shape：

```text
B=1, T=8192, H=96, HV=96, K=128, V=128
B=8, T=1024, H=96, HV=96, K=128, V=128
```

执行：

```bash
python backends/_hygon/benchmarks/benchmark_kda_hygon.py \
  --dtype both \
  --chunk-size 64 \
  --warmup 20 \
  --iterations 100
```

不要添加 `--shape` 参数；当前最终 Benchmark 使用内置 shape。

## 最新结果（p50）

| Workload | FP16 (ms) | BF16 (ms) |
|---|---:|---:|
| B=1, T=8192, H=HV=96, K=V=128 | 61.780 | 62.029 |
| B=8, T=1024, H=HV=96, K=V=128 | 46.948 | 46.929 |
