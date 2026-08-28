# MLA 海光 HIP 适配说明

## 1. 最终提交目录

```text
mla_hygon/
├── benchmarks/
│   └── benchmark_mla_hygon.py
├── ops/
│   └── mla/
│       ├── mla_hip.py
│       └── mla_hip_kernel.cu
└── test/
    └── mla/
        ├── mla_hygon_reference.py
        └── test_mla_hip.py
```

## 2. 应提交的文件

| 文件 | 作用 |
|---|---|
| `mla_hip.py` | MLA Python API、参数检查、Extension 加载和 dispatch |
| `mla_hip_kernel.cu` | 纯 HIP C++ MLA decode/prefill Kernel |
| `benchmark_mla_hygon.py` | 官方 shape 和扩展 shape 性能测试 |
| `test_mla_hip.py` | FP16/BF16 MLA HIP 正确性测试 |
| `mla_hygon_reference.py` | 正确性测试 oracle，不参与生产运行 |

## 3. 不建议提交的文件

```text
__pycache__/
*.csv
*.log
*.json
*.hip                       # 如果只是编译中间文件或备份文件
```

## 4. 支持功能

- FP16/BF16；
- GQA/MQA，`Hkv=1`；
- decode：`Sq=1`；
- prefill：`Sq>1`；
- `D=512/576`；
- `DV=512`；
- `Skv=1024/2048/4096/8192/16384`；
- FP32 累加和 LSE/online softmax 语义；
- 输出保持输入 dtype。

## 5. 正确性测试

```bash
cd /workspace/hy/compare/mla_hygon

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export PYTHONPATH=$PWD/ops/mla:$PWD/test:$PYTHONPATH

test -f test/mla/mla_hygon_reference.py

rm -rf /root/.cache/torch_extensions/py310_cpu/mla_hygon_hip_ext*

python test/mla/test_mla_hip.py
```

期望输出：

```text
MLA HIP correctness: PASS
```

## 6. 官方 shape 性能测试

### Decode KV 长度测试

```bash
python benchmarks/benchmark_mla_hygon.py \
  --dtype both \
  --shape 128 1 128 1 576 512 1024 \
  --shape 128 1 128 1 576 512 2048 \
  --shape 128 1 128 1 576 512 4096 \
  --shape 128 1 128 1 576 512 8192 \
  --shape 128 1 128 1 576 512 16384 \
  --warmup 20 \
  --iterations 50
```

### 小 batch decode

```bash
python benchmarks/benchmark_mla_hygon.py \
  --dtype both \
  --shape 1 1 128 1 576 512 8192 \
  --shape 8 1 128 1 576 512 8192 \
  --shape 128 1 128 1 576 512 8192 \
  --warmup 10 \
  --iterations 30
```

### Prefill shape

```bash
python benchmarks/benchmark_mla_hygon.py \
  --dtype both \
  --shape 1 4096 128 1 576 512 8192 \
  --shape 1 4096 64 1 512 512 8192 \
  --shape 1 4096 128 1 512 512 8192 \
  --warmup 10 \
  --iterations 30
```

## 7. Benchmark 输出

主要记录：

- `mean_ms`；
- `p50_ms`；
- `min_ms`；
- `tok_per_s`；
- `peak_mib`；
- `nan`。

对比 H800 时使用：

```text
海光/H800 Triton = 海光 HIP ms / H800 Triton ms
海光/H800 TLE    = 海光 HIP ms / H800 TLE ms
```

倍率越小越好。

## 8. 最终验收顺序

```bash
python test/mla/test_mla_hip.py

python benchmarks/benchmark_mla_hygon.py \
  --dtype both \
  --shape 128 1 128 1 576 512 1024 \
  --shape 128 1 128 1 576 512 2048 \
  --shape 128 1 128 1 576 512 4096 \
  --shape 128 1 128 1 576 512 8192 \
  --shape 128 1 128 1 576 512 16384 \
  --warmup 20 \
  --iterations 50
```

只有在正确性 PASS 且 `nan=False` 后，才记录最终性能结果。
