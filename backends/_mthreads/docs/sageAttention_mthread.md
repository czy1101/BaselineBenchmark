# Moore Threads MATE SageAttention Baseline 使用与复现说明

本文档说明如何在 Moore Threads MTT S5000（MUSA MP31）上安装和验证官方 MATE SageAttention baseline，并复现其与 FlagAttention Triton-MUSA 的 Core/E2E 性能对比。

相关代码：

- Core/E2E 性能对比：`BaselineBenchmark/backends/_mthreads/benchmarks/sage_attention_benchmark_compare.py`
- 可配置的 MATE/FlagAttention/PyTorch 正确性测试：`BaselineBenchmark/backends/_mthreads/tests/test_sage_attention_mate.py`
- FlagAttention 独立 E2E 测试：`tests/flag_attn/test_sage_attention_e2e.py`

## 1. Baseline 定义

本项目使用 Moore Threads 官方 `sageattention` Python wrapper、`mate` runtime 和官方预生成 MUBIN 作为 baseline，不在仓库中复制或重写 MATE 算子。

当前锁定版本为：

| 组件              | 版本                                |
| ----------------- | ----------------------------------- |
| GPU               | Moore Threads MTT S5000             |
| Capability        | MP31 /`(3, 1)`                    |
| Python            | 3.10                                |
| PyTorch           | `2.7.1`                           |
| torch_musa        | `2.7.1+5ee0a64`                   |
| FlagTree / Triton | `0.6.1a2+mthreads3.6` / `3.6.0` |
| MATE              | `0.2.5`                           |
| sageattention     | `0.2.5+musa`                      |
| mate-mubin        | `0.2.5`                           |
| apache-tvm-ffi    | `0.1.9.post3+musa.1`              |

必须锁定 MATE、sageattention 和 mate-mubin 的相同小版本。不要直接升级到包索引显示的最新版本后继续沿用旧结果；MATE API、量化配方、MUBIN 和性能都可能变化。

## 2. 比较前必须理解的差异

MATE 与 FlagAttention 当前实现不是完全相同的量化算法：

| 项目              | FlagAttention Triton-MUSA | MATE 0.2.5                                    |
| ----------------- | ------------------------- | --------------------------------------------- |
| Q                 | INT8，block=128           | INT8，recipe 中 Q block=128                   |
| K                 | INT8，block=64            | INT8，recipe 中 K block=16，`smooth_k=True` |
| V                 | FP16                      | FP8 E4M3                                      |
| MATE quant recipe | 不适用                    | `(128, 16, -1, 1)`                          |
| 核心执行体        | 当前仓库 Triton kernel    | 官方预编译 MP31 MUBIN                         |

因此应把 MATE 理解为“官方 MP31 专用实现的性能与精度 baseline”，而不是当前 FlagAttention kernel 的逐位等价 reference。

比较时遵守以下原则：

1. FlagAttention 和 MATE 都必须分别与同一个 FP32 attention reference 比较；
2. 默认只约束 `Triton vs MATE` 的整体质量指标，不要求逐位相同；需要排查离群点时再启用严格模式；
3. 性能报告必须区分 Core 和 End-to-End，不能把一方 Core 与另一方 E2E 混在一起；
4. 版本、MUBIN、输入 shape、dtype、warmup、rep 和代码校验值必须随结果一起保存。

## 3. 环境安装

以下命令在容器内执行。进入 FlagAttention 仓库根目录：

```bash
cd /workspace/FlagAttention-main
```

### 3.1 确认基础环境

```bash
python --version

python - <<'PY'
import torch
import torch_musa

print("device:", torch.musa.get_device_name(0))
print("capability:", torch.musa.get_device_capability(0))
print("device_count:", torch.musa.device_count())
print("available:", torch.musa.is_available())
PY

python -m pip show flagtree mate sageattention mate-mubin apache-tvm-ffi || true
```

预期基础设备至少满足：

```text
device: MTT S5000
capability: (3, 1)
available: True
```

### 3.2 从 Moore Threads 包索引安装固定版本

清华 PyPI 等公共镜像通常没有 `torch_musa`、MATE MUSA 版或对应 sageattention 包。使用 Moore Threads 包索引：

```bash
export MTHREADS_INDEX=https://dl.mthreads.com/repo/api/pypi/pypi/simple

python -m pip install --dry-run \
  'sageattention==0.2.5+musa' \
  'mate==0.2.5' \
  --index-url "$MTHREADS_INDEX"
```

确认 dry-run 没有改动 torch/torch_musa 后再安装：

```bash
python -m pip install \
  --no-cache-dir \
  'sageattention==0.2.5+musa' \
  'mate==0.2.5' \
  --index-url "$MTHREADS_INDEX"
```

该安装会把镜像自带的旧 `mate 0.2.0+mu437torch2.7` 升级为 `mate 0.2.5`，并把 `apache-tvm-ffi` 升级到 MATE 0.2.5 所需版本。

当前基础镜像中可能出现以下 resolver 警告：

```text
tilelang-musa ... requires apache-tvm-ffi==0.1.0
but apache-tvm-ffi 0.1.9.post3+musa.1 is installed
```

该警告不代表 MATE 安装失败；若最终显示 `Successfully installed` 且后续验证通过，baseline 可以运行，但 TileLang-MUSA 可能受影响。需要同时使用 TileLang 时应单独建容器或提前保存可恢复镜像。

### 3.3 安装官方 MUBIN

安装 Python 包后还必须安装与 MATE 版本匹配的预生成 MUBIN：

```bash
mate list-mubins
mate install-mubin-wheel --dry-run
mate install-mubin-wheel

python -m pip show mate-mubin
mate list-mubins
```

当前版本预期为：

```text
mate-mubin: 0.2.5
gemm:            Downloaded
flash_attention: Downloaded
flash_mla:       Downloaded
sage_attention:  Downloaded
```

MUBIN 目录 hash 可能随官方构建变化，复现时以 `mate list-mubins` 输出为准并将其保存到实验记录。`Missing` 状态下不要运行正式 benchmark，否则可能触发下载、fallback 或直接失败，结果不具备可比性。

### 3.4 验证最终环境

```bash
python -m pip show flagtree sageattention mate mate-mubin apache-tvm-ffi

python - <<'PY'
import torch
import torch_musa
import triton
import mate
import sageattention

print("torch:", torch.__version__)
print("torch_musa:", torch_musa.__version__)
print("triton:", triton.__version__)
print("mate:", mate.__version__)
print("sageattention:", sageattention.__version__)
print("device:", torch.musa.get_device_name(0))
print("capability:", torch.musa.get_device_capability(0))
print("available:", torch.musa.is_available())
PY

mate list-mubins
```

## 4. 正确性复现

所有命令默认在 FlagAttention 仓库根目录执行，并显式设置：

```bash
PYTHONPATH="$PWD/src"
```

### 4.1 统一可配置正确性测试

```bash
PYTHONPATH="$PWD/src" python -m pytest \
  BaselineBenchmark/backends/_mthreads/tests/test_sage_attention_mate.py \
  -q -s -rs
```

默认执行 `mate-torch + e2e + relaxed`：从 FP16 Q/K/V 出发调用 MATE high-level `sageattn`，再与 FP32 PyTorch reference 比较；该配置不加载 FlagAttention，适合独立验证 baseline。

三个正交配置项如下：

| 环境变量                    | 可选值                                                    | 默认值         | 含义                                     |
| --------------------------- | --------------------------------------------------------- | -------------- | ---------------------------------------- |
| `SAGE_CORRECTNESS_TARGET` | `mate-torch` / `flag-torch` / `flag-mate` / `all` | `mate-torch` | 选择比较对象                             |
| `SAGE_CORRECTNESS_PATH`   | `e2e` / `core` / `both`                             | `e2e`        | 选择完整公开链路、预量化低层执行体或两者 |
| `SAGE_CORRECTNESS_MODE`   | `relaxed` / `strict`                                  | `relaxed`    | 选择整体质量指标或附加严格逐元素检查     |

例如，同时验证 FlagAttention、MATE 和 PyTorch reference 的 E2E/core：

```bash
SAGE_CORRECTNESS_TARGET=all \
SAGE_CORRECTNESS_PATH=both \
SAGE_CORRECTNESS_MODE=relaxed \
PYTHONPATH="$PWD/src" \
python -m pytest \
  BaselineBenchmark/backends/_mthreads/tests/test_sage_attention_mate.py \
  -q -s
```

`relaxed` 模式的强制门槛：

| 比较                          |   cosine | relative L1 |    RMSE |
| ----------------------------- | -------: | ----------: | ------: |
| MATE/FlagAttention vs PyTorch | ≥ 0.995 |     ≤ 0.05 | ≤ 0.02 |
| FlagAttention vs MATE         |  ≥ 0.99 |     ≤ 0.08 | ≤ 0.03 |

`strict` 在上述门槛外增加 output 的 `atol=0.02, rtol=0.02` 逐元素检查。E2E 检查 LSE；core 低层接口以 `return_lse=False` 调用，只检查 output。`relaxed` 仍执行强制质量检查，仅避免少量 FP8 近似离群点使 baseline 失去可用性。

需要注意 LSE 的底数：

- FlagAttention `forward` 返回 log2-LSE；
- MATE high-level API 返回自然对数 LSE；
- 测试会先将 FlagAttention LSE 乘以 `ln(2)`，再执行比较。

测试覆盖 HND/NHD、MHA/GQA、output 与 LSE。MATE 0.2.5 的公开
`sageattn` 签名可能不提供任意 `attn_mask`，这种情况下 bool/additive mask 与
partial-block case 会显示 `SKIPPED`；这不是测试失败，但也不能据此宣称该版本
支持任意 mask。测试还会验证 MATE high-level API 没有 `maxnreg` 调优参数。

## 5. 性能复现

benchmark 使用整卡同步墙钟计时；首次编译和原始输入创建排除在计时外，Core 还排除量化与布局准备，E2E 则包含量化及必要的类型/布局转换。

### 5.1 Core 与 E2E 的含义

#### Core Performance

两边都在计时前完成各自的量化和布局准备：

- FlagAttention：Q/K INT8 per-block、packed K、FP16 V；
- MATE：BNHD BF16 输入准备、Q/K INT8、smooth K、V FP8 及 scale。

计时区只包含各自 attention 执行入口。因此 Core 比较的是“各自准备好官方输入格式后的执行体性能”，不是逐位相同量化数据上的同一个数学 kernel。

#### End-to-End Performance

两边从相同的原始 Q/K/V 开始，将量化、必要布局/类型转换和 attention 都计入延迟。

对于 BF16 输入，当前 FlagAttention kernel 要求 V=FP16，因此 FlagAttention E2E 包含 BF16→FP16 V 转换；Core 在计时前完成该转换。

### 5.2 快速全矩阵与正式分组测速

先以较小重复次数运行完整 FP16/BF16、D64/D128 矩阵进行冒烟：

```bash
PYTHONPATH="$PWD/src" \
python BaselineBenchmark/backends/_mthreads/benchmarks/sage_attention_benchmark_compare.py \
  --dtypes float16 bfloat16 \
  --head-dims 64 128 \
  --seq-lens 1024 4096 8192 16384 \
  --batch4-seq-len 1024 \
  --warmup 5 \
  --rep 10
```

正式结果建议固定 `warmup=50, rep=200` 并按 dtype/head-dim 分组，例如 D64 FP16：

```bash
PYTHONPATH="$PWD/src" \
python BaselineBenchmark/backends/_mthreads/benchmarks/sage_attention_benchmark_compare.py \
  --dtypes float16 \
  --head-dims 64 \
  --seq-lens 1024 4096 8192 16384 \
  --batch4-seq-len 1024 \
  --warmup 50 \
  --rep 200
```

若完整矩阵在较大 warmup/rep 下不稳定，应继续拆分 shape，但同一张结果表必须保持相同计时参数。

### 5.3 只测 Core 或 E2E

在 5.2 的命令基础上追加：

```text
只测 Core：--skip-e2e
只测 E2E： --skip-core
```

不要同时传 `--skip-core --skip-e2e`；脚本会直接报错。

## 6. 结果解读

`Triton_speedup = MATE_ms / Triton_ms`：大于 1 表示 FlagAttention Triton 更快，小于 1 表示 MATE 更快。例如 `0.779` 表示 Triton 速度约为 MATE 的 77.9%，不能解释成 Triton 快 0.779 倍。

表中的 TFLOPS 使用统一理论 FLOP 数：

```text
4 × B × H × T² × D
```

该数值只用于相同 shape 的相对比较，不代表两边真实执行了相同的数据类型、指令数或稠密 FLOP 数，也不能直接视为硬件峰值利用率。

## 7. 常见问题

### 7.1 短序列下 MATE 延迟看起来近似固定

MATE 使用官方预生成 MP31 MUBIN，其调度、固定开销和 shape dispatch 与 Triton 不同。当前 benchmark 已使用整卡同步墙钟并排除首次准备；只要版本、MUBIN、设备空闲状态和重复次数一致，可以记录该结果。不要仅凭 TFLOPS 数值反推 MATE 内部实现，也不要把不同版本或不同计时器产生的数据混表。