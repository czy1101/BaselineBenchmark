# MUSA C Chunk GLA 基准实现说明

## 适用范围

该基准目录包含独立的 MUSA C 实现、PyTorch 自动求导封装、构建入口、
正确性测试和性能对比程序。

生产使用的 Triton/TLE 实现仍保留在：

```text
src/flag_attn/runtime/backend/_mthreads/gla/
```

测试和性能对比程序会直接从上述生产源码目录导入 `ChunkGLAFunction`。
`BaselineBenchmark` 中不会复制 Triton/TLE 实现，确保原来的 Triton/TLE
代码仍然只有一份。

## 目录结构

```text
BaselineBenchmark/backends/_mthreads/
├── ops/chunk_gla_musa_c/
│   ├── __init__.py
│   ├── musa_chunk_gla.py
│   ├── musa_chunk_gla.cpp
│   ├── musa_chunk_gla_kernel.mu
│   ├── torch_reference.py
│   └── build_musa_chunk_gla.py
├── tests/test_chunk_gla_musa_c.py
├── benchmarks/chunk_gla_musa_c_compare.py
└── docs/chunk_gla_musa_c.md
```

各文件用途：

- `musa_chunk_gla.py`：MUSA C 扩展的 Python 封装和自动求导接口；
- `musa_chunk_gla.cpp`：PyTorch C++ 绑定、参数检查和工作区分配；
- `musa_chunk_gla_kernel.mu`：MUSA 内核和 muBLAS 计算实现；
- `torch_reference.py`：仅用于正确性验证的朴素 PyTorch 递推实现；
- `build_musa_chunk_gla.py`：独立扩展构建入口；
- `test_chunk_gla_musa_c.py`：唯一的 MUSA C GLA 正确性测试文件；
- `chunk_gla_musa_c_compare.py`：MUSA C 与 Triton/TLE 性能对比程序；
- `chunk_gla_musa_c.md`：构建、测试和性能测量说明。

## 环境准备

进入 FlagAttention 项目根目录：

```bash
cd /workspace/FlagAttention-main
export PYTHONPATH=$PWD/src:$PWD
```

其中 `$PWD/src` 用于加载 FlagAttention Triton/TLE 实现，`$PWD`
用于加载 `BaselineBenchmark` 中的 MUSA C 基准包。

## 编译 MUSA C 扩展

在摩尔线程 MUSA 容器中，从项目根目录执行：

```bash
python BaselineBenchmark/backends/_mthreads/ops/chunk_gla_musa_c/build_musa_chunk_gla.py build_ext --inplace --force
```

构建脚本会切换到基准实现自身目录，避免继承主项目的 `src` 布局。
生成的 `.so` 位于：

```text
BaselineBenchmark/backends/_mthreads/ops/chunk_gla_musa_c/_musa_chunk_gla*.so
```

对应的 Python 模块路径为：

```text
BaselineBenchmark.backends._mthreads.ops.chunk_gla_musa_c._musa_chunk_gla
```

Python 中的导入方式为：

```python
from BaselineBenchmark.backends._mthreads.ops.chunk_gla_musa_c import (
    is_available,
    musa_chunk_gla,
)
```

可以用下面的命令检查扩展是否加载成功：

```bash
python -c "from BaselineBenchmark.backends._mthreads.ops.chunk_gla_musa_c import is_available; print(is_available())"
```

期望输出：

```text
True
```

## 快速数学模式

快速数学模式默认关闭。只有在明确需要测试近似数学函数性能时才开启：

```bash
MUSA_GLA_FAST_MATH=1 python BaselineBenchmark/backends/_mthreads/ops/chunk_gla_musa_c/build_musa_chunk_gla.py build_ext --inplace --force
```

快速数学模式可能改变指数函数等数学运算的精度。开启后必须重新运行完整正确性测试，
不能直接复用关闭快速数学模式时的一致性结论。

## 正确性测试

运行正确性测试文件：

```bash
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q -s BaselineBenchmark/backends/_mthreads/tests/test_chunk_gla_musa_c.py
```

测试会比较三条路径：

1. 朴素 PyTorch FP32 递推参考实现；
2. `src/flag_attn/runtime/backend/_mthreads/gla/` 中的 Triton/TLE；
3. `BaselineBenchmark` 中的 MUSA C 基准实现。

当前覆盖：

- 前向输出；
- 最终状态；
- 反向传播结果 `dq`、`dk`、`dv` 和 `dg`；
- 小形状；
- 非 2 的幂形状；
- `D=128` 和 `D=256` 宽维度形状。

如果扩展尚未编译，测试会跳过，并提示先运行构建命令。

## 性能测试

性能测试不会自动运行正确性测试。正式测量前应先单独确认正确性测试
已经通过。

### 小形状冒烟测试

```bash
PYTHONPATH="$PWD/src:$PWD" python BaselineBenchmark/backends/_mthreads/benchmarks/chunk_gla_musa_c_compare.py --small --warmup 1 --rep 2
```

### 常规 BF16 对比

```bash
PYTHONPATH="$PWD/src:$PWD" python BaselineBenchmark/backends/_mthreads/benchmarks/chunk_gla_musa_c_compare.py --mode all --dtype bfloat16 --warmup 10 --rep 100
```

默认比较：

```text
MUSA C 基准实现与生产目录中的 Triton/TLE 实现
```

支持的模式：

```text
--mode forward
--mode backward
--mode all
```

支持的数据类型：

```text
--dtype bfloat16
--dtype float16
--dtype float32
```

可以重复传入 `--shape` 指定自定义形状：

```bash
PYTHONPATH="$PWD/src:$PWD" python BaselineBenchmark/backends/_mthreads/benchmarks/chunk_gla_musa_c_compare.py --shape 1 8192 96 128 --shape 4 2048 16 128 --dtype bfloat16 --warmup 10 --rep 100
```

PyTorch 递推实现在常规大形状上非常慢，因此默认不测量。只有明确需要时才增加：

```text
--include-torch
```

输出中的 `musa_c/triton` 大于 1，表示 MUSA C 比 Triton/TLE 慢；小于 1，
表示 MUSA C 比 Triton/TLE 快。

## 注意事项

- MUSA C 基准实现不属于生产 Triton/TLE 后端，不应从生产代码中导入；
- 原来的 Triton/TLE 实现不复制到 `BaselineBenchmark`；
- 性能测试必须直接调用 `ChunkGLAFunction`，确保 Triton/TLE 对比路径明确；
- 编译生成的 `.so`、`build/` 和 Python 缓存不应提交；
- 修改 `.cpp` 或 `.mu` 后必须使用 `--force` 重新编译；
- 修改数值计算、快速数学模式或累加数据类型后，必须重新运行完整正确性测试；
- 服务器运行时应同时设置 `$PWD/src` 和 `$PWD` 到 `PYTHONPATH`。