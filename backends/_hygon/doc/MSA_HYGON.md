# MSA 海光 BW1000 复现与优化说明

## 1. 算子范围

本实现复现 MiniMax M3 Sparse Attention（MSA）的 BF16 推理前向流程，包含：

1. index query 与分页 index KV cache 的 block score；
2. 强制保留 init/local blocks 的 TopK；
3. paged sparse attention；
4. ragged prefill 和多 token decode；
5. GQA/MQA head 映射；
6. FP32 online softmax 和 BF16 输出。

公共接口与官方实现对齐：

```python
minimax_m3_index_score
minimax_m3_index_topk
minimax_m3_index_decode_score
minimax_m3_index_decode
minimax_m3_sparse_attn
minimax_m3_sparse_attn_decode
```

当前稳定版本约束：BF16、`head_dim=128`、`page_size=128`、`topk<=16`、仅前向。FP8 和反向传播不在本版本范围内。

## 2. 最终目录

```text
BaselineBenchmark/backends/_hygon/
├── benchmarks/
│   └── benchmark_msa_hygon.py
├── doc/
│   └── MSA_HYGON.md
├── ops/msa/
│   ├── msa_hip.py
│   ├── msa_hip_kernel.cu
│   ├── msa_triton.py
│   └── msa_hygon_reference.py
└── test/msa/
    └── test_msa_hip.py
```

| 文件 | 用途 |
|---|---|
| `msa_hip.py` | 公共 API、输入检查以及 HIP/Triton 混合分发 |
| `msa_hip_kernel.cu` | index score、TopK、HIP sparse fallback、hipBLAS 调用 |
| `msa_triton.py` | `tl.dot` paged QK、online softmax、PV 融合内核 |
| `msa_hygon_reference.py` | 独立 PyTorch 正确性参考 |
| `test_msa_hip.py` | prefill/decode/GQA/paged KV/TopK 正确性测试 |
| `benchmark_msa_hygon.py` | 官方 14 组 workload 的分阶段性能测试 |

不提交 `__pycache__`、torch extension cache、临时 probe 和旧实验文件。

## 3. 数据布局

```text
q:              [total_q, query_heads, 128]
idx_q:          [total_q, kv_heads, 128]
index_kv_cache: [physical_pages, 128, 128]
kv_cache:       [physical_pages, kv_heads, 128, 256]
block_table:    [batch, max_logical_pages] int32
topk_idx:       [kv_heads, total_q, topk] int32
output:         [total_q, query_heads, 128]
```

`kv_cache[..., :128]` 为 K，`kv_cache[..., 128:]` 为 V。

## 4. 优化过程

### V1：HIP C++ 正确性基线

- 使用 HIP 实现 index score、TopK 和 sparse attention。
- 支持 ragged prefill、decode、GQA 和随机 physical page。
- 使用 FP32 softmax 累加和 BF16 输出。
- 主要目标是先保证公共接口和数学语义完整。

该版本 sparse attention 会构造 K pack、V pack、score 和 probability workspace，存在大量中间显存读写。

### V2：HIP index 与归约修正

- 修正 wave64 block reduction：只读取 CTA 中实际存在的 wave scratch 槽位。
- 修正 128-thread kernel 读取未初始化 LDS 槽导致的 page score 误差。
- 修正 hipBLAS ragged 尾 tile 的固定 batch stride，避免跨 batch 地址别名。
- decode 和短 query 使用直接 page-score kernel，避免构造巨大的 `B*H*max_seq` workspace。
- TopK 使用 wave64 分层选择，不对整行候选做完整排序。

正确性由此稳定通过。

### V3：Triton-on-HIP 融合 sparse attention

核心优化是把原来的：

```text
KV gather/pack → QK GEMM → score workspace → softmax → probability workspace → PV GEMM
```

融合为一个 Triton-on-HIP kernel：

```text
paged KV load → tl.dot(Q,K) → FP32 online softmax → tl.dot(P,V) → output
```

该版本：

- 直接读取 block table 指向的 physical pages；
- 不生成 K/V pack、完整 score 或 probability 张量；
- 使用 `tl.dot` 进入 FlagTree HCU 的矩阵计算路径；
- 使用 base-2 FP32 online softmax；
- HIP/hipBLAS 实现继续作为回退路径。

Prefill 六组相对 V1 平均提升约 `4.427x`。

### V4：Decode 混合分发

实测发现 B=1 decode 的 Triton fused kernel 受 launch/occupancy 限制，约为 `0.31 ms`；原 HIP decode 约为 `0.227 ms`。最终使用：

```text
B=1 decode  → HIP sparse kernel
B>=2 decode → Triton fused kernel
prefill     → Triton fused kernel
```

B=1 三组 sparse attention 相对纯 Triton 路径提升约 `1.36x`。

最终稳定后端标识：

```text
triton_fused_online+hip_decode_b1
```

实验性的 Triton index-decode 路径没有纳入最终性能结论，默认关闭：

```bash
export MSA_HYGON_TRITON_INDEX_DECODE=0
```

## 5. 正确性测试

### 环境

```bash
cd /workspace/FlagGems-vllm/BaselineBenchmark

export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8
export TRITON_ROOT=/workspace/FlagTree
export MSA_HYGON_USE_TRITON=1
export MSA_HYGON_TRITON_INDEX_DECODE=0
export PYTHONPATH=$TRITON_ROOT/build/lib.linux-x86_64-cpython-310:$TRITON_ROOT/python:$PWD/backends/_hygon/ops/msa:$PYTHONPATH
```

### 最终测试命令

```bash
rm -rf /root/.cache/torch_extensions/py310_cpu/msa_hygon_hip_ext_v2_score_reduce
python backends/_hygon/test/msa/test_msa_hip.py
```

覆盖内容：

- ragged prefill；
- 多 token decode；
- B=1 HIP decode 混合分发；
- index score 和 TopK；
- init/local block 强制保留；
- GQA head 映射；
- 随机 physical page；
- `out`/`score_out` strided buffer 与 sentinel；
- BF16 有限值和参考误差。

成功标志：

```text
MSA Hygon correctness: PASS
```

## 6. Benchmark

最终完整命令：

```bash
python backends/_hygon/benchmarks/benchmark_msa_hygon.py \
  --mode both \
  --warmup 10 \
  --iterations 30
```

只测试 decode：

```bash
python backends/_hygon/benchmarks/benchmark_msa_hygon.py \
  --mode decode \
  --start 0 \
  --end 8 \
  --warmup 20 \
  --iterations 100
```

输出字段：

```text
mode,index,B,T,KVH,QH,index_ms,sparse_ms,pipeline_ms,tokens_per_s
```

`pipeline_ms` 是最终端到端指标，包含 index score、TopK 和 sparse attention。

## 7. 最终性能与加速比

加速比定义为 `基准时间 / 当前时间`，使用 `pipeline_ms`。

| Mode | Shape `[B,T,KVH,QH]` | V1 HIP (ms) | 当前 (ms) | 相对 V1 | 海光 FlagAttn (ms) | 当前相对 FlagAttn |
|---|---|---:|---:|---:|---:|---:|
| prefill | 1×8192×16×96 | 1051.0443 | 239.0806 | **4.396×** | 1001.6734 | **4.190×** |
| prefill | 2×16384×8×96 | 2292.9164 | 569.6387 | **4.025×** | 1601.7306 | **2.812×** |
| prefill | 1×32768×16×96 | 5138.6575 | 1500.7570 | **3.424×** | 4645.4653 | **3.095×** |
| prefill | 2×8192×8×96 | 1078.3279 | 239.7062 | **4.499×** | 73.1179 | 0.305×（慢 3.278×） |
| prefill | 4×4096×16×384 | 2237.0503 | 433.3683 | **5.162×** | 133.7850 | 0.309×（慢 3.239×） |
| prefill | 4×4096×16×256 | 2129.4530 | 421.0195 | **5.058×** | 126.8028 | 0.301×（慢 3.320×） |
| decode | 1×4096×16×96 | 0.3812 | 0.3700 | **1.030×** | 0.6336 | **1.712×** |
| decode | 1×16384×16×96 | 0.6318 | 0.6267 | **1.008×** | 0.6408 | **1.022×** |
| decode | 1×65536×16×96 | 1.6943 | 1.6842 | **1.006×** | 0.6696 | 0.398×（慢 2.515×） |
| decode | 4×4096×8×96 | 0.6018 | 0.4720 | **1.275×** | 0.4400 | 0.932×（慢 1.073×） |
| decode | 4×16384×8×96 | 1.0957 | 0.9629 | **1.138×** | 0.6502 | 0.675×（慢 1.481×） |
| decode | 16×4096×8×96 | 1.8841 | 0.9830 | **1.917×** | 0.7909 | 0.805×（慢 1.243×） |
| decode | 32×2048×4×48 | 1.5281 | 0.6302 | **2.425×** | 0.2109 | 0.335×（慢 2.988×） |
| decode | 64×1024×4×48 | 2.4610 | 0.8458 | **2.910×** | 0.2160 | 0.255×（慢 3.916×） |

阶段汇总：

| 类别 | 相对 V1 平均加速 |
|---|---:|
| Prefill 六组 | **4.427×** |
| Decode 八组 | **1.714×** |
| 全部十四组 | **2.877×** |

## 8. 当前瓶颈与结论

- `B=1,T=65536` decode 中，index 约占 pipeline 的 `88.3%`，继续优化 sparse attention 收益有限。
- 小 Batch 长 prefill 已明显快于给定的海光 FlagAttention 参考结果。
- Batch 较大的 prefill 和 decode 仍与海光 FlagAttention 有约 `1.07x–3.92x` 差距。
- 当前稳定版本优先保证正确性、可回退和已实测收益；未完成正确性及同环境 A/B 的实验路径不计入最终结果。
