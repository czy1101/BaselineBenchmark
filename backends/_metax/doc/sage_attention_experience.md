# SageAttention：从 EXP10A 实验到可选原生后端

证据 ID：`sage_formal_equal`、`sage_adapter`、`sage_package_failed`、`sage_production`、
`sage_published`，见[索引](../workflow/runs/20260907/evidence_index.json)。

## 成功路线

先冻结实验源与正式源，隔离构建/测试；再补齐正式 loader、构建入口、打包数据和回退测试。
正式 Triton 文件此前 5/5 字节相同只表示“旧正式版已在目标分支”，不等于 EXP10A 已合入。
实验 adapter 的 23 项通过也不能替代正式集成候选验收。

正式候选最终 42 项通过、0 跳过：原 Sage 测试 8 项、loader CPU 测试 18 项、native GPU
测试 16 项。日志记录 Native 13、Triton 8 次路由调用；调用次数不等于测试项数，
也不能把全部 42 项称为 GPU 用例。wheel/sdist 内容及 wheel 导入 smoke 均通过。
原生构建阶段 41.577 s；这是构建耗时，不是 attention 延迟。

后续源码发布由远端提交
[52f7c60](https://github.com/czy1101/FlagAttention/commit/52f7c6096bd6c433ee51d142e7526185fb9d4e25)
确认：`perf(metax): integrate optional SageAttention native backend`。
因此正式验证日志中的 `merge=false` 是当时隔离阶段状态，不能用来否认后续已发布的源码。

## 配对性能

以下是正式集成候选在 WZB/C550 上、5 个 FP16 HND Shape 的配对端到端结果。
相同测量范围包含原有量化等前处理；不是仅截取原生 attention 内核。

| B,T,H,D | Formal HND ms | Native HND ms | 配对加速比 | CV formal | CV native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,1024,32,128 | 0.573601 | 0.410074 | 1.3988× | 0.058% | 0.061% |
| 4,1024,32,128 | 1.727232 | 1.059445 | 1.6308× | 0.052% | 0.084% |
| 1,4096,32,128 | 5.915920 | 2.795233 | 2.1165× | 0.017% | 0.023% |
| 1,8192,32,128 | 23.128256 | 10.190279 | 2.2698× | 0.020% | 0.032% |
| 1,16384,32,128 | 91.681282 | 39.232256 | 2.3365× | 0.024% | 0.043% |

几何平均加速 1.9128×。日志另有 20 行官方 benchmark 输出；不能把这 5 个 HND
配对结果扩展成“全部 20 行均提速”或“NHD 也使用原生后端”。

## 可复用的集成原则

原生后端必须显式构建并选择可信产物，安装或 import 不自动编译/下载；默认保留 Triton。
产物绑定实际源文件/二进制哈希及运行时身份，而不信任实验中遗留的 build_info 哈希字符串。
不把 SDK 或预编译二进制提交进源码包。变更环境后重新构建、验收；源码已发布不表示
任意容器已构建、启用并验证原生产物。

当前加速域是 C550 eager inference、连续 HND、Q/K/V 同 batch/head、D=128，
INT8 Q/K、FP16 V/output、FP32 block scale；Q 长度对齐 128、KV 对齐 64，沿用既有
量化约定。不暗中做布局转换。NHD 等非加速域沿用 Triton，且不新增 Triton 原本不支持的语义。
缺失或不匹配的产物可回退；真正执行中的原生错误必须上抛，不能静默回退掩盖错误。

具体构建和启用步骤以固定提交的
[NATIVE.md](https://github.com/czy1101/FlagAttention/blob/52f7c6096bd6c433ee51d142e7526185fb9d4e25/src/flag_attn/runtime/backend/_metax/sage_attention/NATIVE.md)
为准；本次文档整理不重新构建或运行。

## 失败经验

- 正式打包检查因 `setuptools-scm` 元数据缺失失败，发生在编译/GPU/benchmark 之前。
  这是构建工具环境问题，不是原生数学错误；使用独立离线构建工具完成验证，保留原 Torch/Triton 运行时。
- 只验证 `.so` 能调用，遗漏 wheel/sdist 的源文件与文档，会造成仓库里成功、安装包中失败。
  必须检查包内容、安装后导入、ABI/版本不匹配、缺失产物和回退行为。
- 长命令粘贴截断、脚本未上传、Windows 路径与容器路径混淆，均不能算成算子失败。
  交付可校验的独立脚本/归档，先确认文件存在，再运行。
- 旧 NATIVE 文档中引用的 23 项实验 adapter 证据是历史证据；正式集成验收应同时查阅本次
  42 项日志记录，不能用旧阶段结论代替新阶段结果。
