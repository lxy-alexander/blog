---
title: "[Bug]: Direct DCP A2A crashes on GLM sparse-MLA strided output"
published: 2026-08-29
description: "[Bug]: Direct DCP A2A crashes on GLM sparse-MLA strided output"
image: ""
tags: ["llm_inference","vllm","[Bug]: Direct DCP A2A crashes on GLM sparse-MLA strided output"]
category: llm_inference / vllm
draft: false
lang: ""
createdAt: "2026-08-29T06:09:46.166.048938854Z"
---

第一阶段：修复崩溃
不兼容布局 → 回退通用 A2A

第二阶段：性能优化
让 sparse MLA kernel 可选地直接输出 packed 布局
→ 再走 Direct A2A



GLM 的稀疏 MLA 后端返回：

```
attn_out.shape  = (32, 64, 512)
attn_out.stride = (512, 1572864, 1)
```

Direct DCP 希望：(32768, 512, 1)

```
物理显存：

Token 0:
    Head 0 [512]
    Head 1 [512]
    Head 2 [512]
    ...
    Head 63 [512]

Token 1:
    Head 0 [512]
    Head 1 [512]
    ...
```

因此：

```
T0H0 T0H1 T0H2 ... T0H63 T1H0 T1H1...
```

而 GLM 给它的是：

```
Head 0:
    Token 0 [512]
    Token 1 [512]
    Token 2 [512]
    ...

Head 1:
    Token 0 [512]
    Token 1 [512]
    ...
```

即：

```
T0H0 T1H0 T2H0 ... | T0H1 T1H1 T2H1 ... | ...
```

逻辑内容一样，**排列完全不同**。







可以用两张 H100 做“近似复现”，但先说明一个关键限制：

原问题环境是 4 张 Blackwell GPU、TP=4、DCP=4、NVFP4 MLA KV Cache。H100 不支持原生 NVFP4，而且可能选择不同的 MLA kernel，返回的张量布局也可能不同。因此：

-   可以验证 Direct DCP A2A 的选择和修复逻辑。
-   不保证能复现完全相同的 `(512, 1572864, 1)` stride 崩溃。
-   两张卡需要改成 `TP=2、DCP=2`。
-   官方 GLM-5.3-Flash 很大，两张 H100 通常放不下，需要问题中相同的量化 checkpoint。

## 1. 检查环境

```
cd /data/home/xli49/lxy/vllm

nvidia-smi -L
nvidia-smi topo -m

.venv/bin/python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory / 1024**3, "GiB")
PY
```

确认两张 GPU 都能看到，并且最好处于同一台机器、具有 NVLink/NVSwitch 或至少支持对称内存通信。

## 2. 准备模型

优先使用 issue 作者使用的量化模型目录：

```
export REPRO_MODEL=zai-org/GLM-5.3-Flash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_LOGGING_LEVEL=DEBUG
export TORCH_SHOW_CPP_STACKTRACES=1
```

如果直接使用：

```
export REPRO_MODEL=zai-org/GLM-5.3-Flash
```

大概率会先因为显存不足或当前版本不支持该架构而失败，这不属于 #54305。

## 3. 用自动选择模式复现

这是最接近原 issue 的方式：

```
unset VLLM_USE_DIRECT_DCP_A2A

.venv/bin/vllm serve "$REPRO_MODEL" \
  --tensor-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

这里 H100 使用 `fp8`，因为不能照搬 Blackwell 的 NVFP4。

这个问题发生在启动阶段的 CUDA Graph 捕获过程中，通常不需要发送请求。

复现成功时，旧代码应出现类似：

```
RuntimeError: partial output must have packed heads and an aligned token stride
```

## 4. 自动模式没有进入 Direct A2A 时

如果日志中没有看到 Direct/Symmetric-memory DCP A2A 被选择，可以强制开启：

```
export VLLM_USE_DIRECT_DCP_A2A=1

.venv/bin/vllm serve "$REPRO_MODEL" \
  --tensor-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

如果这时报告“对称内存不可用”之类的错误，那是 H100 通信环境问题，不是目标 stride bug。

## 5. 做对照实验

关闭 Direct DCP A2A：

```
export VLLM_USE_DIRECT_DCP_A2A=0

.venv/bin/vllm serve "$REPRO_MODEL" \
  --tensor-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend a2a \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 32 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

判断方式：

| Direct 设置 | 旧代码预期                                |
| ----------- | ----------------------------------------- |
| `=1`        | 如果 MLA 返回 head-major stride，启动崩溃 |
| `=0`        | 走通用 stride-aware A2A，启动成功         |
| 未设置      | 自动选择 Direct，结果取决于机器和布局     |

## 6. 在当前修复分支验证

当前分支已经加入布局检查，所以同样设置：

```
export VLLM_USE_DIRECT_DCP_A2A=1
```

再执行上面的启动命令，预期行为是：

-   支持的 packed 布局继续走 Direct A2A。
-   不支持的 head-major/大步长布局自动回退到通用 A2A。
-   不再抛出 `partial output must have packed heads...`。

不依赖模型和 GPU 后端的最小测试为：

```
.venv/bin/python -m pytest \
  tests/distributed/test_dcp_direct_a2a_lse_reduce.py \
  -k "layout or stride" -vv
```

目前最现实的判断是：两张 H100 适合验证修复的分发逻辑；想稳定重现原始崩溃，仍然需要原作者的 GLM 量化 checkpoint 和能产生相同 head-major MLA 输出布局的 Blackwell 后端。











下面是从服务配置、Attention 计算，到 Direct A2A 报错的完整树状流程。

```
vLLM 服务启动
│
├── 并行配置
│   ├── TP = 4
│   ├── DCP = 4
│   └── dcp_comm_backend = "a2a"
│
├── 初始化 MLA Attention
│   │
│   └── 创建 MLADCPManager
│       │
│       ├── 检查是否使用 A2A
│       │   └── self.use_a2a = True
│       │
│       └── get_direct_dcp_a2a_workspace(...)
│           │
│           ├── GPU/通信环境支持 Direct DCP
│           ├── dtype 是 FP16/BF16
│           ├── world_size > 1
│           └── 创建 DirectDCPA2AWorkspace
│
│           结果：
│           MLADCPManager.combine
│               ↓
│           _direct_workspace_combine(...)
│
└── 开始 CUDA Graph capture / Decode
    │
    └── 每一个 MLA Attention 层
        │
        ├── 1. 准备 Query
        │   │
        │   ├── 当前 rank 产生本地 Query heads
        │   └── DCP Query Gather
        │       └── 每个 DCP rank 获得需要的 Query
        │
        ├── 2. 每个 rank 计算局部 Attention
        │   │
        │   └── attention_backend.forward_mqa(...)
        │       │
        │       ├── 使用本 rank 持有的 KV cache shard
        │       ├── 计算 partial_output
        │       └── 计算 partial_lse
        │
        │       报告者环境返回：
        │
        │       partial_output
        │       ├── shape  = [T, 64, 512]
        │       ├── stride = (512, 1572864, 1)
        │       └── 物理布局接近 [head, token_capacity, dim]
        │
        │       partial_lse
        │       ├── shape  = [T, 64]
        │       └── stride = (64, 1)
        │
        ├── 3. 合并不同 DCP rank 的局部结果
        │   │
        │   └── dcp_manager.combine(
        │           partial_output,
        │           partial_lse
        │       )
        │
        └── 4. 进入当前 _direct_workspace_combine()
            │
            ├── 当前 Python 只检查 token 数
            │   │
            │   ├── T <= direct_workspace.max_num_tokens
            │   │   └── 是
            │   │
            │   └── 选择 Direct A2A
            │
            └── direct_workspace.lse_reduce(...)
                │
                └── torch.ops._C.direct_dcp_a2a_lse_reduce(...)
                    │
                    └── C++ 参数检查
                        │
                        ├── output.stride(2) == 1
                        │   └── 1 == 1                       ✅
                        │
                        ├── output.stride(1) == head_dim
                        │   └── 1572864 == 512               ❌
                        │
                        ├── output.stride(0) >= H × D
                        │   └── 512 >= 64 × 512
                        │       512 >= 32768                 ❌
                        │
                        └── output.stride(0) % 8 == 0
                            └── 512 % 8 == 0                 ✅

                            最终条件：
                            true && false && false && true
                                         │
                                         └── false
                                             │
                                             └── STD_TORCH_CHECK 抛异常
                                                 │
                                                 └── RuntimeError:
                                                     partial output must have
                                                     packed heads and an
                                                     aligned token stride
```

当前错误的核心分支可以简化为：

```
partial_output
shape  正确：[T, H, D]
stride 不兼容
        │
        ▼
Python 只检查 token capacity
        │
        ├── token 超容量
        │       └── generic A2A
        │
        └── token 未超容量
                └── Direct A2A
                        │
                        └── C++ 检查 stride
                                │
                                └── 不满足
                                        └── 直接报错
```



最终整体结构就是：

```
Attention backend
│
├── 返回 packed 布局
│   └── Direct A2A
│       └── 高性能路径
│
└── 返回任意合法非 packed 布局
    └── Generic A2A
        └── 兼容路径
```

关键点是：Attention backend 返回的 shape 可以正确，但 stride 不一定符合 Direct kernel 的额外要求。修复后的 `MLADCPManager` 是两种布局之间的安全分流器。
