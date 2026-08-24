---
title: "vllm moe"
published: 2026-08-22
description: "vllm moe"
image: ""
tags: ["llm_inference","vllm moe"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-23T02:35:08.594.736789530Z"
---

下面按你给出的真实配置，把 `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` 从请求进入 ModelRunner，一直代入到 `invoke_fused_moe_wna16_cuda_kernel` 的张量尺寸、TP 切分和 kernel 选择条件。

先给最重要的结论：

>   在这套配置中，`invoke_fused_moe_wna16_cuda_kernel` 主要服务于 decode 和很小的 prefill。
>   当一次 ModelRunner 调度的 token 数 `M ≤ 192` 时，W4A16 MoE 通常会进入该 CUDA kernel；最大 16K 的大块 prefill 会走 WNA16 Triton kernel。

但有一个前提：启动日志必须显示选择了 `VLLM` WNA16 backend。若自动选择了 FlashInfer TRT-LLM backend，则不会经过这个函数。

## 1. 模型的实际架构参数

官方模型配置是：

| 参数                        | 数值      |
| --------------------------- | --------- |
| 总层数                      | 48        |
| hidden size                 | 3072      |
| Routed Experts              | 256       |
| 每 token 选择专家数 Top-K   | 8         |
| 每个 Routed Expert 中间维度 | 1024      |
| Shared Expert 中间维度      | 1024      |
| Attention heads             | 32        |
| KV heads                    | 2         |
| Head dimension              | 256       |
| Full Attention 层           | 12        |
| Linear Attention/GDN 层     | 36        |
| 权重量化                    | GPTQ INT4 |
| GPTQ group size             | 128       |
| 激活 dtype                  | BF16      |
| 模型原生最大长度            | 262144    |

这些数值来自模型的官方 [config.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4/blob/main/config.json)。

48 层的模式为：

```
Layer 0  Linear Attention + MoE
Layer 1  Linear Attention + MoE
Layer 2  Linear Attention + MoE
Layer 3  Full Attention   + MoE
Layer 4  Linear Attention + MoE
...
Layer 47 Full Attention   + MoE
```

也就是每 4 层中：

```
3 × Gated DeltaNet
1 × Full Attention
```

每一层都有一个 Sparse MoE。对应构建代码见 [qwen3_5.py (line 120)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:120)。

## 2. 启动参数对模型的实际影响

### 模型名与服务名

```
model: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
served-model-name: qwen3.5-27b
```

真正加载的是 122B-A10B 模型。

`qwen3.5-27b` 只是 OpenAI API 暴露的别名，例如客户端请求应写：

```
{
  "model": "qwen3.5-27b"
}
```

它不会把模型变成 27B，也不会改变模型架构、权重大小或执行路径。这个别名容易误导监控和使用者，建议改成类似：

```
served-model-name: qwen3.5-122b-a10b-int4
```

### TP/DP

```
tensor-parallel-size: 4
data-parallel-size: 1
```

你没有设置 `enable-expert-parallel`，所以：

```
TP = 4
EP = 1
DP = 1
```

因此不是“每张 GPU 放 64 个专家”，而是：

```
每张 GPU 都保存 256 个 Routed Experts
但每个专家的中间维度按 TP=4 切分
```

每个专家原始中间维度：

```
I = 1024
```

每张 GPU 上：

```
I_local = 1024 / 4 = 256
```

对应代码在 [config.py (line 1350)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/config.py:1350)：

```
intermediate_size_per_partition = intermediate_size // tp_size
```

因此一个 token 选择的仍是全局 8 个专家，但每张 GPU 只计算这 8 个专家各自的四分之一 FFN，最后进行 TP All-Reduce。

### KV Cache

```
kv-cache-dtype: fp8
```

只影响 Full Attention 层的 K/V cache，不影响 MoE 激活。MoE kernel 的输入仍是 BF16，即：

```
W4A16 = INT4 weight + BF16 activation
```

它不会变成 W4A8。

### 最大长度和调度

假设配置中的 `K` 按二进制单位解析：

```
max-model-len = 80K = 81,920 tokens
max-num-batched-tokens = 16K = 16,384 tokens
max-num-seqs = 8
```

含义不同：

-   `max-model-len`：单请求最大上下文长度。
-   `max-num-batched-tokens`：一次 scheduler iteration 最多执行多少 token。
-   `max-num-seqs`：一次最多容纳 8 条请求。

一个 80K prompt 不会一次送入模型，而会被 chunked prefill 分块，例如：

```
80K prompt
├── iteration 1: 16K
├── iteration 2: 16K
├── iteration 3: 16K
├── iteration 4: 16K
└── iteration 5: 16K
```

实际还会受到其他并发请求、缓存命中和调度预算影响。

## 3. 从 ModelRunner 开始的真实调用路径

主要执行链是：

```
GPUModelRunner.execute_model()
│
├── 准备本轮 input_ids / positions
├── 设置 attention metadata
├── prefix-cache / KV offload 处理
│
└── self.model(**model_inputs)
    │
    └── Qwen3_5MoeForConditionalGeneration
        │
        └── language_model: Qwen3_5MoeForCausalLM
            │
            └── Qwen3_5Model
                │
                └── 48 × Qwen3_5DecoderLayer
                    │
                    ├── RMSNorm
                    │
                    ├── Linear Attention 或 Full Attention
                    │
                    ├── RMSNorm
                    │
                    └── Qwen3NextSparseMoeBlock
                        │
                        └── MoERunner
                            ├── Gate: [M,3072] × [3072,256]
                            ├── Top-K: 每 token 选 8 个专家
                            ├── Routed Experts
                            │   ├── W13 expert GEMM
                            │   ├── SwiGLU
                            │   └── W2 expert GEMM
                            ├── Shared Expert
                            └── TP All-Reduce
```

ModelRunner 的模型调用位于 [model_runner.py (line 1674)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1674)。

Qwen3.5 Decoder Layer 的构建位于 [qwen3_5.py (line 120)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:120)。

它继承的 Decoder forward 位于 [qwen3_next.py (line 475)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_next.py:475)。

## 4. 进入 MoE 前的张量

设本轮 ModelRunner 一共执行 `M` 个 token。

由于当前：

```
DP = 1
EP = 1
未开启 sequence-parallel MoE
```

所以四张 GPU 上进入 MoE 的 token 数相同：

```
hidden_states: [M, 3072], BF16
```

Router gate 是 replicated linear：

```
router_logits = hidden_states @ gate_weight.T
```

张量为：

```
hidden_states : [M, 3072] 每个 token 都对应一个 3072 维向量。
gate_weight   : [256, 3072] “Gate”其实就是路由器，它根据输入计算每个专家的权重或分数，然后决定哪些专家会参与当前的计算
router_logits : [M, 256] router_logits 就是 Router 给每个 token 对所有专家打出来的原始分数。
```

然后每个 token 选择 8 个专家：

```
topk_ids     : [M, 8]
topk_weights : [M, 8]
```

代码位于 [moe_runner.py (line 579)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:579)。

## 5. TP=4 后的专家权重尺寸

每张 GPU 保存 256 个专家，但每个专家中间维度为 256。

### W13：Gate + Up Projection

每个专家逻辑计算：

```
[3072] → gate[256] + up[256]
```

合并之后输出维度：

```
2 × 256 = 512
```

未量化的逻辑 weight shape 是：

```
W13 logical: [256 experts, 512, 3072]
```

INT4 每个 byte 打包两个权重，因此实际 `uint8` 张量为：

```
w13_qweight: [256, 512, 3072 / 2]
             [256, 512, 1536]
```

group size 为 128，因此 scale shape：

```
w13_scales: [256, 512, 3072 / 128]
             [256, 512, 24]
```

### W2：Down Projection

SwiGLU 后每个 TP rank 的中间维度是 256：

```
[256] → [3072]
```

逻辑 weight：

```
W2 logical: [256 experts, 3072, 256]
```

实际 INT4 packed weight：

```
w2_qweight: [256, 3072, 256 / 2]
            [256, 3072, 128]
```

scale：

```
w2_scales: [256, 3072, 256 / 128]
           [256, 3072, 2]
```

这些 shape 的创建逻辑位于 [moe_wna16.py (line 244)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:244)。

## 6. 以 8 路并发 decode 为例

假设 8 个请求都进入 decode，每个请求本轮生成一个 token：

```
M = 8
```

进入一个 MoE 层：

```
hidden_states: [8, 3072], BF16
topk_ids:      [8, 8]
topk_weights:  [8, 8]
```

总路由项数：

```
num_valid_tokens = M × top_k
                 = 8 × 8
                 = 64
```

即这一轮最多形成 64 个“token-expert pair”。

### 第一次专家 GEMM：W13

输入：

```
A = hidden_states
A shape = [8, 3072]
```

权重：

```
B = w13_qweight
B shape = [256, 512, 1536], uint8 packed INT4
B_scale = [256, 512, 24], BF16
```

输出：

```
C = intermediate_cache1
C shape = [8, 8, 512], BF16
```

含义是：`max_batch_tokens` 通常是调度器允许一个 batch 里**最多放多少 token 的上限**。 `M = 8` 更准确地说是**这一轮 decode 实际参与计算的 token 数**

```
8 tokens
× 每 token 8 experts
× 每个专家输出 512 = 2 × 256
```

调用参数近似为：

```
invoke_fused_moe_wna16_cuda_kernel(
    A=[8, 3072],
    B=[256, 512, 1536],
    C=[8, 8, 512],
    B_scale=[256, 512, 24],
    topk_weights=None,
    mul_routed_weight=False,
    top_k=8,
    block_shape=[0, 128],
)
```

第一次 GEMM 不乘 router weight：

```
mul_routed_weight = False
```

### SwiGLU

W13 输出的最后一维 512 被拆成：

```
gate: [8, 8, 256]
up:   [8, 8, 256]
```

执行：

```
silu(gate) × up
```

得到：

```
intermediate_cache2: [8 × 8, 256]
                     [64, 256]
```

### 第二次专家 GEMM：W2

输入已经展开成 64 个 token-expert pair：

```
A = [64, 256]
```

权重：

```
B = w2_qweight
B shape = [256, 3072, 128]
B_scale = [256, 3072, 2]
```

输出：

```
C = [8, 8, 3072]
```

调用参数近似为：

```
invoke_fused_moe_wna16_cuda_kernel(
    A=[64, 256],
    B=[256, 3072, 128],
    C=[8, 8, 3072],
    B_scale=[256, 3072, 2],
    topk_weights=[8, 8],
    mul_routed_weight=True,
    top_k=1,
    block_shape=[0, 128],
)
```

注意第二次的 `top_k=1` 并不表示模型改成 Top-1。因为输入已经从：

```
[8 tokens, 8 experts]
```

展开为：

```
[64 token-expert pairs]
```

所以：

```
64 × 1 = 8 × 8
```

第二次 GEMM 会乘 router weight。

最后：

```
moe_sum([8,8,3072]) → [8,3072]
```

然后四个 TP rank 的部分结果执行 All-Reduce，得到完整的 MoE 输出。

## 7. 为什么 decode 会进入 CUDA kernel

CUDA 选择条件在 [fused_moe.py (line 1288)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:1288)：

```
current_platform.is_cuda()
and bit == 4
and group_size in [32, 64, 128]
and num_valid_tokens / num_experts <= 6
```

代入当前模型：

```
bit = 4
group_size = 128
num_experts = 256
num_valid_tokens = M × 8
```

条件变成：

```
(M × 8) / 256 ≤ 6

M / 32 ≤ 6

M ≤ 192
```

因此运行时分界大致为：

| 本轮 token 数 M | 路由项 M×8 | 平均每专家路由数 | kernel |
| --------------- | ---------- | ---------------- | ------ |
| 1               | 8          | 0.03125          | CUDA   |
| 8               | 64         | 0.25             | CUDA   |
| 32              | 256        | 1                | CUDA   |
| 128             | 1024       | 4                | CUDA   |
| 192             | 1536       | 6                | CUDA   |
| 193             | 1544       | 6.03125          | Triton |
| 1024            | 8192       | 32               | Triton |
| 16384           | 131072     | 512              | Triton |

所以你的典型运行状态是：

```
8 路 decode：
M ≤ 8
→ invoke_fused_moe_wna16_cuda_kernel

大块 chunked prefill：
M 接近 16384
→ invoke_fused_moe_wna16_triton_kernel

prefill + decode 混合批：
根据本轮总 M 是否超过 192 动态选择
```

## 8. 16K prefill 的张量尺寸

当 scheduler 给出完整的 16K chunk：

```
M = 16,384
```

Router 输出：

```
topk_ids: [16,384, 8]
路由项数: 16,384 × 8 = 131,072
```

W13：

```
A: [16,384, 3072]
B: [256, 512, 1536]
C: [16,384, 8, 512]
```

W13 输出大小：

```
16,384 × 8 × 512 × 2 bytes
= 134,217,728 bytes
= 128 MiB
```

SwiGLU 输出：

```
[131,072, 256], BF16
```

大小：

```
131,072 × 256 × 2
= 64 MiB
```

W2 输出：

```
[16,384, 8, 3072], BF16
```

大小：

```
16,384 × 8 × 3072 × 2
= 768 MiB
```

仅 W2 的展开输出就接近 768 MiB/GPU。vLLM 会复用 workspace，但 16K MoE prefill 的临时显存峰值仍然很明显。

这也是为什么：

```
max-num-batched-tokens: 16K
```

不只是调度吞吐参数，也直接影响 MoE workspace 峰值。

如果启动时显存紧张或运行中出现较大的临时显存压力，可以优先尝试：

```
max-num-batched-tokens: 8192
```

甚至：

```
max-num-batched-tokens: 4096
```

代价是超长 prompt 需要更多轮 chunked prefill。

## 9. WNA16 backend 选择的关键区别

你的配置只写了：

```
quantization: moe_wna16
```

但没有指定：

```
--moe-backend
```

因此 backend 是自动选择的，优先顺序见 [int_wna16.py (line 112)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/oracle/int_wna16.py:112)：

```
FLASHINFER_TRTLLM
→ MARLIN
→ BATCHED_MARLIN
→ VLLM
→ TRITON
→ HUMMING
→ EMULATION
```

对 `MoeWNA16Config`：

-   Marlin backend 会因 checkpoint layout 被跳过。
-   若 FlashInfer TRT-LLM 支持当前 GPU和配置，可能优先选它。
-   否则 CUDA 平台通常会选 `VLLM`。
-   `VLLM` backend 才会在运行时根据 `M ≤ 192` 动态切换 CUDA/Triton。

你需要在启动日志中确认：

```
Using 'VLLM' WNA16 MoE backend.
Using VllmWNA16Experts
```

如果日志是：

```
Using 'FLASHINFER_TRTLLM' WNA16 MoE backend.
```

则调用链不会进入 `invoke_fused_moe_wna16_cuda_kernel`。

还有一个容易误解的地方：显式指定

```
--moe-backend triton
```

会选择 `TritonWNA16Experts`，它会固定调用 Triton WNA16 kernel，反而不会调用这个原生 CUDA kernel。

## 10. Python 到 CUDA 的最终链路

VLLM backend 的 runtime dispatch：

```
VllmWNA16Experts.apply()
└── VllmWNA16Experts._invoke_wna16_kernel()
    └── dispatch_fused_moe_kernel()
        ├── 小 M: invoke_fused_moe_wna16_cuda_kernel()
        └── 大 M: invoke_fused_moe_wna16_triton_kernel()
```

`invoke_fused_moe_wna16_cuda_kernel()` 位于 [fused_moe.py (line 615)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:615)，调用：

```
ops.moe_wna16_gemm(...)
```

Python custom op wrapper 位于 [_custom_ops.py (line 2324)](/data/home/xli49/lxy/vllm/vllm/_custom_ops.py:2324)。

最后进入 [moe_wna16.cu (line 277)](/data/home/xli49/lxy/vllm/csrc/libtorch_stable/moe/moe_wna16.cu:277)：

```
moe_wna16_gemm(...)
└── run_moe_wna16_gemm(...)
    └── moe_wna16_gemm_kernel<<<...>>>()
```

在 Nsight Systems/Compute 中看到的通常是底层：

```
moe_wna16_gemm_kernel
```

而不是 Python 函数名。

## 11. KV Cache 的数值估算

该模型只有 12 个 Full Attention 层存传统 K/V cache。

每个 TP rank：

```
num_kv_heads = 2
TP = 4
```

由于 KV heads 少于 TP 数量，每张 GPU 至少持有一个 KV head，部分 KV head 会被复制。

每 token、每 Full Attention 层、每 GPU：

```
K: 1 head × 256 × FP8
V: 1 head × 256 × FP8

= 2 × 256 × 1 byte
= 512 bytes
```

12 层：

```
512 × 12 = 6144 bytes/token/GPU
```

单个 80K 请求：

```
81,920 × 6144
= 503,316,480 bytes
= 480 MiB/GPU
```

8 个完整 80K 请求：

```
480 MiB × 8
= 3.75 GiB/GPU
```

这是传统 Full Attention KV 的理论有效数据量，不包括：

-   block/page 对齐；
-   cache allocator 预留；
-   GDN recurrent state；
-   prefix-cache 副本；
-   workspace；
-   metadata；
-   offload staging buffer。

### GDN 状态

36 个 Linear Attention 层使用 recurrent state，而不是随序列长度线性增长的普通 KV。

每层、每请求、每 GPU大约：

```
Conv state:
(12288 / TP4) × 3 × BF16
= 3072 × 3 × 2
= 18,432 bytes

Temporal state:
(64 / TP4) × 128 × 128 × FP32
= 16 × 128 × 128 × 4
= 1,048,576 bytes
```

合计：

```
约 1.018 MiB / GDN layer / request / GPU
```

36 层：

```
约 36.6 MiB / request / GPU
```

8 个请求约：

```
约 293 MiB/GPU
```

启用 prefix caching 后，GDN 的 checkpoint/cache 管理会让实际占用比“当前 recurrent state”更复杂，因此以上是基础状态估算，不是完整 cache pool 预测。

## 12. 64 GiB KV offload 的含义

```
kv-offloading-size: 64
```

当前代码定义为：

```
64 GiB 是所有 TP ranks 合计的 CPU offload 容量
```

不是每个 GPU 64 GiB。TP=4 时，概念上约为：

```
64 / 4 = 16 GiB host buffer per TP rank
```

配置说明见 [cache.py (line 241)](/data/home/xli49/lxy/vllm/vllm/config/cache.py:241)。

它的作用是把已经产生的、适合 offload 的 cache block 放入 CPU pinned memory，命中时再异步搬回 GPU。它：

-   不会把模型权重 offload；
-   不直接增加单请求允许的 `max-model-len`；
-   不能替代 GPU 上正在参与当前计算的 KV；
-   prefix cache 命中时收益最大；
-   若 PCIe 带宽不足，重新载入可能增加首 token 延迟。

## 13. Prefix caching 在该模型上的行为

```
enable-prefix-caching: true
```

对重复 system prompt、相同工具定义、相同长文档前缀很有价值。

Qwen3.5 是 hybrid model，不能使用 Mamba cache mode `all`。代码明确拒绝该模式，见 [qwen3_5.py (line 315)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:315)。

当前一般使用 `align` 模式：

```
Full Attention：
缓存普通 KV blocks

GDN layers：
缓存对齐点上的 recurrent state/checkpoint

重复前缀：
直接复用 Full Attention KV + 对齐的 GDN 状态
```

你的 chat template、tools 和 system prompt 如果稳定，prefix caching 与 64 GiB CPU offload 的组合是有意义的。

## 14. `disable-custom-all-reduce`

```
disable-custom-all-reduce: true
```

MoE 在每个 TP rank 上只计算专家中间维度的四分之一：

```
每 rank I_local = 256
```

W2 后每个 rank 得到一个局部 `[M,3072]` 贡献，必须合并：

```
GPU0 partial output
GPU1 partial output
GPU2 partial output
GPU3 partial output
        ↓
     All-Reduce
        ↓
完整 [M,3072]
```

该参数不会取消 All-Reduce，只是禁用 vLLM 自定义 All-Reduce，通常退回 NCCL 等标准实现。

因此它不会影响是否调用 WNA16 CUDA kernel，但可能影响：

-   Attention 输出 TP 通信；
-   MoE W2 后的 TP 汇总；
-   端到端 latency。

如果四张 GPU 在同一节点、NVLink 拓扑良好，自定义 All-Reduce 可能更快；禁用它通常是为了稳定性、兼容性或规避拓扑问题。

## 15. 其他服务参数的位置

下面这些参数不改变 `moe_wna16_gemm` 的矩阵尺寸：

| 参数                           | 所在阶段                  |
| ------------------------------ | ------------------------- |
| `enable-auto-tool-choice`      | 请求解析/输出解释         |
| `tool-call-parser=qwen3_coder` | 模型输出解析              |
| `reasoning-parser=qwen3`       | reasoning 内容拆分        |
| `chat-template`                | 输入 token 化之前         |
| `xgrammar`                     | logits 约束与 sampling    |
| `disable_any_whitespace`       | structured output grammar |
| `prompt-tokens-details`        | usage 统计                |
| `per-request-metrics`          | 指标采集                  |
| `served-model-name`            | API 模型别名              |

其中 structured output 会影响 logits masking/sampling，但不会改变 Decoder 内部 MoE 的权重 shape。

## 16. 配置中的两个注意点

第一，服务名：

```
served-model-name: qwen3.5-27b
```

与实际 122B-A10B 模型不一致。功能上可以运行，但很容易造成容量规划和调用侧误判。

第二：

```
host: 172.17.0.1
```

如果 vLLM 在 Docker 容器内部启动，`172.17.0.1` 通常是 Docker bridge 的宿主机地址，而不是容器自身持有的地址。此时 server 可能报：

```
Cannot assign requested address
```

容器内一般应绑定：

```
host: 0.0.0.0
port: 8000
```

然后由 Docker/Kubernetes 暴露端口。

## 最终数值调用图

以 8 路 decode 为例：

```
ModelRunner
│  M = 8
│
└── Qwen3.5 Decoder Layer × 48
    │
    ├── Linear Attention × 36
    │   或 Full Attention × 12
    │
    └── Sparse MoE
        │
        ├── Gate
        │   [8,3072] × [3072,256]
        │   → router_logits [8,256]
        │
        ├── TopK=8
        │   → topk_ids [8,8]
        │   → 64 token-expert pairs
        │
        ├── W13 W4A16 GEMM
        │   A      [8,3072] BF16
        │   B      [256,512,1536] UINT8-packed INT4
        │   scales [256,512,24] BF16
        │   C      [8,8,512] BF16
        │
        ├── SwiGLU
        │   [8,8,512] → [64,256]
        │
        ├── W2 W4A16 GEMM
        │   A      [64,256] BF16
        │   B      [256,3072,128] UINT8-packed INT4
        │   scales [256,3072,2] BF16
        │   C      [8,8,3072] BF16
        │
        ├── TopK weighted sum
        │   [8,8,3072] → [8,3072]
        │
        └── TP4 All-Reduce
            四个 rank 的局部专家输出
            → 完整 [8,3072]
```

在 `VLLM` WNA16 backend 下：

```
M=8 ≤ 192
→ invoke_fused_moe_wna16_cuda_kernel
→ ops.moe_wna16_gemm
→ moe_wna16_gemm_kernel
```

而最大 chunked prefill：

```
M=16,384 > 192
→ invoke_fused_moe_wna16_triton_kernel
```
