---
title: "vllm forward modular"
published: 2026-08-26
description: "vllm forward modular"
image: ""
tags: ["llm_inference","vllm forward modular"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-27T03:29:43.398.252025935Z"
---

```python
GPUModelRunner.execute_model()
└── self.model(...)
    └── Qwen3.5 DecoderLayer.forward()
        └── Qwen3NextSparseMoeBlock.forward()
            └── MoERunner.forward()
                └── MoERunner._forward_impl()
                    ├── gate(hidden_states)
                    └── MoERunner._apply_quant_method()
                        ├── router.select_experts()
                        │   ├── topk_ids
                        │   └── topk_weights
                        └── RoutedExperts.forward_modular()
                            └── MoeWNA16Method.apply()
                                └── self.moe_kernel.apply()
                                    └── TritonWNA16Experts.apply()
                                        ├── W13
                                        ├── activation
                                        └── W2
```

按你最初的服务配置带入：

```
model: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
quantization: moe_wna16

tensor-parallel-size: 4
data-parallel-size: 1

max-num-seqs: 8
max-num-batched-tokens: 16384
```

模型在每张卡上的 MoE 关键尺寸为：

```
全局专家数 E             = 256
每个token选择专家数 topk = 8
hidden_size K            = 3072
全局intermediate         = 1024
TP=4后每卡intermediate   = 1024 / 4 = 256
W13输出宽度 N            = gate 256 + up 256 = 512
group_size               = 128
激活dtype                = BF16
权重dtype                = packed INT4/uint8
```

## 一次 `MoeWNA16Method.apply()` 收到什么

假设当前 ModelRunner 执行形状是 `M`：

```
MoeWNA16Method.apply(
    layer=当前MoE层,
    x=hidden_states,             # [M, 3072], BF16
    topk_weights=topk_weights,   # [M, 8], FP32/BF16
    topk_ids=topk_ids,           # [M, 8], int32/int64
    shared_experts=...,
    shared_experts_input=...,
)
```

调用发生在 Router 之后：

```
hidden_states [M,3072]
        │
        ▼
Router Gate
        │
        ├── topk_ids     [M,8]
        └── topk_weights [M,8]
        │
        ▼
MoeWNA16Method.apply()
```

它不关心当前是 prefill 还是 decode，只要执行到了这个 MoE 层，就会调用。

## `apply()` 内部传入的每卡权重

### W13 权重

```
w13_qweight.shape
= [E, 2×intermediate_per_rank, hidden_size/2]
= [256, 512, 1536]
```

最后一维除以 2，是因为一个 `uint8` 保存两个 INT4。

Scale：

```
w13_scales.shape
= [256, 512, hidden_size/group_size]
= [256, 512, 3072/128]
= [256, 512, 24]
```

### W2 权重

```
w2_qweight.shape
= [E, hidden_size, intermediate_per_rank/2]
= [256, 3072, 128]
```

Scale：

```
w2_scales.shape
= [256, 3072, intermediate_per_rank/group_size]
= [256, 3072, 256/128]
= [256, 3072, 2]
```

## 一次 `apply()` 的数学过程

```
输入
x [M,3072]
│
├── Router已经为每个token选择8个专家
│
▼
W13 GEMM
[M,3072] × 8个专家的W13
→ [M,8,512]
│
▼
SwiGLU
[M,8,512]
→ [M,8,256]
→ view为[M×8,256]
│
▼
W2 GEMM
[M×8,256] × W2
→ [M,8,3072]
│
▼
乘topk_weights并对8个专家求和
→ [M,3072]
```

代码调用：

```
return self.moe_kernel.apply(
    x,                       # [M,3072]
    layer.w13_weight,        # [256,512,1536] uint8
    layer.w2_weight,         # [256,3072,128] uint8
    topk_weights=[M,8],
    topk_ids=[M,8],
    ...
)
```

## 场景一：单请求 Decode，C1

假设只有一个活跃请求，每次生成一个 token：

```
M = 1
```

进入 `apply()`：

```
x.shape            = [1,3072]
topk_ids.shape     = [1,8]
topk_weights.shape = [1,8]
```

W13：

```
[1,3072]
→ 复制/路由到8个专家
→ [1,8,512]
```

Activation：

```
[1,8,512]
→ [8,256]
```

W2：

```
[8,256]
→ [1,8,3072]
```

合并：

```
[1,8,3072]
× topk_weights[1,8]
→ sum(dim=experts)
→ [1,3072]
```

旧 CUDA 判断：

```
num_valid_tokens = M × topk
                 = 1 × 8
                 = 8

num_valid_tokens / E
= 8 / 256
= 0.03125
<= 6
```

因此旧路径应该选择：

```
CUDA W4A16
```

但是当前 Modular 路径：

```
MoeWNA16Method.apply()
└── TritonWNA16Experts.apply()
    ├── W13 → Triton
    └── W2  → Triton
```

仍然固定走 Triton。

## 场景二：8请求并发 Decode，C8

每个请求本 step 生成一个 token：

```
M = 8
```

输入：

```
x                [8,3072]
topk_ids         [8,8]
topk_weights     [8,8]
routed token数   8×8 = 64
```

W13：

```
[8,3072]
→ [8,8,512]
```

Activation：

```
[8,8,512]
→ [64,256]
```

W2：

```
[64,256]
→ [8,8,3072]
```

最终：

```
[8,3072]
```

CUDA 判断：

```
num_valid_tokens / E
= 64 / 256
= 0.25
<= 6
```

旧路径同样应该走 CUDA。

## 场景三：临界值 M=192

```
M = 192
topk = 8
```

路由后：

```
num_valid_tokens
= 192 × 8
= 1536
```

判断：

```
1536 / 256 = 6
```

刚好满足：

```
CUDA W4A16
```

张量形状：

```
输入       [192,3072]
W13输出    [192,8,512]
Activation [1536,256]
W2输出     [192,8,3072]
最终输出   [192,3072]
```

## 场景四：M=193

```
num_valid_tokens = 193 × 8 = 1544
1544 / 256 = 6.03125
```

超过阈值：

```
Triton W4A16
```

所以边界是：

```
M <= 192 → CUDA
M >= 193 → Triton
```

## 场景五：PP2048 Prefill

一个请求 prompt 长度 2048，假设这个 step 一次处理完整 prompt：

```
M = 2048
```

进入 `apply()`：

```
x                [2048,3072]
topk_ids         [2048,8]
topk_weights     [2048,8]
num_valid_tokens 2048×8 = 16384
```

W13：

```
[2048,3072]
→ [2048,8,512]
```

Activation：

```
[2048,8,512]
→ [16384,256]
```

W2：

```
[16384,256]
→ [2048,8,3072]
```

最终：

```
[2048,3072]
```

CUDA判断：

```
16384 / 256 = 64
```

显然：

```
64 > 6
```

因此 PP2048 应该使用：

```
Triton W4A16
```

所以 PP2048 也会调用 `MoeWNA16Method.apply()`，只是它下游应该选择 Triton。

## 场景六：8个PP2048请求合批

你的配置允许：

```
max-num-seqs = 8
max-num-batched-tokens = 16384
```

理论上一个 step 可以接近：

```
8 × 2048 = 16384 tokens
```

因此：

```
M = 16384
```

输入形状：

```
x            [16384,3072]
topk_ids     [16384,8]
routed rows  16384×8 = 131072
```

判断：

```
131072 / 256 = 512
```

所以必定选择：

```
Triton W4A16
```

## Workspace 也由这次 `apply()` 的 M 决定

当前 workspace 计算：

```
workspace13 = (
    M,
    topk,
    max(activation_out_dim, K),
)
```

代入：

```
topk              = 8
activation_out_dim= 256
K                 = 3072
```

得到：

```
workspace13 = [M,8,3072]
```

但实际 activation 只需要：

```
[M,8,256]
```

### PP2048

当前分配：

```
2048 × 8 × 3072 × 2 bytes
= 96 MiB
```

实际 activation：

```
2048 × 8 × 256 × 2 bytes
= 8 MiB
```

冗余：

```
96 - 8 = 88 MiB/卡
```

### M=16384

当前分配：

```
16384 × 8 × 3072 × 2
= 768 MiB
```

实际需要：

```
16384 × 8 × 256 × 2
= 64 MiB
```

冗余：

```
768 - 64 = 704 MiB/卡
```

这是为什么即使小 decode 的计算量不大，大 `max-num-batched-tokens` warmup/profile 仍可能产生明显显存压力。

## TG256-C1 benchmark 中调用多少次

你的 benchmark：

```
input_len  = 1
output_len = 256
concurrency= 1
```

对每个请求：

```
第1次模型执行：
prefill M=1
→ 产生第1个输出token

后续255次：
decode M=1
→ 产生剩余255个token
```

所以每个请求大约执行：

```
256次model forward
```

每次 forward 的每个 MoE 层都执行一次 MoE 计算。

32 个请求、C1 串行时，对每个 MoE 层：

```
32 × 256 = 8192次MoE forward
```

语义上对应约：

```
8192次MoeWNA16Method.apply()
```

如果使用 CUDA Graph replay，Python 函数体不一定实际进入 8192 次，但其中记录的 W13/W2 GPU 工作会重放 8192 次。

## 最关键的理解

`MoeWNA16Method.apply()` 不只在小 batch 调用，也不只在 CUDA 路径调用：

```
C1 decode，M=1       → 调用apply()
C8 decode，M=8       → 调用apply()
prefill，M=2048      → 调用apply()
大合批，M=16384      → 调用apply()
```

区别应该发生在它下游：

```
MoeWNA16Method.apply()
└── WNA16 Experts
    └── 根据当前执行形状M选择kernel
        ├── M<=192 → CUDA W4A16
        └── M>=193 → Triton W4A16
```

而当前代码缺少这个动态选择，所以所有这些 `M` 最终都直接进入 Triton。



