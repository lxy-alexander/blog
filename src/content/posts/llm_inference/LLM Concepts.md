---
title: "LLM Concepts"
published: 2026-08-10
description: "LLM Concepts"
image: ""
tags: ["llm_inference","LLM Concepts"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-10T22:10:34.234.410929590Z"
---

## MHA | MQA | GQA

1）**MHA — Multi-Head Attention**

Each Query head has its own corresponding Key and Value heads.

2）**MQA — Multi-Query Attention**

Multiple Query heads share a single Key head and a single Value head.

3）**GQA — Grouped-Query Attention**

Query heads are divided into groups, and the Query heads within each group share the same Key and Value heads.



## Slice Runtime

```python
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve repro/kimi-k3-slice \
  --trust-remote-code \
  --skip-tokenizer-init \
  --load-format dummy \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --attention-backend TRITON_MLA \
  --spec-method dspark \
  --spec-model repro/kimi-k3-dspark-slice \
  --spec-tokens 3 \
  --port 8000
```





## emplace_back

vector `push_back` copies or moves an existing object into the vector, while `emplace_back` constructs the object directly in place. `push_back` 将已有对象拷贝或移动进 vector，而 `emplace_back` 直接在 vector 尾部原地构造对象。







## 量化

### 分组量化

By splitting the weights into groups and letting each group have its own scale and offset, you can compress large floating-point tensors into lower-bit representations. This reduces the storage and bandwidth requirements, speeding up computation. 

量化是把原始的浮点数映射成较少位数的整数，这本质上是对精度的压缩。当你把一组数字共用一个缩放尺度时，组内的不同值会被“近似”成有限的离散整数。这种近似会带来一些精度损失，尤其是当数值跨度很大或者细微变化很重要时。不过，合理的分组和动态调整缩放尺度，可以在保持大部分精度的同时，显著减少模型的存储与计算量。

量化就是把一把很长的尺子变成一小段一小段的尺子。因为不同区间的数字范围不一样，用同一把尺子量会不精确。分组后，每个小组都用适合自己的尺子，量起来精度就更好。现代量化技术就是尽量让这种分组更智能、更灵活，让模型在减少计算量的同时，仍然保持不错的表现。

常见的量化技术有几类。首先有定点量化，比如用整数来表示原来的浮点数。其次有对称量化和非对称量化，对称量化让数字围绕零对称，而非对称量化可以针对偏移的数值范围。还有像 GPTQ 和 AWQ 这样的感知量化方法，它们在量化时考虑了模型的误差敏感度。此外还有分组量化，也就是我们刚谈到的那种分批共享尺度的方法。不同技术各有权衡，目标都是在压缩的同时尽量保持模型的准确性。



## shared_ptr

shared_ptr 的核心逻辑——通过引用计数自动管理对象的生命周期，一旦没人再需要那个对象，它就会被自动清理掉！



etcd



minikube





## File discovery

File discovery 是一种简单的服务发现方式，本质上就是用文件来记录服务的位置。这种方式很简单，适合单机或小规模环境，但它只做服务列表，不提供自动调度、监控、容错等能力。

Kubernetes 是一个完整的容器编排系统，除了服务发现，它还能自动调度工作负载、管理资源、监控健康状态，并在故障时自动恢复。简单说，file discovery 是一个小工具，而 Kubernetes 是一个强大的全面平台，用来管理大规模分布式系统。



## Kubernetes 在 llm-d 的作用

Kubernetes 在 llm-d 的底层，主要负责资源编排和容器管理。它托管 llm-d 的各个组件（比如 EPP 和模型服务），并且负责调度、扩缩容、服务发现等基础设施层面的工作。简单来说，Kubernetes 是 llm-d 运行的环境管理者，而 llm-d 则负责模型推理请求的路由和调度。





## GPTQ 和 AWQ

1）**GPTQ：一种面向 GPT/LLM 的 Post-Training Quantization（PTQ，训练后量化）算法。**
通常指一种针对 GPT/大语言模型的**训练后量化（Post-Training Quantization, PTQ）**方法。其经典论文名称是 *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*。

2）**AWQ**：Activation-aware Weight Quantization
中文一般译为**激活感知权重量化**，通过激活值识别重要权重，从而降低低比特权重量化带来的精度损失。

简单记忆：**GPTQ = 基于误差优化的 PTQ；AWQ = 根据 Activation 判断哪些 Weight 更重要。**

GPTQ 和 AWQ 都是“大模型权重量化”方法，常见用途是把 FP16 权重压成 INT4，从而减少显存。

1）GPTQ 是什么、怎么做

GPTQ 的核心是：把一组浮点权重压成低比特整数，同时尽量减小量化误差。

例如原始权重是：

```text
[0.12, 0.55, -0.30, 0.90]
```

假设只能用 4bit 表示，量化后可能变成：

```text
[1, 4, -2, 7]
```

再通过一个 scale 还原：

```text
scale = 0.13
```

所以实际推理时近似成：

```text
[0.13, 0.52, -0.26, 0.91]
```

GPTQ 不只是简单四舍五入，它会利用一小批校准数据，观察这一层的输入，再调整后续权重，让整个层输出的误差尽量小。

优点：成熟、兼容性好、4bit 效果不错。
缺点：量化过程相对慢，而且主要关注整体误差。

2）AWQ 是什么、怎么做

AWQ 也是把 FP16 权重量化成 INT4，但它多做了一件事：

先看真实输入激活，找出“特别重要”的权重。

比如：

```text
权重：
[0.12, 0.55, 0.30, 0.90]

对应输入激活：
[0.2, 0.1, 12.0, 0.3]
```

第三个权重虽然只有 `0.30`，但它对应的激活是 `12.0`：

```text
0.30 × 12 = 3.6
```

如果把它粗暴量化成：

```text
0.30 → 0.20
```

输出就变成：

```text
0.20 × 12 = 2.4
```

误差达到：

```text
3.6 - 2.4 = 1.2
```

所以 AWQ 会认为这个权重很重要，通过缩放等方式，让它在量化后尽量保留精度。

3）核心区别

可以记成：

```text
GPTQ：
重点是“怎么让整体量化误差更小”

AWQ：
重点是“先找出对真实输出影响最大的权重，再重点保护”
```

因此两者都能做到 4bit、省显存；AWQ 的特点是利用激活信息判断哪些权重更值得保留。





## MLA

MLA 指 **Multi-head Latent Attention**。它基本沿用了 DeepSeek-V3 的 MLA 思路：**把每个 token 的 K/V 压缩到一个低维 latent，只缓存 latent，需要做 Attention 时再恢复/吸收到计算中**。

最应该和这三个对比：

1）MHA：Multi-Head Attention
2）GQA：Grouped-Query Attention
3）MQA：Multi-Query Attention

因为它们都在解决同一个核心问题：

$\boxed{\text{历史 token 的 K/V 到底怎么存}}$

而 MLA 是其中更进一步的一种设计。

最推荐你先只对比：

$\boxed{\text{MHA vs MLA}}$

把这个搞懂以后，再看 GQA/MQA 会很容易。

1）MHA 怎么做

假设有 64 个 attention heads。

每个 head 都有自己独立的：

$K_i,\ V_i$

所以一个历史 token 要保存：

$K_1,V_1,K_2,V_2,\dots,K_{64},V_{64}$

如果每个 K、V 都是 256 维：

$64\times(256+256) = 32768$

所以：

$\boxed{\text{MHA：每个 head 都存自己的 K/V}}$

这就是 KV Cache 很大的原因。

2）MQA 怎么做

MQA 比较激进。

虽然 Q 还是有 64 个 heads：

$Q_1,Q_2,\dots,Q_{64}$

但是所有 Query head 共用同一份：

$K,V$

也就是：

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤
... │── 共用 K,V
Q64─┘
```

于是一个 token 只保存：

$256+256=512$

而不是：

$32768$

所以 MQA 极大节省 KV Cache。

但代价是：

$\boxed{\text{所有 head 看同一套 K/V}}$

表达能力可能受影响。

3）GQA 怎么做

GQA 在 MHA 和 MQA 中间。

例如：

64 个 Q heads，

但只有 8 个 KV heads。

每 8 个 Query head 共用一套 K/V。

比如：

```text
Q1-Q8     → K1,V1
Q9-Q16    → K2,V2
...
Q57-Q64   → K8,V8
```

那么 KV Cache 是：

$8\times(256+256) = 4096$

所以：

$\text{MHA}:32768$$\text{GQA}:4096$$\text{MQA}:512$

这是传统路线。

4）MLA 的思路不一样

MLA 不直接说：

>   到底留多少个 KV heads？

而是说：

>   干脆不要缓存完整 K/V head。

它只缓存一个：

$c^{KV}\in\mathbb R^{512}$

然后不同 head 可以通过各自的权重，从这个共享 latent 中得到自己需要的信息。

概念上：

```text
                 cKV 512维
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
     Head 1       Head 2       Head 64
     自己的K/V     自己的K/V      自己的K/V
```

所以 MLA 有点像同时想要：

$\boxed{\text{MQA 的小 Cache}}$

和：

$\boxed{\text{MHA 的多 head 表达能力}}$

这是它最值得理解的地方。

5）四种 Attention 放一起看

假设都是：

$64\text{ 个 Query heads}$

那么：

| 方法 | KV 怎么存       | 直觉                                      |
| ---- | --------------- | ----------------------------------------- |
| MHA  | 64 套 K/V       | 每个人一本自己的书                        |
| GQA  | 例如 8 套 K/V   | 8 人共用一本                              |
| MQA  | 1 套 K/V        | 64 人共用一本                             |
| MLA  | 1 个压缩 latent | 64 人共用一个数据库，但各自用不同方式查询 |

最后这个类比很重要。

MQA 是：

$\boxed{\text{所有 head 真的共用同一个 K/V}}$

而 MLA 是：

$\boxed{\text{所有 head 共用同一个 latent，但可以得到不同的信息}}$

这两者不是一回事。







## DCP

如果你这里说的是 GLM / MLA 推理里的 **DCP**，它通常指：

$\boxed{\text{Decode Context Parallelism}}$

也就是“解码阶段的上下文并行”。它不是 MLA 本身的数学公式，而是**多 GPU 推理时怎么把 MLA 的 KV Cache 分摊到不同 GPU 上**。([ROCm](https://rocm.docs.amd.com/projects/atom/en/main/context_parallel_guide.html?utm_source=chatgpt.com))

1）为什么 MLA 还需要 DCP

MLA 已经把每个 token 的 KV 压成类似：

$c^{KV}\in\mathbb R^{512}$

这确实很省。

但如果上下文特别长，比如 100 万 token，那么：

$1000000\times512$

再乘很多层，仍然很大。

而且 MLA 的 latent KV 是所有 Query head 共享的，所以普通 Tensor Parallel 很难继续按“KV head”拆。结果常常是每张 GPU 都复制一份完整 latent KV Cache。([vLLM](https://vllm.ai/blog/2026-08-07-decode-context-parallelism?utm_source=chatgpt.com))

2）DCP 就把“历史 token”拆开

比如有 6 个历史 token：

$t_0,t_1,t_2,t_3,t_4,t_5$

有两个 GPU。

DCP 可以这样分：

$GPU_0:\ t_0,t_2,t_4$$GPU_1:\ t_1,t_3,t_5$

也就是说：

$\boxed{\text{不是按 head 拆，而是按 sequence/token 拆 KV Cache}}$

每张 GPU 只保存一部分历史 token 的 KV latent。([ROCm](https://rocm.docs.amd.com/projects/atom/en/main/context_parallel_guide.html?utm_source=chatgpt.com))

3）Attention 怎么算

当前 Query 是：

$Q$

先让两张 GPU 都拿到需要的 Q。

然后：

$GPU_0$

只计算：

$Q\cdot\{K_0,K_2,K_4\}$

而：

$GPU_1$

计算：

$Q\cdot\{K_1,K_3,K_5\}$

各自得到一部分 Attention 结果。

最后两张 GPU 再通信，把部分结果合起来，得到完整 Attention 输出。DCP 的实现会处理 softmax 归一化对应的 LSE 和 partial output 合并。([vLLM](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/dcp/?utm_source=chatgpt.com))

4）所以你可以把关系记成

MLA：

$\boxed{\text{把每个 token 的 KV 压小}}$

DCP：

$\boxed{\text{把很多 token 的 KV 分给不同 GPU}}$

两者解决不同维度的问题。

例如：

$\text{原始 KV}$

先经过 MLA：

$32768 \rightarrow 576$

然后如果还有 100 万 token，DCP 再把这 100 万个 token 分给多个 GPU。

所以是：

$\boxed{ \text{MLA：压“每个 token”} }$$\boxed{ \text{DCP：拆“token 数量”} }$

如果你刚才的 `dcp` 其实是指源码里的某个变量名而不是 Decode Context Parallelism，把那一行源码贴出来，我可以直接按那一行解释。
