---
title: "GPTQ"
published: 2026-08-22
description: "GPTQ"
image: ""
tags: ["llm_inference","GPTQ"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-23T01:04:56.834.790759367Z"
---

可以把 **GPTQ Int4 权重量化**理解成：把模型里原本用 BF16/FP16 保存的“大量小数权重”，压缩成 4 bit 整数保存；推理时再结合每组权重对应的 scale 等量化参数，把它近似还原出来参与计算。

GPTQ splits the model’s weights into groups, and each group gets its own scaling factor and offset. The beauty of this is that you can fine-tune the range of values just for that group, without messing up the whole network. So, the quantization errors stay localized, and overall, the model keeps more of its accuracy.

带入你这个模型：

```yaml
model: "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
quantization: "moe_wna16"
kv-cache-dtype: "fp8"
tensor-parallel-size: 4
```

核心可以拆成下面几层。

1）原始 BF16 权重是什么样

假设模型某一小段权重原本是：

```text
0.73
-1.21
0.18
2.04
-0.56
...
```

如果用 BF16，每个权重占：

```text
16 bit = 2 Byte
```

Qwen3.5-122B 大约有 1220 亿参数，那么纯理论权重体积：

```text
122B × 2 Byte
≈ 244 GB
```

所以如果是 BF16：

```text
Qwen3.5-122B BF16
理论权重大小 ≈ 244 GB
```

这还没算 KV Cache、激活值、CUDA buffer 等。

2）GPTQ Int4 做了什么

GPTQ 会把类似：

```text
0.73
-1.21
0.18
2.04
-0.56
```

这样的高精度权重，按一定的 group 分组，然后找一个合适的缩放比例。

比如这里只做一个极度简化的示例：

```text
原始权重：

0.73
-1.21
0.18
2.04
-0.56
```

压成 Int4 后可能近似表示成：

```text
3
-4
1
7
-2
```

同时保存一个：

```text
scale ≈ 0.29
```

推理的时候近似恢复：

```text
3 × 0.29  =  0.87
-4 × 0.29 = -1.16
1 × 0.29  =  0.29
7 × 0.29  =  2.03
-2 × 0.29 = -0.58
```

于是：

```text
原始        量化后近似值

 0.73   →    0.87
-1.21   →   -1.16
 0.18   →    0.29
 2.04   →    2.03
-0.56   →   -0.58
```

它不是完全一样，但比较接近。

这就是量化带来的本质：

```text
用更少的 bit 表示权重
↓
显存占用显著下降
↓
代价是产生一些量化误差
```

GPTQ 做得比这种简单四舍五入复杂得多，它会考虑哪些权重误差对模型输出影响比较大，尽可能减少量化造成的精度损失。

3）为什么叫 Int4

因为一个权重大约只占：

```text
4 bit
```

而 BF16 是：

```text
16 bit
```

所以理想情况下：

```text
BF16 → INT4

16 bit → 4 bit

理论压缩比例 ≈ 4 倍
```

带入你的 122B：

```text
BF16：

122B × 16 bit
≈ 244 GB
```

INT4：

```text
122B × 4 bit
≈ 61 GB
```

所以非常直观：

```text
Qwen3.5-122B

BF16       ≈ 244 GB
FP8/INT8   ≈ 122 GB
INT4       ≈ 61 GB
```

但是实际 GPTQ 文件不会正好只有 61GB，因为还需要保存：

```text
scale
zero point
group 信息
量化 metadata
部分未量化层
```

所以实际占用会高于纯理论 61GB。

4）为什么还需要 `moe_wna16`

你的配置：

```yaml
quantization: "moe_wna16"
```

这里可以重点理解成：

```text
W4 A16
```

也就是：

```text
Weight = 4 bit
Activation = 16 bit
```

并不是整个计算过程都使用 INT4。

更准确地说，你的运行方式大致是：

```text
磁盘 / GPU 显存中的模型权重
        ↓
     GPTQ INT4
        ↓
计算时读取 Int4 权重
        ↓
配合 scale 等信息
        ↓
与 FP16/BF16 Activation 做矩阵计算
        ↓
输出 FP16/BF16 类结果
```

所以不要把 GPTQ Int4 理解成：

```text
整个模型所有计算都是 4 bit
```

不是。

主要压的是：

**模型权重。**

5）带入你这个 122B-A10B MoE 更有意思

模型叫：

```text
Qwen3.5-122B-A10B
```

这里两个数字代表不同东西。

大致可以理解：

```text
总参数：
122B

每个 Token 实际参与计算的参数：
约 10B
```

因为这是 MoE 模型。

比如假设模型有很多专家：

```text
Expert 1
Expert 2
Expert 3
Expert 4
Expert 5
...
Expert N
```

当一个 Token 进来：

```text
"北京"
```

Router 不会让所有 Expert 都计算。

可能只选：

```text
Expert 3
Expert 17
Expert 28
...
```

所以每个 Token 实际只走约 10B 参数。

但是有一个非常容易误解的地方：

**122B-A10B 并不意味着只需要存 10B 权重。**

你的机器仍然基本上得把：

```text
122B 总权重
```

准备好。

因此未量化情况下：

```text
122B BF16 ≈ 244GB
```

GPTQ Int4 后：

```text
122B INT4 ≈ 61GB + 量化开销
```

而 A10B 主要帮你降低的是：

```text
计算量
```

不是把模型存储直接变成 10B。

所以可以这样区分：

```text
122B → 决定模型有多少总权重需要保存

A10B → 决定每个 Token 大约计算多少参数

INT4 → 决定这些权重每个参数占多少显存
```

这是理解你这个模型最关键的一点。

6）再带入你的 4 张 GPU

你设置：

```yaml
tensor-parallel-size: 4
```

也就是模型权重分布到 4 张 GPU。

纯理论上：

```text
INT4 权重 ≈ 61 GB
```

平均到 4 卡：

```text
61 / 4
≈ 15.25 GB / 卡
```

所以你可以想象成：

```text
GPU 0      GPU 1      GPU 2      GPU 3

~15GB      ~15GB      ~15GB      ~15GB
权重       权重       权重       权重
```

实际不会恰好是 15.25GB，因为还会有：

```text
GPTQ scale
量化元数据
未量化权重
MoE 分布
通信 buffer
CUDA graph
Activation
KV Cache
vLLM runtime
```

所以实际显存会明显高于这个数字。

比如你是 4×24GB 卡，那么虽然：

```text
理论 INT4 权重：
~15GB/GPU
```

但剩余只有：

```text
24 - 15 ≈ 9GB
```

还要塞 KV Cache、激活、CUDA graph 等，80K 上下文就会非常紧。

7）你的 FP8 KV Cache 又是另一件事

你还有：

```yaml
kv-cache-dtype: "fp8"
```

千万不要把它和 GPTQ Int4 混在一起。

它们压的是完全不同的东西：

```text
GPTQ Int4
    ↓
压模型权重

FP8 KV Cache
    ↓
压上下文缓存
```

比如用户输入：

```text
一篇 50,000 Token 的文档
```

模型在生成过程中需要保存过去 Token 的 Key / Value。

这个东西叫：

```text
KV Cache
```

如果 KV Cache 用 FP16：

```text
每个值 16 bit
```

你这里改成 FP8：

```text
每个值 8 bit
```

理论上 KV Cache 显存可以接近减半。

因此你的整个配置实际上是：

```text
Qwen3.5 122B-A10B

模型权重
    ↓
GPTQ INT4
    ↓
约 4bit / parameter

Activation
    ↓
A16
    ↓
大致 FP16/BF16

KV Cache
    ↓
FP8
    ↓
8bit

并行
    ↓
TP = 4
    ↓
模型拆到 4 GPU
```

8）为什么不直接全部 INT4？

因为神经网络不同部分对低精度的敏感程度不同。

权重是：

```text
相对固定的
```

模型加载后基本不变，因此特别适合提前离线量化。

但是 Activation 是每次推理实时产生：

```text
Token 1 → 一组 activation
Token 2 → 另一组 activation
Token 3 → 又一组
```

数值范围变化比较大。

如果也暴力压到 INT4：

```text
精度可能明显下降
实现也更加复杂
```

所以常见方案就是：

```text
W4A16
```

即：

```text
权重 INT4
激活 16bit
```

用模型质量换很少，换取巨大的权重显存节省。

9）可以用一个“仓库”类比你的配置

把模型想成一个拥有 1220 亿件商品的仓库。

BF16 相当于：

```text
每件商品使用一个 16 单位大小的箱子
```

GPTQ Int4 相当于：

```text
重新压缩包装
每件只用 4 单位空间
```

因此整个仓库体积大约下降到：

```text
1/4
```

但取出来真正工作时：

```text
INT4 商品
↓
根据 scale 解码
↓
进入 16-bit 工作台
↓
进行计算
```

而 MoE 的 A10B 意味着：

```text
仓库里有 122B 商品

但每次处理一个 Token
只需要从其中约 10B 参数对应的专家里工作
```

所以你的组合非常典型：

```text
122B
= 模型容量很大

A10B
= 每 Token 计算量没有 122B Dense 那么夸张

GPTQ Int4
= 让 122B 权重能塞进更少显存

FP8 KV Cache
= 进一步省长上下文的显存

TP4
= 把模型摊到 4 张 GPU
```

如果用一句最简洁的话描述你的配置，就是：

**“把一个总参数 122B、每 Token 激活约 10B 的 MoE 模型，其主要权重从 BF16 的约 244GB 压成 GPTQ Int4 的约 61GB+开销，再拆到 4 张 GPU；计算时主要按 W4A16 路径执行，同时把 80K 长上下文产生的 KV Cache 再用 FP8 压缩。”**
