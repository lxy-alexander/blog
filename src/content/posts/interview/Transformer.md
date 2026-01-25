---
title: Transformer
published: 2026-01-14
description: "Transformer"
image: ""
tags: ["interview", "Transformer"]
category: Guides
draft: false
---

# Transformer

## Tansformer Arch

Transformer follows an encoder–decoder architecture, where each layer consists of multi-head attention, feed-forward networks, residual connections, and layer normalization.





## Self-Attention 机制

Self-attention computes the relevance between different positions(tokens) in a sequence and aggregates information to produce context-aware representations.

Self-Attention 通过计算序列内部各位置之间的相关性，对信息进行加权汇聚，从而得到上下文相关的表示





### KV cache缓存的具体内容

在一个 Transformer 层的推理过程中，对于每一个已经处理过的词（Token），系统会把它的 x \cdot W_k 和 x \cdot W_v 的结果存下来：

- **Key Cache (K 缓存)**：存储了过去所有词的“身份标签”。
  - 数值例子：如果之前有 2 个词，这里就存了 2 条 1 * 4 的向量。
- **Value Cache (V 缓存)**：存储了过去所有词的“内容特征”。
  - 数值例子：这里也存了 2 条 1 * 4 的向量。





###  为什么只缓存 K 和 V，而不缓存 Q？

这是最关键的点。在 **Decode（生成）阶段**：

1. **Q (Query) 不需要缓存**：
   - q_i 代表的是“当前正在生成的这个词”要找什么。
   - 一旦第 i 个词生成完了，它的 q_i 就没用了，因为下一个词会有它自己全新的 q_{i+1}。
2. **K 和 V 必须缓存**：
   - 未来的每一个新词（q_{i+1}, q_{i+2} \dots）在计算时，都需要回过头来和**之前所有词**的 K 做对比，并提取**之前所有词**的 V。
   - 如果不缓存，模型每吐出一个字，都要把前面所有的字重新送进 W_k 和 W_v 算一遍，这会产生巨大的重复计算。



假设对话正在进行，已经输入了“我想吃”：

| **Token** | **x⋅Wk (Key)**         | **x⋅Wv (Value)**       |
| --------- | ---------------------- | ---------------------- |
| **我**    | `[1.1, 0.2, 0.5, 0.1]` | `[0.1, 0.9, 0.0, 0.2]` |
| **想**    | `[0.3, 1.2, 0.1, 0.4]` | `[0.5, 0.1, 0.8, 0.0]` |
| **吃**    | `[0.9, 0.1, 1.1, 0.3]` | `[0.2, 0.2, 0.1, 0.9]` |

**这 6 条向量就是 KV Cache。**

当模型接下来要生成下一个词（比如“苹果”）时：

1. 它算出“苹果”的查询向量 q_{apple}。
2. 它直接从内存里取出上面这 **3 条 Key** 进行点积，算出权重。
3. 它直接从内存里取出上面这 **3 条 Value** 进行加权求和。





### Q、K、V 分别代表什么？

Attention机制本质是一个**信息检索过程**:

- **Q (Query)**: 当前token更关注哪些信息
- **K (Key)**: 其他token的索引标签,用来匹配
- **V (Value)**: 其他token的实际内容,用来提取， 通过乘以 W_V，模型可以将原始 Embedding 投影到一个**特定的特征空间**。**Embedding 的局限性**：它包含了“苹果”的所有信息（水果、电脑公司、重力实验对象）。但在特定的句子里，我们不需要它的**所有**信息。比如，在这一层 Transformer 中，W_V 可能被训练成专门提取“口感”和“味道”相关的特征。那么“苹果”的 Embedding 乘以 W_V 后，得到的 V 向量就只保留了和“好吃、甜”相关的部分，过滤掉了和“公司、乔布斯”相关的部分。**一句话总结**：Embedding 是“我是谁”，而 V 是“在这层讨论中，我能贡献出的相关内容”。



## Q / K / V 从哪里来？

Q、K、V 都来自同一个输入，通过三组不同的线性变换得到。

```
Q = XWq,  K = XWk,  V = XWv
```

Q, K, and V are obtained by applying different linear projections to the same input.



### W_v 是从哪里来的？

和 W_Q, W_K 一样，W_V **不是计算出来的，而是学习出来的权重参数**。

- **初始化**：模型刚开始训练时，W_V 也是一组随机数。
- **训练过程**：
  1. 模型把 Embedding X 乘以随机的 W_V 得到 V。
  2. 模型用这些 V 进行加权融合，得到结果。
  3. 如果最后模型任务（比如翻译）做得不好，它就会通过**反向传播**（Backpropagation）来调整 W_V 里的数值。
  4. 经过几亿次调整，W_V 终于学会了：*“当输入是这类词时，我应该提取它的哪些维度，才能让后面的层更好地理解句子。”*

假设“苹果”的 Embedding X = [1, 2, 0, 0]（前两个维度代表水果特征，后两个代表科技公司特征）。

如果此时我们的 Transformer 层是在分析食谱，那么训练好的 W_V 可能会长这样：







(这组权重的作用是：只保留前两维，抹杀后两维)

计算 V = X * W_V：



[1, 2, 0, 0] * W_V = [3, 3, 0, 0]

1. **输入 X** 还是那个原始的 Embedding。
2. **W_V** 像是一个“过滤器”或“处理器”。
3. **结果 V** 变成了一个纯粹描述“食物属性”的向量。

这样，当“苹果”去和“甜”进行加权融合时，它贡献出来的就是 [3, 3, 0, 0]，而不是杂乱的原始 Embedding。

### 总结：整个链条是这样的

1. **Embedding (X)**：从词表中查出来的原始向量（原材料）。
2. **W_V**：模型训练好的“加工准则”（加工机器）。
3. **Value (V)**：加工后的半成品，专门准备给其他词来“抓取”的。
4. **Attention 权重 (Score)**：决定抓取多少。
5. **输出 (Z)**：抓取并融合后的成品（深度理解了语境的向量）。

**如果没有 W_V，模型就失去了“重新理解和筛选信息”的能力**

如果没有 W_V（即直接用 Embedding X 代替 V）：

1. **无法定制化**：模型无法在不同的层关注不同的特征。比如第一层可能关注语法，第十层可能关注语义。没有 W_V，每一层处理的东西都一样。
2. **没有可训练参数**：如果没有 W_V，模型就失去了一个可以调节的“旋钮”，也就没法通过学习来提升表达能力。
3. **空间限制**：Embedding 的维度通常很大（如 512, 1024），而通过 W_V 我们可以把 V 映射到更小或更聚焦的空间（比如多头注意力中每个头只看一小部分）。







### Why scale by √dₖ（square root of d sub k）?

因为点积的数值会随着维度变大而变大，如果不做缩放，softmax 很容易饱和，导致训练变得困难。所以我们用 d_k 的平方根来进行缩放，这样就能让数值保持在一个合理的范围，让梯度不会太小，训练过程也更加稳定。这正是为什么我们选择用这个平方根作为缩放的原因。





## 为什么要用多头注意力？

多头注意力允许模型在 不同的表示子空间中 并行地关注不同类型的关系。

Multi-head attention allows the model to attend to information from different representation subspaces in parallel.





## 为什么 Transformer 需要位置编码？

因为注意力机制本身不包含顺序信息，位置编码为模型提供序列的位置信息。

Since self-attention is permutation-invariant, positional encoding is required to inject order information into the model.





## Transformer 中的例子来说明一下前向和反向传播是怎么计算的

在 Transformer 的注意力机制里，假设你有一个注意力权重的矩阵 W，前向传播的时候，你会用这个 W 去对输入的查询和键进行线性变换，然后计算注意力分数，最后通过 softmax 得到一个注意力分布。

在反向传播的时候，我们要计算相对于损失函数的梯度是怎么传回到 W 的。我们会先计算损失对 softmax 输出的梯度，再根据链式法则一步步往回传播，求出每一层的梯度。最后，通过反向传播，我们可以求出损失对 W 的梯度，也就是 dW。这个 dW 就是用来更新 W 的方向和大小。

通过这个例子你可以看到，前向传播是计算输出，而反向传播就是用链式法则把梯度一层层传回去，最终得到每个参数的梯度。这样就完成了从前向到反向的一整个过程。





## Softmax用的是什么函数呀?

Softmax 本质上使用的是指数函数。它的计算方式是先对每个输入的数值取指数，然后把所有这些指数值加起来作为分母，每个数值的指数值作为分子，这样就得到了一个概率分布。

简单来说，Softmax 会把输入的数值用 e 的幂次来变换，然后归一化成一个概率分布。这样一来，所有输出的值都会在 0 到 1 之间，并且总和为 1。

所以 Softmax 的核心就是用指数函数来放大数值的差异，然后再归一化成概率分布。











