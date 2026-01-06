---
title: Transformer
published: 2025-01-06
description: "transformer"
image: "./cover.jpeg"
tags: ["Blogging", "transformer"]
category: Guides
draft: false

---



# Transformer

## Tansformer Arch

Transformer follows an encoder–decoder architecture, where each layer consists of multi-head attention, feed-forward networks, residual connections, and layer normalization.

<img src="/Users/alexanderlee/Library/Application Support/typora-user-images/image-20260105235257138.png" alt="image-20260105235257138" style="zoom:50%;" />









## Self-Attention 机制

Self-attention computes the relevance between different positions(tokens) in a sequence and aggregates information to produce context-aware representations.

Self-Attention 通过计算序列内部各位置之间的相关性，对信息进行加权汇聚，从而得到上下文相关的表示



### Q、K、V 分别代表什么？

- Query 表示当前词关注什么
- Key 表示每个词用于匹配的特征
- Value 表示每个词携带的实际语义信息

Query represents what the current token is looking for, Key represents how each token can be matched, and Value contains the actual semantic information.





## Q / K / V 从哪里来？

Q、K、V 都来自同一个输入，通过三组不同的线性变换得到。

```
Q = XWq,  K = XWk,  V = XWv
```

Q, K, and V are obtained by applying different linear projections to the same input.





### Why scale by √dₖ（square root of d sub k）?

因为点积的数值会随着维度变大而变大，如果不做缩放，softmax 很容易饱和，导致训练变得困难。所以我们用 d_k 的平方根来进行缩放，这样就能让数值保持在一个合理的范围，让梯度不会太小，训练过程也更加稳定。这正是为什么我们选择用这个平方根作为缩放的原因。



## 为什么要用多头注意力？

多头注意力允许模型在  不同的表示子空间中  并行地关注不同类型的关系。

Multi-head attention allows the model to attend to information from different representation subspaces in parallel.





## 为什么 Transformer 需要位置编码？

因为注意力机制本身不包含顺序信息，位置编码为模型提供序列的位置信息。

Since self-attention is permutation-invariant, positional encoding is required to inject order information into the model.









## Transformer 中的例子来说明一下前向和反向传播是怎么计算的

在 Transformer 的注意力机制里，假设你有一个注意力权重的矩阵 W，前向传播的时候，你会用这个 W 去对输入的查询和键进行线性变换，然后计算注意力分数，最后通过 softmax 得到一个注意力分布。

在反向传播的时候，我们要计算相对于损失函数的梯度是怎么传回到 W 的。我们会先计算损失对 softmax 输出的梯度，再根据链式法则一步步往回传播，求出每一层的梯度。最后，通过反向传播，我们可以求出损失对 W 的梯度，也就是 dW。这个 dW 就是用来更新 W 的方向和大小。

通过这个例子你可以看到，前向传播是计算输出，而反向传播就是用链式法则把梯度一层层传回去，最终得到每个参数的梯度。这样就完成了从前向到反向的一整个过程。





## Softmax它用的是什么函数呀?

Softmax 本质上使用的是指数函数。它的计算方式是先对每个输入的数值取指数，然后把所有这些指数值加起来作为分母，每个数值的指数值作为分子，这样就得到了一个概率分布。

简单来说，Softmax 会把输入的数值用 e 的幂次来变换，然后归一化成一个概率分布。这样一来，所有输出的值都会在 0 到 1 之间，并且总和为 1。

所以 Softmax 的核心就是用指数函数来放大数值的差异，然后再归一化成概率分布。









