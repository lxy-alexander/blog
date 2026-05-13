---
title: "MoE"
published: 2026-05-12
description: "MoE"
image: ""
tags: ["llm","MoE"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-12T21:43:53.162.316560050Z"
---



# MoE and Transformer

MoE (混合专家模型) is not a replacement for Transformer (Transformer架构), but a sparse module (稀疏模块) usually used to replace the FFN (前馈神经网络) inside a Transformer block (Transformer模块).

![Transformer vs. Mixture of Experts in LLMs - by Avi Chawla](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/https%253A%252F%252Fsubstack-post-media.s3.amazonaws.com%252Fpublic%252Fimages%252F0681d04f-0cd6-45a7-b1f1-a37f9269d01d_1116x1126)



## 1. Transformer

Transformer (Transformer架构) uses Self-Attention (自注意力机制) to exchange information between tokens (词元), and uses FFN (前馈神经网络) to process each token representation (词元表示).

```
Transformer Block = Self-Attention + FFN
```

<br>

## 2. MoE

MoE (混合专家模型, Mixture of Experts) replaces one dense FFN (稠密前馈网络) with multiple Experts (专家网络), and use a Router (路由器) to select a few Experts for each token (词元).

```
MoE Transformer Block = Self-Attention + MoE-FFN
```

<br>

## 3. Main Difference

Transformer (Transformer架构) usually sends every token (词元) through the same FFN (前馈神经网络), while MoE (混合专家模型) sends each token to selected Experts (专家网络).

```
Dense Transformer:
token → same FFN

MoE Transformer:
token → Router → selected Experts
```

<br>

## 4. How Routing Works

Router (路由器) computes routing logits (路由分数) for all Experts (专家网络), applies Softmax (归一化指数函数), and selects the Top-k Experts (前k个专家) for the current token (词元).
$$
logits=xW_r
$$

```
token → Router → expert scores → Top-k Experts
```

For example, given one token representation (词元表示):
$$
x=[1,2]
$$
and a router weight matrix (路由权重矩阵):
$$
W_r=
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 1
\end{bmatrix}
$$
the routing logits (路由分数) are:
$$
logits=[1,2]
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & 1
\end{bmatrix}
=
[1,2,4]
$$
After Softmax (归一化指数函数):
$$
p=\text{softmax}([1,2,4])\approx[0.04,0.11,0.84]
$$
With Top-2 Routing (前2路由), the Router (路由器) selects Expert 3 (专家3) and Expert 2 (专家2):

```
token → Expert 3 + Expert 2
```

The final output (最终输出) is a weighted combination (加权组合) of selected Expert outputs (专家输出):
$$
y=0.84\cdot Expert_3(x)+0.11\cdot Expert_2(x)
$$
