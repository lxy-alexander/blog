---
title: "Transformer"
published: 2026-07-08
description: "Transformer"
image: ""
tags: ["llm","Transformer"]
category: llm
draft: false
lang: ""
createdAt: "2026-07-08T19:07:41.585.216898968Z"
---

# Input Embedding

**把 token 变成向量，并把向量放大到合适的尺度**。`sqrt(d_model)` 不是随便选的固定超参数，而是根据 embedding 维度自动调整尺度，让不同大小的模型都比较稳定。

如果 embedding 数值太小，位置编码会占主导，token 本身的信息会被压弱。`x = embedding + positional_encoding`

```python
class InputEmbedding(nn.Module):
	def __init__(self, vocab_size, d_model):
		super().__init__()
		self.d_model = d_model
		self.embedding = nn.Embedding(vocab_size, d_model)
		
	def forward(self, x):
		return self.embedding(x) * math.sqrt(self.d_model)
```



```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input):
        return F.embedding(input, self.weight)
```







# Positional Encoding

## dropout

`dropout` 不是删除 token，而是训练时随机屏蔽一部分向量维度，让模型不要过度依赖某些固定特征。

训练时随机让一部分特征变成 0，模型就不能每次都依赖同几个维度。它必须学会：即使少了一些特征，也能判断正确

这样模型会更稳，不容易死记硬背训练集。一次 forward 里，被置 0 的位置这次不提供信息。但其他位置还在，仍然会参与计算和反向传播。下一次 forward，随机屏蔽的位置会变，之前被屏蔽的维度可能又会参与训练。



```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # pe 用来保存所有位置的位置编码
        # shape: [max_len, d_model]
        # max_len 表示最多支持多长的序列
        pe = torch.zeros(max_len, d_model)

        # position 表示每个 token 所在的位置
        # shape: [max_len, 1]
        # 例如位置 0, 1, 2, 3, ...
        position = torch.arange(0, max_len).unsqueeze(1)

        # div_term 控制不同维度上的频率变化
        # 偶数维使用 sin，奇数维使用 cos
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # 偶数维度位置编码
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇数维度位置编码
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度
        # [max_len, d_model] -> [1, max_len, d_model]
        # 这样后面可以直接和 x 相加
        pe = pe.unsqueeze(0)

        # register_buffer 表示 pe 不是模型参数，不会被训练更新
        # 但是会跟着模型一起保存和加载
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]

        seq_len = x.size(1)

        # 取出当前序列长度对应的位置编码
        # self.pe[:, :seq_len, :] shape: [1, seq_len, d_model]
        # 会通过广播机制加到每个 batch 上
        x = x + self.pe[:, :seq_len, :]

        # dropout 用于防止过拟合
        return self.dropout(x)
```

`register_buffer` 的作用是：**把 `pe` 注册成模型的一部分，但它不是可训练参数。****当一个张量属于模型状态，但不需要训练时，就适合用 `register_buffer`**。x

**用不同频率的 sin/cos 波，把一个位置编号编码成一个固定长度的向量。前面的维度负责短距离细节，后面的维度负责长距离趋势，并且数值稳定在 [-1, 1]。**

```
position =
[[0],
 [1],
 [2]]

div_term =
[1.0, 0.01, 0.0001]
```

相乘后得到：

```
[[0 * 1.0, 0 * 0.01, 0 * 0.0001],
 [1 * 1.0, 1 * 0.01, 1 * 0.0001],
 [2 * 1.0, 2 * 0.01, 2 * 0.0001]]
```

```
[[0, 0, 0],
 [1, 0.01, 0.0001],
 [2, 0.02, 0.0002]]
```

准备每个 token 的位置编号，并把它整理成适合后面广播计算的位置矩阵。**







# Multi-Head Attention

```python
import math  # 👈 1. 记得导入 math 库
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # 统一使用 4 个空格缩进，解决 IndentationError
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
         
    def attention(self, q, k, v, mask=None):
        # 这里的 float('-inf') 也可以保留，但若后续换用 FP16/BF16 训练，建议改为 -1e9
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        return output
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # 拆头：[batch_size, seq_len, num_heads, d_k]
        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 转置：[batch_size, num_heads, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # 👈 2. 修复 Bug：把原来的 k.transpose 改为 v.transpose
        
        # 3. 接收 attention 的两个返回值
        o = self.attention(q, k, v, mask)
        
        # 合并多头：[batch_size, seq_len, d_model]
        o = o.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(o)
        
        return output
```

`mask` 的形状不一定固定，但必须能够广播到注意力分数的形状：

```python
scores.shape = [batch_size, num_heads, q_len, k_len]
```

常见情况：

1）Padding Mask

```python
mask.shape = [batch_size, 1, 1, k_len]
```

表示哪些 key 是有效 token。它会自动广播到所有注意力头和所有 query。

2）Causal Mask

```python
mask.shape = [1, 1, q_len, k_len]
```

表示每个 query 能关注哪些 key，通常用于屏蔽未来位置。

3）Padding Mask 和 Causal Mask 合并后

```python
mask.shape = [batch_size, 1, q_len, k_len]
```

4）完整 Mask

```python
mask.shape = [batch_size, num_heads, q_len, k_len]
```

每个样本、每个注意力头都可以有不同的 mask，但通常不需要这么大。

例如：

```python
q.shape = [32, 8, 10, 64]
k.shape = [32, 8, 10, 64]
scores.shape = [32, 8, 10, 10]
```

padding mask 可以是：

```python
mask.shape = [32, 1, 1, 10]
```

它会广播成：

```python
[32, 8, 10, 10]
```

所以在你的代码中，最常见的 `mask` 形状是：

```python
[batch_size, 1, 1, k_len]
```

或者：

```python
[1, 1, q_len, k_len]
```





因为 `softmax` 不会减少维度，它的输出形状和输入完全一样，所以不需要 `keepdim`。

```
attn_weights = F.softmax(scores, dim=-1)
```



# Feed Forward Network

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
```





LayerNorm + Residual
Encoder Layer
Decoder Layer
Encoder
Decoder
Transformer


