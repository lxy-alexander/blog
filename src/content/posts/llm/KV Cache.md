---
title: "KV Cache"
published: 2026-05-10
description: "KV Cache"
image: ""
tags: ["llm","KV Cache"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-11T01:34:13.743.790211896Z"
---

# KV Cache

KV Cache (键值缓存) stores previous Key and Value tensors during autoregressive decoding (自回归解码) to avoid recomputing them for every new Token (词元), especially for long context.

## 1. Core Idea

In Transformer Attention (Transformer注意力), the new Query (查询) only needs to attend to cached past Key and Value so the model computes KV (键值) only for the new Token (词元).
$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
<br>

## 3. Cost

KV Cache (键值缓存) trades GPU Memory (显存) for speed because each Layer (层) stores past Key (键) and Value (值) tensors.
$$
KV\ Cache\ Memory \propto Layers \times SequenceLength \times KVHeads \times HeadDim
$$




## 3. Example

The following Python example simulates storing KV Cache (键值缓存) for generated Tokens (词元).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k_new, v_new):
        # k_new, v_new: [B, H, T_new, D]
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        return self.k, self.v


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # q: What does the current token want to query. k: What features does the current token have that can be matched by others. v: The information content that the current token actually provides to others
        
        # The linear layers project the same input token into separate semantic spaces—Query, Key, and Value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cache=None):
        # x: [B, T, C]
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        if cache is None:
            mask = torch.tril(torch.ones(T, T, device=x.device))
            mask = mask.view(1, 1, T, T)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out

if __name__ == "__main__":
    attn = MultiHeadAttention(dim=512, num_heads=8)
    cache = KVCache()

    for _ in range(3):
        x = torch.randn(1, 1, 512)
        y = attn(x, cache)

    print(cache.k.shape)
    print(cache.v.shape)
```

<br>
