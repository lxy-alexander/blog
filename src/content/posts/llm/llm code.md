---
title: "llm code"
published: 2026-06-01
description: "llm code"
image: ""
tags: ["llm","llm code"]
category: llm
draft: false
lang: ""
createdAt: "2026-06-01T22:02:17.785.323606866Z"
---

# KVCache

1）**Why is the causal mask added to attention scores?**
==Because attention scores become attention weights after softmax. Masking the scores controls which tokens receive attention.==
==因为注意力分数经过 softmax 后会变成注意力权重。对分数加 mask，可以控制哪些 token 能被关注。==

2）**Why is setting a masked score to `0` not enough?**
Because `0` can still get non-zero weight after softmax. A masked score must be very negative, like `-∞`.
因为 `0` 经过 softmax 后仍然可能得到非零权重。被 mask 的分数必须是非常大的负数，比如 `-∞`。

3）**What happens to `-∞` after softmax?**
It becomes `0` attention weight, so that position is ignored.
它会变成 `0` 注意力权重，所以这个位置会被忽略。

4）**How does attention weight affect `V`?**
The output is a weighted sum of `V`. If the weight is `0`, that `V` contributes nothing.
输出是 `V` 的加权和。如果权重是 `0`，对应的 `V` 就不会产生贡献。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, k_new, v_new):
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        return self.k, self.v


class SelfAttentionWithKVCache(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim) # q = x @ self.q_proj.weight.T + self.q_proj.bias
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cache: KVCache = None, use_causal_mask=True):
        batch, seq_len, dim = x.shape

        q = self.q_proj(x) # [batch, seq_len, dim] @ [dim, dim] → [batch, seq_len, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # Adjust the dimension arrangement of the tensor, This can facilitate parallel computing for different attention heads
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k, v = cache.update(k, v)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        scores = scores / (self.head_dim ** 0.5)

        if use_causal_mask:
            q_len = q.size(-2)
            k_len = k.size(-2)

            mask = torch.tril(torch.ones(q_len, k_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)

        return self.out_proj(out)


dim = 512
num_heads = 8

attention = SelfAttentionWithKVCache(dim, num_heads)

# prefill 阶段：一次输入多个 token，需要 causal mask
x_prompt = torch.randn(1, 4, dim)
out = attention(x_prompt, cache=None, use_causal_mask=True)

print(out.shape)
# torch.Size([1, 4, 512])

# decode 阶段：每次只输入一个新 token，使用 KVCache
cache = KVCache()

x1 = torch.randn(1, 1, dim)
out1 = attention(x1, cache=cache)

x2 = torch.randn(1, 1, dim)
out2 = attention(x2, cache=cache)

x3 = torch.randn(1, 1, dim)
out3 = attention(x3, cache=cache)

print(cache.k.shape)
print(cache.v.shape)
# torch.Size([1, 8, 3, 64])
# torch.Size([1, 8, 3, 64])
```











