---
title: "Principle of W.T"
published: 2026-05-11
description: "Principle of W.T"
image: ""
tags: ["pytorch","Principle of W.T"]
category: pytorch
draft: false
lang: ""
createdAt: "2026-05-11T16:20:34.239.453238827Z"
---

# Principle of `W.T`

`W.T` does not move data; it creates a View (视图) by permute Shape (形状) and Stride (步长).

-   Shape: how many dimensions the tensor appears to have, and the length of each dimension. 这个 tensor 看起来有几维、每维多长
-   Stride: how many elements to jump in the underlying storage when an index in each dimension increases by 1.每个维度索引加 1 时，底层内存跳几个元素
-   Offset: the position in the underlying storage where this tensor starts. 这个 tensor 从底层 storage 的第几个元素开始

```python
storage_index = storage_offset + index0 * stride0 + index1 * stride1 + ...

Example:

import torch

x = torch.arange(6).reshape(2, 3)

x =
tensor([[0, 1, 2],
        [3, 4, 5]])

Underlying storage:

storage = [0, 1, 2, 3, 4, 5]

For x:

shape  = (2, 3)
stride = (3, 1)
offset = 0

Meaning:

shape = 2 rows, 3 columns

stride = (3, 1)

row index +1    -> jump 3 elements in storage
column index +1 -> jump 1 element in storage

Now calculate x[1, 2]:

storage_index = offset + row_index * row_stride + column_index * column_stride
              = 0 + 1 * 3 + 2 * 1
              = 5

So:

x[1, 2] = storage[5] = 5

Another example:

y = x[:, 1:]

y =
tensor([[1, 2],
        [4, 5]])

For y:

shape  = (2, 2)
stride = (3, 1)
offset = 1

Now calculate y[1, 1]:

storage_index = offset + row_index * row_stride + column_index * column_stride
              = 1 + 1 * 3 + 1 * 1
              = 5

So:

y[1, 1] = storage[5] = 5
```





## 1. Core Principle

A Tensor (张量) reads data from Storage (存储) using Shape (形状), Stride (步长), and Offset (偏移量).
$$
address = storage\_offset + i \times stride_0 + j \times stride_1
$$
For a 2D Tensor (二维张量), `W.T` swaps the two dimensions:
$$
W.shape=(m,n),\quad W.stride=(s_0,s_1)
$$

## 2. Simple Example

`W.T` works even when `W` is Non-Contiguous (非连续内存), because it only changes how indices map to the same Storage (存储).

```py
import torch

x = torch.arange(12).reshape(3, 4)


# take all rows. / take one every two columns
W = x[:, ::2]   # Non-contiguous view  
WT = W.T        # Transposed view

print("W shape:", W.shape)
print("W stride:", W.stride())
print("WT shape:", WT.shape)
print("WT stride:", WT.stride())
print(WT)

# Output:
# W shape: torch.Size([3, 2])
# W stride: (4, 2)
# WT shape: torch.Size([2, 3])
# WT stride: (2, 4)
# tensor([[ 0,  4,  8],
#         [ 2,  6, 10]])
```

<br> <br>
