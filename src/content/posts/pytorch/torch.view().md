---
title: "torch.view()"
published: 2026-05-12
description: "torch.view()"
image: ""
tags: ["pytorch","torch.view()"]
category: pytorch
draft: false
lang: ""
createdAt: "2026-05-12T17:37:42.515.532787367Z"
---

# PyTorch View

`view()` usually requires the Tensor (张量) to be contiguous (连续内存), because it only changes the shape (形状) without copying data (复制数据).

## 1. Core Idea

`view()` reshapes a Tensor (张量) by reinterpreting the same memory (内存), so the original Tensor usually must be contiguous (连续内存).

<br>

## 2. Non-Contiguous Tensor

A non-contiguous Tensor (非连续张量) may appear after operations like `transpose()` because the logical order (逻辑顺序) no longer matches the physical memory order (物理内存顺序).

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)

print(y)
print(y.is_contiguous())

# Output:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
# False
```

<br>

## 3. View Error

Calling `view()` on a non-contiguous Tensor (非连续张量) may fail because `view()` cannot safely reinterpret the memory layout (内存布局).

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)

print(y.is_contiguous())

try:
    z = y.view(2, 3)
    print(z)
except RuntimeError as e:
    print(type(e).__name__)
    print("view failed because tensor is non-contiguous")

# Output:
# False
# RuntimeError
# view failed because tensor is non-contiguous
```

<br>

## 4. Solutions

Use `contiguous().view()` when you want to force contiguous memory (连续内存), or use `reshape()` when you want PyTorch (PyTorch框架) to handle copying automatically if needed.

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)

z1 = y.contiguous().view(2, 3)
z2 = y.reshape(2, 3)

print(z1)
print(z2)

# Output:
# tensor([[1, 4, 2],
#         [5, 3, 6]])
# tensor([[1, 4, 2],
#         [5, 3, 6]])
```

<br> <br>
