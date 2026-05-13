---
title: "torch.reshape()"
published: 2026-05-12
description: "torch.reshape()"
image: ""
tags: ["pytorch","torch.reshape()"]
category: pytorch
draft: false
lang: ""
createdAt: "2026-05-12T17:39:54.166.398582510Z"
---

# PyTorch Reshape

`reshape()` changes the shape (形状) of a Tensor (张量), and it can return a view (视图) when possible or create a copy (拷贝) when necessary.

## 1. Core Idea

`reshape()` is more flexible than `view()` because it can handle non-contiguous Tensor (非连续张量) by copying data (复制数据) if needed.

<br>

## 2. Reshape on Contiguous Tensor

For a contiguous Tensor (连续张量), `reshape()` usually returns a view (视图) that shares the same storage (存储空间).

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.reshape(3, 2)

print(y)
print(y._base is x)

# Output:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
# True
```

<br>

## 3. Reshape on Non-Contiguous Tensor

For a non-contiguous Tensor (非连续张量), `reshape()` may create a new contiguous Tensor (连续张量) internally.

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)
z = y.reshape(2, 3)

print(y.is_contiguous())
print(z)
print(z.is_contiguous())

# Output:
# False
# tensor([[1, 4, 2],
#         [5, 3, 6]])
# True
```

<br>

## 4. Reshape vs View

`view()` requires compatible memory layout (兼容内存布局), while `reshape()` is safer because it automatically returns a view (视图) or a copy (拷贝).

```
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = x.transpose(0, 1)

try:
    print(y.view(2, 3))
except RuntimeError:
    print("view failed")

print(y.reshape(2, 3))

# Output:
# view failed
# tensor([[1, 4, 2],
#         [5, 3, 6]])
```

<br> <br>
