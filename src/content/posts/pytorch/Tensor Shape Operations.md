---
title: "Tensor Shape Operations"
published: 2026-05-17
description: "Tensor Shape Operations"
image: ""
tags: ["pytorch","Tensor Shape Operations"]
category: pytorch
draft: false
lang: ""
createdAt: "2026-05-17T15:44:47.437.901847133Z"
---

# PyTorch Tensor Shape Operations

Tensor shape operations are used to reorganize memory views (内存视图) without changing tensor values.

## 1. `reshape`

`reshape()` changes tensor shape safely and copies memory only when necessary.

```
import torch

x = torch.arange(6)
print(x)
# tensor([0, 1, 2, 3, 4, 5])

y = x.reshape(2, 3)
print(y)
# tensor([[0, 1, 2],
#         [3, 4, 5]])
```

Interview sentence:

`reshape()` returns a new shape and automatically handles non-contiguous tensors.

<br>

## 2. `view`

`view()` changes tensor shape without copying memory, but requires contiguous memory (连续内存).

```
import torch

x = torch.arange(6)
y = x.view(2, 3)

print(y)
# tensor([[0, 1, 2],
#         [3, 4, 5]])
```

Non-contiguous example:

```
import torch

x = torch.arange(6).reshape(2, 3)
y = x.T

print(y.is_contiguous())
# False

# y.view(6)  # RuntimeError

z = y.reshape(6)
print(z)
# tensor([0, 3, 1, 4, 2, 5])
```

Interview sentence:

`view()` is fast because it only changes metadata (元数据) without copying data.

<br>

## 3. `flatten`

`flatten()` compresses multiple dimensions into one dimension.

```
import torch

x = torch.arange(12).reshape(2, 2, 3)

print(x)
# tensor([[[ 0,  1,  2],
#          [ 3,  4,  5]],
#
#         [[ 6,  7,  8],
#          [ 9, 10, 11]]])

y = x.flatten()

print(y)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

Interview sentence:

`flatten()` is commonly used before feeding tensors into linear layers.

<br>

## 4. `squeeze` and `unsqueeze`

`squeeze()` removes dimensions with size 1, while `unsqueeze()` adds a new dimension.

```
import torch

x = torch.randn(1, 3, 1)

print(x.shape)
# torch.Size([1, 3, 1])

y = x.squeeze()

print(y.shape)
# torch.Size([3])

z = y.unsqueeze(0)

print(z.shape)
# torch.Size([1, 3])
```

Interview sentence:

`squeeze()` and `unsqueeze()` are useful for batch dimension (批次维度) management.

<br>

## 5. `transpose` and `permute`

`transpose()` swaps two dimensions, while `permute()` reorders all dimensions.

```
import torch

x = torch.arange(6).reshape(2, 3)

print(x)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

y = x.transpose(0, 1)

print(y)
# tensor([[0, 3],
#         [1, 4],
#         [2, 5]])
```

`permute()` example:

```
import torch

x = torch.randn(2, 3, 4)

y = x.permute(2, 0, 1)

print(y.shape)
# torch.Size([4, 2, 3])
```

Interview sentence:

`permute()` changes tensor dimension order by modifying strides (步长).

<br>

## 6. `contiguous`

`contiguous()` creates a contiguous memory copy of a tensor.

```
import torch

x = torch.arange(6).reshape(2, 3)

y = x.T

print(y.is_contiguous())
# False

z = y.contiguous()

print(z.is_contiguous())
# True
```

Interview sentence:

`contiguous()` is usually required before calling `view()` on transposed tensors.

<br> <br>
