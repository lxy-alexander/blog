---
title: "Triton Reduction"
published: 2026-04-27
description: "Triton Reduction"
image: ""
tags: ["triton","api","Triton Reduction"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:19:44.557.097074608Z"
---

# Triton Reduction (归约)

Reductions (归约) collapse a dimension (维度) of a block tensor (块张量) to a single value or smaller tensor, used in softmax (softmax), normalization (归一化), and statistics (统计量) kernels.

## 1. `tl.sum`

`tl.sum(x, axis)` sums elements along an axis (轴), producing a scalar (标量) or reduced tensor (降维张量).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    s = tl.sum(x, axis=0)                              # scalar sum
    tl.store(out_ptr, s)

BLOCK = 8
x = torch.arange(1, 9, dtype=torch.float32, device='cuda')   # 1+2+...+8 = 36
out = torch.empty(1, device='cuda')

sum_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([36.], device='cuda:0')
```

<br>

## 2. `tl.max`

`tl.max(x, axis)` returns the maximum value (最大值) along an axis, often used in numerically stable softmax (数值稳定的 softmax).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    m = tl.max(x, axis=0)
    tl.store(out_ptr, m)

BLOCK = 8
x = torch.tensor([3., 1., 4., 1., 5., 9., 2., 6.], device='cuda')
out = torch.empty(1, device='cuda')

max_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([9.], device='cuda:0')
```

<br>

## 3. `tl.min`

`tl.min(x, axis)` returns the minimum value (最小值) along an axis, symmetric (对称) to `tl.max`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    m = tl.min(x, axis=0)
    tl.store(out_ptr, m)

BLOCK = 8
x = torch.tensor([3., 1., 4., 1., 5., 9., 2., 6.], device='cuda')
out = torch.empty(1, device='cuda')

min_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([1.], device='cuda:0')
```

<br>

## 4. `tl.argmax`

`tl.argmax(x, axis)` returns the index (索引) of the maximum value along an axis, used for classification (分类) outputs.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    idx = tl.argmax(x, axis=0)
    tl.store(out_ptr, idx)

BLOCK = 8
x = torch.tensor([3., 1., 4., 1., 5., 9., 2., 6.], device='cuda')   # max=9 at idx 5
out = torch.empty(1, dtype=torch.int32, device='cuda')

argmax_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([5], device='cuda:0', dtype=torch.int32)
```

<br>

## 5. `tl.argmin`

`tl.argmin(x, axis)` returns the index (索引) of the minimum value along an axis, symmetric to `tl.argmax`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    idx = tl.argmin(x, axis=0)
    tl.store(out_ptr, idx)

BLOCK = 8
x = torch.tensor([3., 1., 4., 1., 5., 9., 2., 6.], device='cuda')   # min=1 at idx 1
out = torch.empty(1, dtype=torch.int32, device='cuda')

argmin_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([1], device='cuda:0', dtype=torch.int32)
```

<br>

## 6. `tl.reduce` (Custom Reduction)

`tl.reduce(x, axis, fn)` allows a user-defined reduction (自定义归约) using a combine function (合并函数), enabling custom operations like product (乘积) or logsumexp (对数求和指数).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _mul(a, b):
    return a * b

@triton.jit
def product_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    p = tl.reduce(x, axis=0, combine_fn=_mul)          # custom: product
    tl.store(out_ptr, p)

BLOCK = 4
x = torch.tensor([1., 2., 3., 4.], device='cuda')      # 1*2*3*4 = 24
out = torch.empty(1, device='cuda')

product_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([24.], device='cuda:0')
```

<br> <br>
