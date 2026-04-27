---
title: "Triton Tensor Manipulation"
published: 2026-04-27
description: "Triton Tensor Manipulation"
image: ""
tags: ["triton","api","Triton Tensor Manipulation"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:21:06.647.177221761Z"
---

# Triton Tensor Manipulation (张量操作)

These APIs reshape, broadcast (广播), transpose (转置), and combine block tensors (块张量) inside a kernel, used to build 2D tile (二维分块) computations like matmul (矩阵乘) and attention (注意力).

## 1. `tl.reshape`

`tl.reshape(x, shape)` changes the shape (形状) of a block tensor while keeping the same total number of elements (元素总数).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def reshape_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)                          # shape (8,)
    x2 = tl.reshape(x, (2, 4))                         # shape (2, 4)
    out_offs = tl.arange(0, 2)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + out_offs, x2)

BLOCK = 8
x = torch.arange(BLOCK, dtype=torch.float32, device='cuda')
out = torch.empty(2, 4, dtype=torch.float32, device='cuda')

reshape_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]], device='cuda:0')
```

<br>

## 2. `tl.broadcast_to`

`tl.broadcast_to(x, shape)` expands a tensor to a target shape (目标形状) following broadcasting rules (广播规则), without copying data.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(x_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK_N))         # shape (N,)
    x2 = x[None, :]                                    # shape (1, N)
    x2 = tl.broadcast_to(x2, (BLOCK_M, BLOCK_N))       # shape (M, N)
    out_offs = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(out_ptr + out_offs, x2)

BLOCK_M, BLOCK_N = 3, 4
x = torch.tensor([10., 20., 30., 40.], device='cuda')
out = torch.empty(BLOCK_M, BLOCK_N, dtype=torch.float32, device='cuda')

broadcast_kernel[(1,)](x, out, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
print(out)
# tensor([[10., 20., 30., 40.],
#         [10., 20., 30., 40.],
#         [10., 20., 30., 40.]], device='cuda:0')
```

<br>

## 3. `tl.expand_dims` (via `[None, :]`)

Adding a new dimension (新增维度) is done with `[None, :]` indexing (索引), equivalent to `unsqueeze` in PyTorch.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def expand_kernel(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.load(x_ptr + tl.arange(0, M))              # (M,)
    cn = tl.load(y_ptr + tl.arange(0, N))              # (N,)
    grid = rm[:, None] + cn[None, :]                   # (M,1) + (1,N) -> (M,N)
    out_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(out_ptr + out_offs, grid)

M, N = 3, 4
rm = torch.tensor([0., 10., 20.], device='cuda')
cn = torch.tensor([1., 2., 3., 4.], device='cuda')
out = torch.empty(M, N, dtype=torch.float32, device='cuda')

expand_kernel[(1,)](rm, cn, out, M=M, N=N)
print(out)
# tensor([[ 1.,  2.,  3.,  4.],
#         [11., 12., 13., 14.],
#         [21., 22., 23., 24.]], device='cuda:0')
```

<br>

## 4. `tl.trans`

`tl.trans(x)` transposes (转置) the last two dimensions of a 2D block tensor, equivalent to `x.T` in NumPy.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def trans_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    in_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + in_offs)                       # (M, N)
    xt = tl.trans(x)                                   # (N, M)
    out_offs = tl.arange(0, N)[:, None] * M + tl.arange(0, M)[None, :]
    tl.store(out_ptr + out_offs, xt)

M, N = 2, 4
x = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)
out = torch.empty(N, M, dtype=torch.float32, device='cuda')

trans_kernel[(1,)](x, out, M=M, N=N)
print("input:")
print(x)
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]], device='cuda:0')
print("transposed:")
print(out)
# tensor([[0., 4.],
#         [1., 5.],
#         [2., 6.],
#         [3., 7.]], device='cuda:0')
```

<br>

## 5. `tl.dot`

`tl.dot(a, b)` performs block-level matrix multiplication (块级矩阵乘法), mapping to Tensor Cores (张量核心) — the workhorse (主力) of GEMM and attention kernels.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    a_offs = tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :]
    b_offs = tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :]
    a = tl.load(a_ptr + a_offs)
    b = tl.load(b_ptr + b_offs)
    c = tl.dot(a, b)                                   # (M,K) @ (K,N) -> (M,N)
    c_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(c_ptr + c_offs, c)

M, N, K = 16, 16, 16                                   # tl.dot needs >=16
a = torch.ones(M, K, device='cuda', dtype=torch.float32)
b = torch.full((K, N), 3.0, device='cuda', dtype=torch.float32)
c = torch.empty(M, N, device='cuda', dtype=torch.float32)

dot_kernel[(1,)](a, b, c, M=M, N=N, K=K)
print(c[0, :5])                                        # each = 16 * 1 * 3 = 48
# tensor([48., 48., 48., 48., 48.], device='cuda:0')
print(c.shape)
# torch.Size([16, 16])
```

<br>

## 6. `tl.where`

`tl.where(cond, a, b)` selects elements from `a` where `cond` is true (真) and from `b` otherwise — a per-element ternary operator (三元运算符).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def where_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    out = tl.where(x > 0.0, x * 2.0, x * -1.0)         # double if positive, negate if not
    tl.store(out_ptr + offs, out)

BLOCK = 4
x = torch.tensor([-2.0, 1.0, -3.0, 4.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

where_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([2., 2., 3., 8.], device='cuda:0')
```

<br>

## 7. `tl.cat`

`tl.cat(a, b)` concatenates (拼接) two block tensors along axis 0 (since Triton 2.x), used to combine partial results (部分结果).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cat_kernel(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    a = tl.load(a_ptr + tl.arange(0, BLOCK))           # (BLOCK,)
    b = tl.load(b_ptr + tl.arange(0, BLOCK))           # (BLOCK,)
    c = tl.cat(a, b, can_reorder=True)                 # (2*BLOCK,)
    tl.store(out_ptr + tl.arange(0, 2 * BLOCK), c)

BLOCK = 4
a = torch.tensor([1., 2., 3., 4.], device='cuda')
b = torch.tensor([5., 6., 7., 8.], device='cuda')
out = torch.empty(2 * BLOCK, device='cuda')

cat_kernel[(1,)](a, b, out, BLOCK=BLOCK)
print(sorted(out.tolist()))                            # can_reorder=True allows reorder
# [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

<br>

## 8. `tl.zeros` and `tl.zeros_like`

`tl.zeros(shape, dtype)` creates a zero-filled (零填充) block tensor as an accumulator (累加器), used in multi-step reductions (多步归约).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def accum_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    acc = tl.zeros([BLOCK], dtype=tl.float32)          # accumulator
    for i in range(0, N, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        acc += x
    tl.store(out_ptr + tl.arange(0, BLOCK), acc)

N, BLOCK = 16, 4
x = torch.ones(N, dtype=torch.float32, device='cuda')   # all 1s, 16 total
out = torch.empty(BLOCK, dtype=torch.float32, device='cuda')

accum_kernel[(1,)](x, out, N=N, BLOCK=BLOCK)
print(out)                                              # each accumulator = 4 (sum of 4 chunks of 1s)
# tensor([4., 4., 4., 4.], device='cuda:0')
```

<br>

## 9. `tl.full`

`tl.full(shape, value, dtype)` creates a block tensor filled with a constant value (常量值), useful for initial bias (初始偏置) or padding (填充).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def full_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    val = tl.full([BLOCK], 3.14, dtype=tl.float32)
    tl.store(out_ptr + offs, val)

BLOCK = 4
out = torch.empty(BLOCK, device='cuda')

full_kernel[(1,)](out, BLOCK=BLOCK)
print(out)
# tensor([3.1400, 3.1400, 3.1400, 3.1400], device='cuda:0')
```

<br>

## 10. `tl.ravel` (Flatten)

`tl.ravel(x)` flattens (展平) a multi-dim block tensor into 1D, useful before storing to a contiguous output (连续输出).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def ravel_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    in_offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + in_offs)                       # (M, N)
    flat = tl.ravel(x)                                 # (M*N,)
    tl.store(out_ptr + tl.arange(0, M * N), flat)

M, N = 2, 3
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
out = torch.empty(M * N, dtype=torch.float32, device='cuda')

ravel_kernel[(1,)](x, out, M=M, N=N)
print(out)
# tensor([1., 2., 3., 4., 5., 6.], device='cuda:0')
```

<br> <br>
