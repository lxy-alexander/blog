---
title: "Triton Memory Load&Store"
published: 2026-04-27
description: "Triton Memory Load&Store"
image: ""
tags: ["triton","api","Triton Memory Load&Store"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:18:58.623.058148098Z"
---

# Triton Memory Load/Store (内存加载与存储)

These APIs transfer data between global memory (全局内存) and registers (寄存器), with masking (掩码) to safely handle out-of-bounds (越界) indices.

## 1. `tl.load(ptr, mask, other)`

`tl.load` reads from a pointer (指针), with a mask (掩码) controlling which elements are loaded and `other` providing fallback values (默认值) for masked-out positions.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def load_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-1.0)   # -1 for OOB
    tl.store(out_ptr + offs, x)

n = 5
BLOCK = 8                                              # larger than n
x = torch.arange(n, dtype=torch.float32, device='cuda')
out = torch.full((BLOCK,), 99.0, dtype=torch.float32, device='cuda')

load_kernel[(1,)](x, out, n, BLOCK=BLOCK)
print(out)
# tensor([ 0.,  1.,  2.,  3.,  4., -1., -1., -1.], device='cuda:0')
```

<br>

## 2. `tl.store(ptr, val, mask)`

`tl.store` writes a block tensor (块张量) to memory at the given pointer, with a mask (掩码) preventing writes to invalid (无效) positions.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def store_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                                    # only write valid positions
    val = offs.to(tl.float32) * 2.0
    tl.store(out_ptr + offs, val, mask=mask)

n = 5
BLOCK = 8
out = torch.full((BLOCK,), 99.0, dtype=torch.float32, device='cuda')

store_kernel[(1,)](out, n, BLOCK=BLOCK)
print(out)
# tensor([ 0.,  2.,  4.,  6.,  8., 99., 99., 99.], device='cuda:0')
```

<br>

## 3. `tl.make_block_ptr`

`tl.make_block_ptr` creates a block pointer (块指针) describing a 2D tile (二维分块) with shape, strides (步长), offsets (偏移), and block shape — preferred for matrix kernels (矩阵内核).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def block_ptr_kernel(x_ptr, out_ptr, M, N,
                     stride_m, stride_n,
                     BM: tl.constexpr, BN: tl.constexpr):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_m, stride_n),
        offsets=(0, 0),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    x = tl.load(x_block_ptr, boundary_check=(0, 1))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(M, N), strides=(stride_m, stride_n),
        offsets=(0, 0), block_shape=(BM, BN), order=(1, 0),
    )
    tl.store(out_block_ptr, x * 2.0, boundary_check=(0, 1))

M, N = 4, 4
x = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)
out = torch.empty_like(x)

block_ptr_kernel[(1,)](x, out, M, N, x.stride(0), x.stride(1), BM=4, BN=4)
print(out)
# tensor([[ 0.,  2.,  4.,  6.],
#         [ 8., 10., 12., 14.],
#         [16., 18., 20., 22.],
#         [24., 26., 28., 30.]], device='cuda:0')
```

<br>

## 4. `tl.advance`

`tl.advance(block_ptr, offsets)` moves a block pointer (块指针) by a given offset (偏移), commonly used inside loops (循环) over tiles in matmul (矩阵乘) kernels.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def advance_kernel(x_ptr, out_ptr, M, N,
                   stride_m, stride_n,
                   BM: tl.constexpr, BN: tl.constexpr):
    x_bp = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
        offsets=(0, 0), block_shape=(BM, BN), order=(1, 0),
    )
    x_bp = tl.advance(x_bp, (0, BN))                   # move right by BN
    x = tl.load(x_bp, boundary_check=(0, 1))
    out_bp = tl.make_block_ptr(
        base=out_ptr, shape=(BM, BN), strides=(BN, 1),
        offsets=(0, 0), block_shape=(BM, BN), order=(1, 0),
    )
    tl.store(out_bp, x)

M, N = 2, 4
x = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)
print("input:")
print(x)
# tensor([[0., 1., 2., 3.],
#         [4., 5., 6., 7.]], device='cuda:0')

out = torch.empty(2, 2, dtype=torch.float32, device='cuda')
advance_kernel[(1,)](x, out, M, N, x.stride(0), x.stride(1), BM=2, BN=2)
print("after advance(0,2) — right tile:")
print(out)
# tensor([[2., 3.],
#         [6., 7.]], device='cuda:0')
```

<br> <br>
