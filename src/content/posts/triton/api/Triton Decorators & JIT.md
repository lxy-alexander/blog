---
title: "Triton Decorators & JIT"
published: 2026-04-27
description: "Triton Decorators & JIT"
image: ""
tags: ["triton","api","Triton Decorators & JIT"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:17:06.960.859527703Z"
---

# Triton Decorators & JIT (装饰器与即时编译)

This category contains the **core decorators (核心装饰器)** that compile (编译) Python functions into GPU kernels (GPU 内核), along with launch helpers (启动辅助函数).

## 1. `@triton.jit`

The `@triton.jit` decorator (装饰器) marks a Python function as a GPU kernel (GPU 内核) and triggers JIT compilation (即时编译) when called with a launch grid (启动网格).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 8
x = torch.arange(n, dtype=torch.float32, device='cuda')
y = torch.ones(n, dtype=torch.float32, device='cuda')
out = torch.empty_like(x)

add_kernel[(1,)](x, y, out, n, BLOCK=8)
print(out)
# tensor([1., 2., 3., 4., 5., 6., 7., 8.], device='cuda:0')
```

<br>

## 2. `tl.constexpr`

`tl.constexpr` marks a kernel argument as a compile-time constant (编译期常量), enabling the compiler to specialize (特化) and unroll loops (展开循环).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def scale_kernel(x_ptr, out_ptr, n,
                 BLOCK: tl.constexpr,         # compile-time
                 SCALE: tl.constexpr):        # compile-time
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * SCALE, mask=mask)

n = 8
x = torch.arange(n, dtype=torch.float32, device='cuda')
out = torch.empty_like(x)

scale_kernel[(1,)](x, out, n, BLOCK=8, SCALE=10.0)
print(out)
# tensor([ 0., 10., 20., 30., 40., 50., 60., 70.], device='cuda:0')
```

<br>

## 3. `triton.cdiv`

`triton.cdiv(a, b)` is ceiling division (向上取整), used to compute the number of blocks (块数) needed to cover `n` elements with block size `BLOCK`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

n = 1000           # not a multiple of BLOCK
BLOCK = 256
x = torch.arange(n, dtype=torch.float32, device='cuda')
out = torch.empty_like(x)

grid = (triton.cdiv(n, BLOCK),)         # ceil(1000/256) = 4 blocks
print("grid:", grid)                    # grid: (4,)

copy_kernel[grid](x, out, n, BLOCK=BLOCK)
print(out[:5])                          # tensor([0., 1., 2., 3., 4.], device='cuda:0')
print(out[-5:])                         # tensor([995., 996., 997., 998., 999.], device='cuda:0')
```

<br> <br>
