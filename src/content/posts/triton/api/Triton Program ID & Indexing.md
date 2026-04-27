---
title: "Triton Program ID & Indexing"
published: 2026-04-27
description: "Triton Program ID & Indexing"
image: ""
tags: ["triton","api","Triton Program ID & Indexing"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:18:21.338.728475026Z"
---

# Triton Program ID & Indexing (程序 ID 与索引)

These APIs identify the current block (当前块) within the launch grid (启动网格) and generate index vectors (索引向量) for parallel computation (并行计算).

## 1. `tl.program_id(axis)`

`tl.program_id(axis)` returns the current block's index (块索引) along a given axis (轴) of the launch grid (启动网格), used to compute data offsets (数据偏移).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def write_pid_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # block index along axis 0
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, pid.to(tl.float32) + tl.zeros([BLOCK], tl.float32))

n = 16
BLOCK = 4
out = torch.empty(n, dtype=torch.float32, device='cuda')

grid = (triton.cdiv(n, BLOCK),)                  # 4 blocks
write_pid_kernel[grid](out, BLOCK=BLOCK)
print(out)
# tensor([0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.], device='cuda:0')
```

<br>

## 2. `tl.num_programs(axis)`

`tl.num_programs(axis)` returns the total number of blocks (总块数) launched along a given axis (轴), useful for boundary checks (边界检查) or grid-stride loops (网格步长循环).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def write_total_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    total = tl.num_programs(0)                   # total blocks in axis 0
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, total.to(tl.float32) + tl.zeros([BLOCK], tl.float32))

n = 12
BLOCK = 4
out = torch.empty(n, dtype=torch.float32, device='cuda')

grid = (triton.cdiv(n, BLOCK),)                  # 3 blocks
write_total_kernel[grid](out, BLOCK=BLOCK)
print(out)
# tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], device='cuda:0')
```

<br>

## 3. `tl.arange(start, end)`

`tl.arange(start, end)` generates a compile-time index vector (编译期索引向量); both `start` and `end` must be powers-of-two constants (2 的幂常量).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def arange_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK)                    # [0, 1, 2, ..., BLOCK-1]
    offs = pid * BLOCK + idx
    tl.store(out_ptr + offs, idx.to(tl.float32))

n = 8
BLOCK = 8
out = torch.empty(n, dtype=torch.float32, device='cuda')

arange_kernel[(1,)](out, BLOCK=BLOCK)
print(out)
# tensor([0., 1., 2., 3., 4., 5., 6., 7.], device='cuda:0')
```

<br> <br>
