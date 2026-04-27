---
title: "Triton Atomic Operations"
published: 2026-04-27
description: "Triton Atomic Operations"
image: ""
tags: ["triton","api","Triton Atomic Operations"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:21:23.798.623796971Z"
---

# Triton Atomic Operations (原子操作)

Atomic operations (原子操作) safely update shared memory locations (共享内存位置) when multiple programs (多个程序) write to the same address, preventing race conditions (竞态条件).

## 1. `tl.atomic_add`

`tl.atomic_add(ptr, val)` atomically adds (原子加) a value to memory and returns the old value (旧值), used in scatter (散射) and histogram (直方图) kernels.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_add_kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    # multiple programs may write to same out_ptr[idx] -> need atomic
    tl.atomic_add(out_ptr + idx, 1, mask=mask)

n = 10
idx = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 0],
                   dtype=torch.int32, device='cuda')
out = torch.zeros(3, dtype=torch.int32, device='cuda')

grid = (triton.cdiv(n, 4),)
atomic_add_kernel[grid](idx, out, n, BLOCK=4)
print(out)                                             # histogram: 0 appears 4x, 1 appears 3x, 2 appears 3x
# tensor([4, 3, 3], device='cuda:0', dtype=torch.int32)
```

<br>

## 2. `tl.atomic_max`

`tl.atomic_max(ptr, val)` atomically updates memory with the maximum value (最大值) between current and new, used in reduction (归约) across blocks (跨块).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-2**30)
    block_max = tl.max(x, axis=0)
    tl.atomic_max(out_ptr, block_max)

n = 10
x = torch.tensor([3, 1, 7, 4, 9, 2, 8, 5, 6, 0],
                 dtype=torch.int32, device='cuda')
out = torch.full((1,), -2**30, dtype=torch.int32, device='cuda')

grid = (triton.cdiv(n, 4),)
atomic_max_kernel[grid](x, out, n, BLOCK=4)
print(out)
# tensor([9], device='cuda:0', dtype=torch.int32)
```

<br>

## 3. `tl.atomic_min`

`tl.atomic_min(ptr, val)` atomically updates memory with the minimum (最小值), symmetric (对称) to `atomic_max`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_min_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=2**30)
    block_min = tl.min(x, axis=0)
    tl.atomic_min(out_ptr, block_min)

n = 10
x = torch.tensor([3, 1, 7, 4, 9, 2, 8, 5, 6, 0],
                 dtype=torch.int32, device='cuda')
out = torch.full((1,), 2**30, dtype=torch.int32, device='cuda')

grid = (triton.cdiv(n, 4),)
atomic_min_kernel[grid](x, out, n, BLOCK=4)
print(out)
# tensor([0], device='cuda:0', dtype=torch.int32)
```

<br>

## 4. `tl.atomic_xchg`

`tl.atomic_xchg(ptr, val)` atomically swaps (原子交换) memory with a new value and returns the old, used to claim ownership (声明所有权) of a slot.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_xchg_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # only program 0 swaps; demonstrates atomic_xchg on a scalar
    if pid == 0:
        old = tl.atomic_xchg(out_ptr, 999)
        tl.store(out_ptr + 1, old)

BLOCK = 1
out = torch.tensor([42, 0], dtype=torch.int32, device='cuda')

atomic_xchg_kernel[(1,)](out, BLOCK=BLOCK)
print(out)                                             # out[0]=999 (new), out[1]=42 (old)
# tensor([999,  42], device='cuda:0', dtype=torch.int32)
```

<br>

## 5. `tl.atomic_cas` (Compare-and-Swap)

`tl.atomic_cas(ptr, expected, value)` atomically writes `value` only if memory equals `expected`, returning the original value — building block (基础块) for lock-free (无锁) algorithms.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_cas_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        # if out_ptr[0] == 5, write 100; else leave unchanged
        old = tl.atomic_cas(out_ptr, 5, 100)
        tl.store(out_ptr + 1, old)

BLOCK = 1
out = torch.tensor([5, 0], dtype=torch.int32, device='cuda')

atomic_cas_kernel[(1,)](out, BLOCK=BLOCK)
print(out)                                             # out[0]=100 (CAS succeeded), out[1]=5 (old)
# tensor([100,   5], device='cuda:0', dtype=torch.int32)
```

<br>

## 6. `tl.atomic_xor` (Bitwise XOR)

`tl.atomic_xor(ptr, val)` atomically XORs (异或) memory with a value, useful for bit-flag (位标志) toggling and parity (奇偶性) tracking.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_xor_kernel(vals_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(vals_ptr + offs, mask=mask, other=0)
    # XOR-reduce all values into out_ptr[0] across blocks
    block_xor = tl.reduce(v, axis=0, combine_fn=lambda a, b: a ^ b)
    tl.atomic_xor(out_ptr, block_xor)

n = 8
vals = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8],
                    dtype=torch.int32, device='cuda')   # 1^2^...^8 = 8
out = torch.zeros(1, dtype=torch.int32, device='cuda')

grid = (triton.cdiv(n, 4),)
atomic_xor_kernel[grid](vals, out, n, BLOCK=4)
print(out)
# tensor([8], device='cuda:0', dtype=torch.int32)
```

<br> <br>
