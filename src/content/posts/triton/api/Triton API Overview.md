---
title: "Triton API Overview"
published: 2026-04-27
description: "Triton API Overview"
image: ""
tags: ["triton","api","Triton API Overview"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:01:06.316.324677176Z"
---

# Triton Language API Overview

Triton is an open-source GPU programming language (GPU 编程语言) by OpenAI for writing high-performance kernels (高性能内核); its core API surface is small (~50 functions) compared to CUDA, focused on block-level (块级) tensor operations.

## 1. Total API Scope

Triton's API is intentionally minimal (极简) — roughly **50–70 core functions** in `triton.language` (`tl`) cover almost all kernel writing needs (内核编写需求).

| Category                       | Approximate Count | Importance    |
| ------------------------------ | ----------------- | ------------- |
| Decorators & JIT (装饰器)      | ~3                | ⭐⭐⭐ Must know |
| Program ID & Indexing (索引)   | ~3                | ⭐⭐⭐ Must know |
| Memory Load/Store (加载/存储)  | ~4                | ⭐⭐⭐ Must know |
| Math Operations (数学运算)     | ~15               | ⭐⭐⭐ Must know |
| Reduction (归约)               | ~6                | ⭐⭐ Important  |
| Tensor Manipulation (张量操作) | ~10               | ⭐⭐ Important  |
| Atomic Operations (原子操作)   | ~6                | ⭐ Optional    |
| Debug & Misc (调试)            | ~5                | ⭐ Optional    |

<br>

## 2. Decorators and Launch (Must Know)

The `@triton.jit` decorator (装饰器) compiles a Python function into a GPU kernel (内核), and `triton.cdiv` computes the grid size (网格大小).

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


def add(x, y):
    out = torch.empty_like(x) # apply for memory as large as x, Automatically inherit the attributes of x (device, dtype, shape, layout)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)  # grid must be a iterable object(tuple or list) (x axis)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


def main():
    n = 10000
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = add(x, y)
    torch_out = x + y
    print(out)
    print(torch.allclose(out, torch_out)) # 


if __name__ == "__main__":
    main()
```

<br>

## 3. Indexing APIs (Must Know)

These three functions identify the current program (当前程序) in the launch grid (启动网格), used in every Triton kernel.

| API                     | Purpose                                     |
| ----------------------- | ------------------------------------------- |
| `tl.program_id(axis)`   | Current block ID along axis (当前块ID)      |
| `tl.num_programs(axis)` | Total programs/blocks along axis (总程序数) |
| `tl.arange(start, end)` | Generate index vector (生成索引向量)        |

```python
import triton.language as tl
# Inside @triton.jit kernel:
# pid = tl.program_id(0)            # block index
# total = tl.num_programs(0)        # grid size
# offs = tl.arange(0, 128)          # [0,1,...,127]
```

<br>

## 4. Memory Access (Must Know)

`tl.load` and `tl.store` access global memory (全局内存) with optional mask (掩码) to handle out-of-bounds (越界) safely.

| API                              | Purpose                             |
| -------------------------------- | ----------------------------------- |
| `tl.load(ptr, mask, other)`      | Load with mask (带掩码加载)         |
| `tl.store(ptr, val, mask)`       | Store with mask (带掩码存储)        |
| `tl.make_block_ptr(...)`         | Block pointer (块指针) for 2D tiles |
| `tl.advance(block_ptr, offsets)` | Move block pointer (移动块指针)     |

<br>

## 5. Math Operations (Must Know)

Triton supports element-wise math (逐元素运算) on block tensors (块张量), similar to NumPy syntax (语法).

| Category        | APIs                                                |
| --------------- | --------------------------------------------------- |
| Basic           | `+ - * / %`, `tl.abs`, `tl.minimum`, `tl.maximum`   |
| Exp/Log         | `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`            |
| Trig (三角)     | `tl.sin`, `tl.cos`, `tl.sqrt`, `tl.rsqrt`           |
| Special         | `tl.sigmoid`, `tl.softmax` (via reduce), `tl.where` |
| Cast (类型转换) | `tl.cast(x, tl.float16)`, `x.to(tl.bfloat16)`       |

<br>

## 6. Reduction APIs (Important)

Reductions (归约) collapse a dimension (维度) of a block tensor, used in softmax, normalization (归一化), etc.

| API                      | Purpose                    |
| ------------------------ | -------------------------- |
| `tl.sum(x, axis)`        | Sum reduction (求和)       |
| `tl.max(x, axis)`        | Max reduction (最大值)     |
| `tl.min(x, axis)`        | Min reduction (最小值)     |
| `tl.argmax(x, axis)`     | Index of max (最大值索引)  |
| `tl.argmin(x, axis)`     | Index of min (最小值索引)  |
| `tl.reduce(x, axis, fn)` | Custom reduce (自定义归约) |

<br>

## 7. Tensor Manipulation (Important)

Block tensor (块张量) reshape and broadcast (广播), used to build 2D tile computations (二维分块计算).

| API                         | Purpose                                    |
| --------------------------- | ------------------------------------------ |
| `tl.reshape(x, shape)`      | Reshape (重塑形状)                         |
| `tl.broadcast_to(x, shape)` | Broadcast (广播)                           |
| `tl.expand_dims(x, axis)`   | Add dimension (增加维度)                   |
| `tl.trans(x)`               | Transpose (转置)                           |
| `tl.dot(a, b)`              | Matrix multiply (矩阵乘法) — core for GEMM |
| `tl.where(cond, a, b)`      | Conditional select (条件选择)              |
| `tl.cat(a, b, axis)`        | Concatenate (拼接)                         |

<br>

## 8. Atomic Operations (Optional)

Atomic ops (原子操作) safely update shared memory locations (共享内存位置) from multiple programs, used in scatter (散射) or histogram (直方图) kernels.

| API              | Purpose                       |
| ---------------- | ----------------------------- |
| `tl.atomic_add`  | Atomic add (原子加)           |
| `tl.atomic_max`  | Atomic max                    |
| `tl.atomic_min`  | Atomic min                    |
| `tl.atomic_xchg` | Atomic exchange (原子交换)    |
| `tl.atomic_cas`  | Compare and swap (比较并交换) |

<br>

## 9. Autotuning and Heuristics (Important)

Triton provides decorators (装饰器) for automatic kernel tuning (自动调优), avoiding manual benchmarking (手动基准测试).

| API                  | Purpose                                           |
| -------------------- | ------------------------------------------------- |
| `@triton.autotune`   | Auto-pick best config (自动选择最优配置)          |
| `triton.Config`      | Define a tuning candidate (调优候选)              |
| `@triton.heuristics` | Dynamic config based on inputs (基于输入动态配置) |
| `tl.constexpr`       | Compile-time constant (编译期常量)                |

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=4),
        triton.Config({'BLOCK': 256}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def kernel(x_ptr, n, BLOCK: tl.constexpr):
    pass
```

<br>

## 10. Debug APIs (Optional)

Used during kernel development (内核开发) to print values (打印数值) or check assertions (断言).

| API                          | Purpose                            |
| ---------------------------- | ---------------------------------- |
| `tl.device_print(prefix, x)` | Print on GPU (GPU 上打印)          |
| `tl.static_print(...)`       | Print at compile time (编译期打印) |
| `tl.static_assert(...)`      | Compile-time assert (编译期断言)   |

<br>

## 11. Learning Roadmap

A practical learning path (学习路径) — most kernels (90%) only need the **first 20 APIs** in the "Must Know" tier (必学层).

| Stage                            | APIs to Master                                              | Time      |
| -------------------------------- | ----------------------------------------------------------- | --------- |
| **Stage 1: Basics (基础)**       | `@jit`, `program_id`, `arange`, `load`, `store`, basic math | 1–2 days  |
| **Stage 2: Reductions (归约)**   | `sum`, `max`, `where`, `exp`, `log` (softmax, layernorm)    | 2–3 days  |
| **Stage 3: Matmul (矩阵乘)**     | `dot`, `make_block_ptr`, `advance`, 2D tiling               | 3–5 days  |
| **Stage 4: Optimization (优化)** | `autotune`, `Config`, `num_warps`, `num_stages`             | 3–5 days  |
| **Stage 5: Advanced (进阶)**     | atomics, custom `reduce`, FlashAttention-style kernels      | 1–2 weeks |

<br>

## 12. Summary

You only need to **master ~20 core APIs** to write 90% of useful Triton kernels (内核); the full API count is around **50–70 functions**, far smaller than CUDA's massive runtime API (运行时 API).

| Tier (层级)          | Count | Coverage                          |
| -------------------- | ----- | --------------------------------- |
| ⭐⭐⭐ Must Know (必学) | ~20   | Covers 90% kernels                |
| ⭐⭐ Important (重要)  | ~20   | GEMM, autotune, advanced ops      |
| ⭐ Optional (可选)    | ~15   | Atomics, debug, special use cases |

<br> <br>
