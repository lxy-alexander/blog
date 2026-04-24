---
title: "Triton Performance APIs"
published: 2026-04-20
description: "Triton Performance APIs"
image: ""
tags: ["triton","Triton Performance APIs"]
category: triton
draft: false
lang: ""
---

# I. Triton Tier-2: Performance APIs (性能关键 API)

These ~12 APIs take your Kernel from "runs correctly" to "runs fast" — Autotuning (自动调优), Block Pointers (块指针), Atomic Operations (原子操作), and Type Casting (类型转换).

------

## 1. `triton.autotune` — Automatic Configuration Search (自动配置搜索)

Decorator that searches over multiple Configurations (配置) of `BLOCK_SIZE`, `num_warps`, `num_stages` and picks the fastest — Triton's killer feature.

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],  # Re-tune when this argument changes
)
@triton.jit
def vector_add_tuned(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 100000
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
vector_add_tuned[grid](x, y, out, n)
print(f"Correct: {torch.allclose(out, x + y)}")
```

------

## 2. `num_warps` — Warps per Block (每块线程束数)

Controls how many Warps (线程束, 32 threads each) execute per Block — more warps = more parallelism but more Register Pressure (寄存器压力).

![image-20260420121418173](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260420121418173)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def warp_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)

n = 1024
x = torch.randn(n, device='cuda')
out = torch.empty_like(x)
# num_warps=4 means 128 threads per block
warp_demo[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256, num_warps=4)
print(f"Correct: {torch.allclose(out, x * 2)}")
```

**Tuning guide:**

| `BLOCK_SIZE` | Recommended `num_warps` |
| ------------ | ----------------------- |
| 64-128       | 2-4                     |
| 256-512      | 4-8                     |
| 1024+        | 8                       |

------

## 3. `num_stages` — Pipeline Stages (流水线级数)

Controls Software Pipelining (软件流水线) depth — higher values prefetch more data tiles to hide Memory Latency (内存延迟), at the cost of more Shared Memory (共享内存).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def pipeline_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)

n = 4096
x = torch.randn(n, device='cuda')
out = torch.empty_like(x)
# num_stages=3: prefetch 3 tiles simultaneously
pipeline_demo[(triton.cdiv(n, 512),)](x, out, n, BLOCK_SIZE=512, num_stages=3)
print(f"Correct: {torch.allclose(out, x + 1.0)}")
```

**Rule of thumb:** `num_stages=2` for Ampere, `num_stages=3-4` for Hopper.

------

## 4. `tl.atomic_add(ptr, val, mask)` — Atomic Addition (原子加法)

Thread-safe accumulation to the same address from multiple Program Instances (程序实例) — used for Histogram (直方图), Gradient Accumulation (梯度累加).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_add_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    partial_sum = tl.sum(x, axis=0)
    # All blocks atomically add to position 0
    tl.atomic_add(out_ptr, partial_sum)

x = torch.ones(1024, device='cuda')
out = torch.zeros(1, device='cuda')
atomic_add_demo[(triton.cdiv(1024, 256),)](x, out, 1024, BLOCK_SIZE=256)
print(f"Global sum: {out.item()}")  # 1024.0
```

------

## 5. `tl.atomic_max(ptr, val)` / `tl.atomic_min(ptr, val)` — Atomic Max/Min (原子最大/最小)

Thread-safe max/min update across blocks — used for finding Global Maximum (全局最大值).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def atomic_max_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    local_max = tl.max(x, axis=0)
    tl.atomic_max(out_ptr, local_max)

x = torch.randn(1024, device='cuda')
out = torch.full((1,), -float('inf'), device='cuda')
atomic_max_demo[(triton.cdiv(1024, 256),)](x, out, 1024, BLOCK_SIZE=256)
print(f"Global max: {out.item():.4f}, PyTorch: {x.max().item():.4f}")
```

------

## 6. `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` — Block Pointer (块指针)

Creates a structured pointer for 2D Tile (瓦片) access — the compiler generates more efficient memory patterns than manual offset arithmetic.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def block_ptr_demo(x_ptr, out_ptr, M, N, stride_m, stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    x_block = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_m, stride_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),  # Row-major
    )
    tile = tl.load(x_block, boundary_check=(0, 1))
    tile = tile * 2.0
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(M, N), strides=(stride_m, stride_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(out_block, tile, boundary_check=(0, 1))

M, N = 64, 64
x = torch.randn(M, N, device='cuda')
out = torch.empty_like(x)
grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
block_ptr_demo[grid](x, out, M, N, x.stride(0), x.stride(1), BLOCK_M=32, BLOCK_N=32)
print(f"Match: {torch.allclose(out, x * 2)}")
```

------

## 7. `tl.advance(block_ptr, offsets)` — Advance Block Pointer (移动块指针)

Moves a Block Pointer (块指针) by the given offsets — used to iterate over K-dimension tiles in GEMM loops.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def advance_demo(x_ptr, out_ptr, M, K, stride_m, stride_k,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(M, K), strides=(stride_m, stride_k),
        offsets=(pid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        tile = tl.load(x_block, boundary_check=(0, 1))
        acc += tl.sum(tile, axis=1)
        x_block = tl.advance(x_block, (0, BLOCK_K))  # Move right
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs_m < M
    tl.store(out_ptr + offs_m, acc, mask=mask)

M, K = 32, 128
x = torch.randn(M, K, device='cuda')
out = torch.empty(M, device='cuda')
advance_demo[(triton.cdiv(M, 16),)](x, out, M, K, x.stride(0), x.stride(1),
                                     BLOCK_M=16, BLOCK_K=32)
print(f"Match: {torch.allclose(out, x.sum(dim=1), atol=1e-3)}")
```

------

## 8. `tl.trans(x)` — Transpose 2D Tile (转置二维瓦片)

Transposes a 2D Block (块) in registers — useful for matrix transpose and preparing data layout for `tl.dot`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def trans_demo(x_ptr, out_ptr, M, N, stride_xm, stride_xn,
               stride_om, stride_on,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    tile_t = tl.trans(tile)
    o_ptrs = out_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    tl.store(o_ptrs, tile_t, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))

M, N = 32, 64
x = torch.randn(M, N, device='cuda')
out = torch.empty(N, M, device='cuda')
grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
trans_demo[grid](x, out, M, N, x.stride(0), x.stride(1),
                 out.stride(0), out.stride(1), BLOCK_M=16, BLOCK_N=16)
print(f"Transpose match: {torch.allclose(out, x.T)}")
```

------

## 9. `tl.broadcast_to(x, shape)` — Broadcast Tensor (广播张量)

Expands a lower-dimensional Tensor (张量) to a target shape — like NumPy Broadcasting (广播).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_demo(bias_ptr, out_ptr, M, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    # Broadcast from [1, N] to [M, N]
    bias_2d = bias[None, :]
    bias_broadcast = tl.broadcast_to(bias_2d, (BLOCK_M, BLOCK_N))
    offs_m = tl.arange(0, BLOCK_M)
    ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrs, bias_broadcast, mask=mask)

M, N = 4, 8
bias = torch.arange(8, dtype=torch.float32, device='cuda')
out = torch.empty(M, N, device='cuda')
broadcast_demo[(1,)](bias, out, M, N, BLOCK_M=4, BLOCK_N=8)
print(f"Each row equals bias: {torch.allclose(out, bias.unsqueeze(0).expand(M, N))}")
```

------

## 10. `x.to(dtype)` — Type Cast (类型转换)

Converts a Tensor (张量) to a different Data Type (数据类型) — critical for Mixed Precision (混合精度): load FP16, accumulate FP32, store FP16.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cast_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x_f16 = tl.load(x_ptr + offs, mask=mask)        # Load as float16
    x_f32 = x_f16.to(tl.float32)                     # Cast to float32
    result_f32 = x_f32 * x_f32 + x_f32               # Compute in float32
    result_f16 = result_f32.to(tl.float16)            # Cast back to float16
    tl.store(out_ptr + offs, result_f16, mask=mask)

x = torch.randn(8, device='cuda', dtype=torch.float16)
out = torch.empty_like(x)
cast_demo[(1,)](x, out, 8, BLOCK_SIZE=8)
ref = (x.float() * x.float() + x.float()).half()
print(f"Match: {torch.allclose(out, ref, atol=1e-2)}")
```

------

# II. Tier-2 Combination Patterns (Tier-2 组合模式)

## 1. GEMM with Block Pointers (使用块指针的矩阵乘法)

**APIs:** `program_id(0,1)` + `make_block_ptr` + `advance` + `zeros` + `dot` + `.to()` + `store`

The canonical high-performance GEMM — 2D grid, tile accumulation, Block Pointers for efficiency.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_block = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block, boundary_check=(0, 1))
        b_tile = tl.load(b_block, boundary_check=(0, 1))
        acc += tl.dot(a_tile, b_tile)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))
    c_block = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block, acc.to(tl.float32), boundary_check=(0, 1))

M, N, K = 64, 64, 64
a = torch.randn(M, K, device='cuda')
b = torch.randn(K, N, device='cuda')
c = torch.empty(M, N, device='cuda')
grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
gemm_kernel[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
print(f"GEMM match: {torch.allclose(c, a @ b, atol=1e-2)}")
```

## 2. Histogram with Atomics (使用原子操作的直方图)

**APIs:** `program_id` + `arange` + `load` + `atomic_add` + `minimum` + `maximum`

When multiple blocks write to the same output — Atomic Operations (原子操作) serialize access.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def histogram_kernel(x_ptr, hist_ptr, n, num_bins, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bins = tl.maximum(tl.minimum(x, num_bins - 1), 0).to(tl.int32)
    tl.atomic_add(hist_ptr + bins, 1, mask=mask)

n = 10000
num_bins = 8
x = torch.randint(0, num_bins, (n,), device='cuda', dtype=torch.int32)
hist = torch.zeros(num_bins, device='cuda', dtype=torch.int32)
histogram_kernel[(triton.cdiv(n, 256),)](x, hist, n, num_bins, BLOCK_SIZE=256)
print(f"Histogram: {hist.tolist()}")
print(f"Match: {torch.equal(hist, x.bincount(minlength=num_bins))}")
```

## 3. Autotuned Softmax (自动调优 Softmax)

**APIs:** `autotune` + all Tier-1 reduction APIs

Combine autotuning with Softmax for best performance across different sequence lengths.

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_cols'],
)
@triton.jit
def tuned_softmax(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    num = tl.exp(x - x_max)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * n_cols + offs, num / den, mask=mask)

rows, cols = 32, 512
x = torch.randn(rows, cols, device='cuda')
out = torch.empty_like(x)
tuned_softmax[(rows,)](x, out, cols)
print(f"Match: {torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-5)}")
```

------

# III. Tier-2 Quick Reference (速查表)

| API                 | Purpose (用途)                             | When to Use                              |
| ------------------- | ------------------------------------------ | ---------------------------------------- |
| `triton.autotune`   | Auto-search best config (自动搜索最优配置) | **Always** — never hardcode `BLOCK_SIZE` |
| `num_warps`         | Warps per block (每块线程束数)             | Set in `triton.Config`                   |
| `num_stages`        | Pipeline depth (流水线深度)                | Set in `triton.Config`, 2-4 typical      |
| `tl.atomic_add`     | Atomic accumulation (原子累加)             | Multiple blocks → same address           |
| `tl.atomic_max/min` | Atomic max/min (原子极值)                  | Global max/min across blocks             |
| `tl.make_block_ptr` | Structured 2D pointer (结构化 2D 指针)     | GEMM, 2D tile loops                      |
| `tl.advance`        | Move block pointer (移动块指针)            | K-loop in GEMM                           |
| `tl.trans`          | Transpose tile (转置瓦片)                  | Matrix transpose, data layout            |
| `tl.broadcast_to`   | Broadcast (广播)                           | Bias addition, row/col ops               |
| `x.to(dtype)`       | Type cast (类型转换)                       | Mixed precision: FP16→FP32→FP16          |

------

# IV. Tuning Parameter Guide (调优参数指南)

| Parameter    | Range   | Effect                                              |
| ------------ | ------- | --------------------------------------------------- |
| `BLOCK_SIZE` | 64-2048 | Larger = more work per block, better latency hiding |
| `num_warps`  | 2-8     | More = more threads, more register pressure         |
| `num_stages` | 1-5     | More = more prefetch overlap, more SMEM usage       |

![image-20260420121358606](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260420121358606)

**Best practice:** Always use `triton.autotune` with 3-6 configs covering different `BLOCK_SIZE` × `num_warps` × `num_stages` combinations.
