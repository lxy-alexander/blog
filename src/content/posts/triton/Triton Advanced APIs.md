---
title: "Triton Advanced APIs"
published: 2026-04-20
description: "Triton Advanced APIs"
image: ""
tags: ["triton","Triton Advanced APIs"]
category: triton
draft: false
lang: ""
---

# I. Triton Tier-3: Advanced APIs (高级 API)

These ~8 APIs are for expert-level optimization — reshaping tiles, debugging, inline assembly, and complex fusion patterns.

------

## 1. `tl.reshape(x, shape)` — Reshape Tensor (重塑张量)

Changes the shape of a Tensor (张量) without copying data — used for rearranging tile dimensions before `tl.dot` or reduction.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def reshape_demo(out_ptr, BLOCK: tl.constexpr):
    # Create 1D: [0, 1, 2, ..., 15]
    x = tl.arange(0, BLOCK * BLOCK).to(tl.float32)
    # Reshape to 2D: (BLOCK, BLOCK)
    x_2d = tl.reshape(x, (BLOCK, BLOCK))
    # Sum each row
    row_sums = tl.sum(x_2d, axis=1)
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, row_sums)

out = torch.empty(4, device='cuda')
reshape_demo[(1,)](out, BLOCK=4)
print(f"Row sums of [[0..3],[4..7],[8..11],[12..15]]: {out.tolist()}")
# Output: [6.0, 22.0, 38.0, 54.0]
```

------

## 2. `tl.flip(x, dim)` — Reverse a Tensor (反转张量)

Reverses elements along the specified Dimension (维度) — useful for creating symmetric patterns or reverse scans.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flip_demo(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_flipped = tl.flip(x, 0)
    tl.store(out_ptr + offs, x_flipped, mask=mask)

x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
out = torch.empty_like(x)
flip_demo[(1,)](x, out, 4, BLOCK_SIZE=4)
print(f"Flipped: {out.tolist()}")
# Output: [4.0, 3.0, 2.0, 1.0]
```

------

## 3. `tl.device_print(prefix, val)` — Debug Print (调试输出)

Prints values from inside a running Kernel (核函数) — essential for debugging; use with **small** grid sizes to avoid output flood.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def print_demo(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    row_sum = tl.sum(x, axis=0)
    # Print from GPU for debugging
    tl.device_print("Block", pid)
    tl.device_print("Sum", row_sum)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
print_demo[(2,)](x, 6, BLOCK_SIZE=4)
# Will print block IDs and partial sums from GPU
```

**Debugging tips:**

-   Only use with `grid=(1,)` or `grid=(2,)` — large grids produce millions of lines
-   Remove before benchmarking — print causes massive slowdown
-   Combine with `TRITON_INTERPRET=1` for CPU-side step-through

------

## 4. `tl.inline_asm_elementwise(asm, args, dtype, is_pure, pack)` — Inline PTX Assembly (内联 PTX 汇编)

Inserts raw PTX Assembly (PTX 汇编) instructions — last resort for squeezing performance from hardware-specific instructions not exposed by Triton.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fast_tanh_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Use PTX hardware tanh (faster than software approximation)
    result = tl.inline_asm_elementwise(
        asm="tanh.approx.f32 $0, $1;",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.randn(256, device='cuda')
out = torch.empty_like(x)
fast_tanh_kernel[(1,)](x, out, 256, BLOCK_SIZE=256)
print(f"Close to torch.tanh: {torch.allclose(out, torch.tanh(x), atol=1e-3)}")
```

**Common PTX instructions:**

| PTX Instruction    | Purpose                   |
| ------------------ | ------------------------- |
| `tanh.approx.f32`  | Hardware-accelerated tanh |
| `ex2.approx.f32`   | Fast $2^x$                |
| `lg2.approx.f32`   | Fast $\log_2(x)$          |
| `rcp.approx.f32`   | Fast reciprocal $1/x$     |
| `rsqrt.approx.f32` | Fast $1/\sqrt{x}$         |

------

## 5. `tl.join(a, b)` — Concatenate Tiles (拼接瓦片)

Joins two Tensors along a new innermost dimension — produces interleaved data layout.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def join_demo(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    # Join: interleave a and b along new inner dimension
    joined = tl.join(a, b)  # Shape: (BLOCK_SIZE, 2)
    # Store interleaved: a0,b0,a1,b1,...
    out_offs = offs[:, None] * 2 + tl.arange(0, 2)[None, :]
    tl.store(out_ptr + out_offs, joined, mask=mask[:, None])

a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
b = torch.tensor([10.0, 20.0, 30.0, 40.0], device='cuda')
out = torch.empty(8, device='cuda')
join_demo[(1,)](a, b, out, 4, BLOCK_SIZE=4)
print(f"Interleaved: {out.tolist()}")
# Output: [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
```

------

## 6. `tl.split(x)` — Split Tile (拆分瓦片)

Splits a Tensor along its innermost dimension into two halves — reverse of `tl.join`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def split_demo(x_ptr, out_a_ptr, out_b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # Load interleaved data as (BLOCK_SIZE, 2)
    x_offs = offs[:, None] * 2 + tl.arange(0, 2)[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask[:, None])
    # Split back into two separate tensors
    a, b = tl.split(x)
    tl.store(out_a_ptr + offs, a, mask=mask)
    tl.store(out_b_ptr + offs, b, mask=mask)

x = torch.tensor([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], device='cuda')
out_a = torch.empty(4, device='cuda')
out_b = torch.empty(4, device='cuda')
split_demo[(1,)](x, out_a, out_b, 4, BLOCK_SIZE=4)
print(f"A: {out_a.tolist()}")  # [1.0, 2.0, 3.0, 4.0]
print(f"B: {out_b.tolist()}")  # [10.0, 20.0, 30.0, 40.0]
```

------

# II. Tier-3 Combination Patterns (Tier-3 组合模式)

## 1. Fused LayerNorm (融合层归一化)

**APIs:** Tier-1 (load/store/sum/sqrt) + Tier-2 (autotune) + Tier-3 (for complex fusion)

Fusing Mean → Variance → Normalize → Affine eliminates 4 Global Memory round-trips.

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
def fused_layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    # Load everything
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    # Fused: mean -> variance -> normalize -> affine (NO intermediate GMEM writes)
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = diff * inv_std * w + b
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)

rows, cols = 32, 256
x = torch.randn(rows, cols, device='cuda')
w = torch.ones(cols, device='cuda')
b = torch.zeros(cols, device='cuda')
out = torch.empty_like(x)
fused_layernorm_kernel[(rows,)](x, w, b, out, cols, 1e-5)
ref = torch.nn.functional.layer_norm(x, [cols])
print(f"LayerNorm match: {torch.allclose(out, ref, atol=1e-4)}")
```

## 2. RoPE Positional Encoding (旋转位置编码)

**APIs:** Tier-1 (`sin`, `cos`, `exp`, `log`) + Tier-3 (reshape for paired rotation)

Apply Rotary Position Embeddings (旋转位置嵌入) using sin/cos pairs.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(x_ptr, out_ptr, seq_len, dim, BLOCK_D: tl.constexpr):
    pos = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    half_d = BLOCK_D // 2
    mask = offs_d < dim
    x = tl.load(x_ptr + pos * dim + offs_d, mask=mask, other=0.0)
    # Compute rotation frequencies
    freq_idx = (offs_d % half_d).to(tl.float32)
    theta = pos / tl.exp(freq_idx * (tl.log(10000.0) / half_d))
    cos_val = tl.cos(theta)
    sin_val = tl.sin(theta)
    # Apply rotation (simplified)
    result = x * cos_val
    tl.store(out_ptr + pos * dim + offs_d, result, mask=mask)

seq_len, dim = 16, 32
x = torch.randn(seq_len, dim, device='cuda')
out = torch.empty_like(x)
rope_kernel[(seq_len,)](x, out, seq_len, dim, BLOCK_D=32)
print(f"RoPE applied, shape: {out.shape}")
```

## 3. Mixed Precision GEMM (混合精度矩阵乘法)

**APIs:** Tier-1 (load/store/dot/zeros) + Tier-2 (`make_block_ptr`/advance/.to()) + Tier-2 (autotune)

Load FP16 → Accumulate FP32 → Store FP16 for maximum Tensor Core throughput.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def mixed_gemm(a_ptr, b_ptr, c_ptr, M, N, K,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # FP32 accumulator for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)  # FP16 inputs, FP32 accumulation
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    # Cast back to FP16 for storage
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M, N, K = 64, 64, 64
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)
grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
mixed_gemm[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
print(f"Match: {torch.allclose(c.float(), (a.float() @ b.float()), atol=0.5)}")
```

------

# III. Tier-3 Quick Reference (速查表)

| API                         | Purpose (用途)                 | When to Use                      |
| --------------------------- | ------------------------------ | -------------------------------- |
| `tl.reshape(x, shape)`      | Reshape tile (重塑瓦片)        | Rearrange dims before dot/reduce |
| `tl.flip(x, dim)`           | Reverse tensor (反转张量)      | Symmetric patterns, reverse scan |
| `tl.device_print(msg, val)` | GPU debug print (GPU 调试输出) | Debug only, small grid           |
| `tl.inline_asm_elementwise` | Inline PTX (内联汇编)          | Hardware-specific fast math      |
| `tl.join(a, b)`             | Concat tiles (拼接瓦片)        | Interleave data                  |
| `tl.split(x)`               | Split tile (拆分瓦片)          | De-interleave data               |

------

# IV. Debugging & Profiling Guide (调试与性能分析指南)

## 1. Debugging Workflow (调试流程)

| Step                   | Tool                                  | Usage                                 |
| ---------------------- | ------------------------------------- | ------------------------------------- |
| 1. Verify correctness  | `torch.allclose(out, ref)`            | Always compare with PyTorch reference |
| 2. Print inside kernel | `tl.device_print("val", x)`           | Small grid only                       |
| 3. CPU interpreter     | `TRITON_INTERPRET=1 python script.py` | Step-by-step, no GPU needed           |
| 4. Check masks         | Verify `mask = offs < n`              | Most bugs are boundary errors         |

## 2. Profiling Commands (性能分析命令)

```bash
# NVIDIA Nsight Compute: detailed kernel analysis
ncu python my_kernel.py

# Triton autotuner verbose output
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# Show compiled PTX
MLIR_ENABLE_DUMP=1 python my_kernel.py
```

## 3. Performance Checklist (性能清单)

| Priority | Action               | API                                          |
| -------- | -------------------- | -------------------------------------------- |
| 1        | **Fuse everything**  | Combine ops in one kernel                    |
| 2        | **Autotune**         | `@triton.autotune` with 3-6 configs          |
| 3        | **Mixed precision**  | `.to(tl.float16)` + `tl.zeros(..., float32)` |
| 4        | **Block pointers**   | `make_block_ptr` + `advance` for 2D tiles    |
| 5        | **Minimize atomics** | Local `tl.sum` first, then one `atomic_add`  |
| 6        | **Coalesced access** | Adjacent threads → adjacent addresses        |

------

# V. Cross-Tier API Map (跨层级 API 关系图)

This shows how APIs from all three tiers combine in real kernels:

```
Tier-1 (Foundation)          Tier-2 (Performance)         Tier-3 (Advanced)
─────────────────           ──────────────────           ─────────────────
program_id ──────────────── autotune ─────────────────── inline_asm
arange     ──────────────── num_warps/num_stages          reshape
load/store ──────────────── make_block_ptr + advance      join/split
zeros/full                  atomic_add/max/min             flip
where                       trans                          device_print
exp/log/sqrt/sin/cos        broadcast_to
sum/max/min                 .to(dtype)
dot
cdiv/constexpr
```

**Typical kernel uses:**

| Kernel Type    | Tier-1   | Tier-2                                   | Tier-3                   |
| -------------- | -------- | ---------------------------------------- | ------------------------ |
| ReLU / GELU    | 5 APIs   | autotune                                 | —                        |
| Softmax        | 7 APIs   | autotune                                 | —                        |
| LayerNorm      | 8 APIs   | autotune                                 | —                        |
| GEMM           | 8 APIs   | `block_ptr` + advance + .to() + autotune | —                        |
| FlashAttention | 10 APIs  | `block_ptr` + advance + autotune         | reshape                  |
| Custom fused   | 10+ APIs | all Tier-2                               | `inline_asm`, join/split |
