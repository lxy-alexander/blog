---
title: "Triton Essential APIs"
published: 2026-04-20
description: "Triton Essential APIs"
image: ""
tags: ["triton","Triton Essential APIs"]
category: triton
draft: false
lang: ""
---

# I. Triton Tier-1: Essential APIs (必备 API)

These ~20 APIs are required for writing **any** Triton Kernel (核函数) — you cannot skip them. Master this file first.

$$ \text{Total Elements} = \text{num_programs} \times \text{BLOCK_SIZE} $$

------

## 1. `tl.program_id(axis)` — Get Block Index (获取块索引)

Returns the ID of the current Program Instance (程序实例) along the given Axis (轴) — equivalent to `blockIdx.x` in CUDA.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def show_pid_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, pid)

out = torch.empty(16, device='cuda', dtype=torch.int32)
show_pid_kernel[(4,)](out, BLOCK_SIZE=4)
print(f"Block IDs: {out.tolist()}")
# Output: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
```

------

## 2. `tl.num_programs(axis)` — Get Grid Size (获取网格大小)

Returns the total number of Program Instances (程序实例) launched along the given Axis (轴) — equivalent to `gridDim.x` in CUDA.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def grid_info_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, num_pids)

out = torch.empty(12, device='cuda', dtype=torch.int32)
grid_info_kernel[(3,)](out, BLOCK_SIZE=4)
print(f"Grid size seen by each element: {out.tolist()}")
# Output: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
```

------

## 3. `tl.arange(start, end)` — Generate Index Range (生成索引范围)

Creates a 1D Tensor of consecutive integers `[start, start+1, ..., end-1]` — used to compute per-element offsets within a Block (块).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def arange_demo_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    local_offs = tl.arange(0, BLOCK_SIZE)
    global_offs = pid * BLOCK_SIZE + local_offs
    tl.store(out_ptr + global_offs, local_offs)

out = torch.empty(8, device='cuda', dtype=torch.int32)
arange_demo_kernel[(2,)](out, BLOCK_SIZE=4)
print(f"Local offsets: {out.tolist()}")
# Output: [0, 1, 2, 3, 0, 1, 2, 3]
```

------

## 4. `tl.load(ptr, mask, other)` — Load from Memory (从内存加载)

Reads a Block (块) of data from Global Memory (全局内存); `mask` prevents Out-of-Bounds (越界) access, `other` fills masked positions with a default value.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def load_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # 'other=-1.0' fills out-of-bounds positions
    x = tl.load(x_ptr + offs, mask=mask, other=-1.0)
    tl.store(out_ptr + offs, x, mask=mask)

x = torch.tensor([10.0, 20.0, 30.0], device='cuda')
out = torch.zeros(4, device='cuda')
load_demo_kernel[(1,)](x, out, 3, BLOCK_SIZE=4)
print(f"Loaded (with boundary): {out.tolist()}")
# Output: [10.0, 20.0, 30.0, 0.0]
```

------

## 5. `tl.store(ptr, value, mask)` — Store to Memory (写回内存)

Writes a Block (块) of data to Global Memory (全局内存); `mask` prevents writing Out-of-Bounds (越界) positions.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def store_demo_kernel(out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    values = (offs * 10).to(tl.float32)
    tl.store(out_ptr + offs, values, mask=mask)

out = torch.full((5,), -1.0, device='cuda')
store_demo_kernel[(1,)](out, 5, BLOCK_SIZE=8)
print(f"Stored: {out.tolist()}")
# Output: [0.0, 10.0, 20.0, 30.0, 40.0]
```

------

## 6. `tl.zeros(shape, dtype)` — Create Zero Tensor (创建零张量)

Creates a Block-sized Tensor (张量) filled with zeros — commonly used as an Accumulator (累加器) in GEMM loops.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def zeros_demo_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc += 42.0
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, acc)

out = torch.empty(4, device='cuda')
zeros_demo_kernel[(1,)](out, BLOCK_SIZE=4)
print(f"Accumulated: {out.tolist()}")
# Output: [42.0, 42.0, 42.0, 42.0]
```

------

## 7. `tl.full(shape, value, dtype)` — Create Constant Tensor (创建常量张量)

Creates a Block-sized Tensor (张量) filled with a constant value — useful for initializing with `-inf` for max-reduction.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def full_demo_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    neg_inf = tl.full((BLOCK_SIZE,), value=-float('inf'), dtype=tl.float32)
    fives = tl.full((BLOCK_SIZE,), value=5.0, dtype=tl.float32)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, fives)

out = torch.empty(4, device='cuda')
full_demo_kernel[(1,)](out, BLOCK_SIZE=4)
print(f"Filled: {out.tolist()}")
# Output: [5.0, 5.0, 5.0, 5.0]
```

------

## 8. `tl.where(condition, x, y)` — Conditional Select (条件选择)

Selects elements from `x` where condition is True, from `y` otherwise — equivalent to `torch.where`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def where_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = tl.where(x > 0, x, 0.0)  # ReLU
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], device='cuda')
out = torch.empty_like(x)
where_demo_kernel[(1,)](x, out, 5, BLOCK_SIZE=8)
print(f"ReLU result: {out.tolist()}")
# Output: [0.0, 0.0, 0.0, 2.0, 5.0]
```

------

## 9. `tl.exp(x)` — Exponential (指数运算)

Computes Elementwise (逐元素) $e^x$ — essential for Softmax (归一化指数函数).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def exp_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = tl.exp(x)
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.tensor([0.0, 1.0, 2.0, -1.0], device='cuda')
out = torch.empty_like(x)
exp_demo_kernel[(1,)](x, out, 4, BLOCK_SIZE=4)
print(f"exp: {out.tolist()}")
print(f"Match: {torch.allclose(out, torch.exp(x), atol=1e-5)}")
```

------

## 10. `tl.log(x)` — Natural Logarithm (自然对数)

Computes Elementwise (逐元素) $\ln(x)$ — used in Cross-Entropy Loss (交叉熵损失).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def log_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    result = tl.log(x)
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.tensor([1.0, 2.718, 7.389, 0.5], device='cuda')
out = torch.empty_like(x)
log_demo_kernel[(1,)](x, out, 4, BLOCK_SIZE=4)
print(f"log: {out.tolist()}")
print(f"Match: {torch.allclose(out, torch.log(x), atol=1e-3)}")
```

------

## 11. `tl.sqrt(x)` — Square Root (平方根)

Computes Elementwise (逐元素) $\sqrt{x}$ — used in LayerNorm (层归一化) for standard deviation.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = tl.sqrt(x)
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.tensor([1.0, 4.0, 9.0, 16.0], device='cuda')
out = torch.empty_like(x)
sqrt_demo_kernel[(1,)](x, out, 4, BLOCK_SIZE=4)
print(f"sqrt: {out.tolist()}")  # [1.0, 2.0, 3.0, 4.0]
```

------

## 12. `tl.abs(x)` — Absolute Value (绝对值)

Computes Elementwise (逐元素) $|x|$ — used in L1 Loss and gradient clipping.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def abs_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    result = tl.abs(x)
    tl.store(out_ptr + offs, result, mask=mask)

x = torch.tensor([-5.0, -2.0, 0.0, 3.0], device='cuda')
out = torch.empty_like(x)
abs_demo_kernel[(1,)](x, out, 4, BLOCK_SIZE=4)
print(f"abs: {out.tolist()}")  # [5.0, 2.0, 0.0, 3.0]
```

------

## 13. `tl.sin(x)` / `tl.cos(x)` — Trigonometric Functions (三角函数)

Computes Elementwise (逐元素) sine/cosine — used in Positional Encoding (位置编码) like RoPE.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sincos_demo_kernel(x_ptr, sin_ptr, cos_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(sin_ptr + offs, tl.sin(x), mask=mask)
    tl.store(cos_ptr + offs, tl.cos(x), mask=mask)

x = torch.tensor([0.0, 1.5708, 3.1416, 4.7124], device='cuda')
sin_out = torch.empty_like(x)
cos_out = torch.empty_like(x)
sincos_demo_kernel[(1,)](x, sin_out, cos_out, 4, BLOCK_SIZE=4)
print(f"sin: {sin_out.tolist()}")  # [0, 1, 0, -1]
print(f"cos: {cos_out.tolist()}")  # [1, 0, -1, 0]
```

------

## 14. `tl.minimum(a, b)` / `tl.maximum(a, b)` — Elementwise Min/Max (逐元素最小/最大)

Computes Elementwise (逐元素) min or max of two Tensors — different from `tl.min`/`tl.max` which are Reductions (归约).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def clamp_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.minimum(x, hi)
    x = tl.maximum(x, lo)
    tl.store(out_ptr + offs, x, mask=mask)

x = torch.tensor([-10.0, -1.0, 0.5, 3.0, 100.0], device='cuda')
out = torch.empty_like(x)
clamp_kernel[(1,)](x, out, -2.0, 2.0, 5, BLOCK_SIZE=8)
print(f"Clamped [-2, 2]: {out.tolist()}")
# Output: [-2.0, -1.0, 0.5, 2.0, 2.0]
```

------

## 15. `tl.sum(x, axis)` — Sum Reduction (求和归约)

Reduces a Tensor (张量) by summing along the given Axis (轴) — returns a scalar or lower-dimensional Tensor.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sum_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + pid * n + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, total)

x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                   [10.0, 20.0, 30.0, 40.0]], device='cuda')
out = torch.empty(2, device='cuda')
sum_demo_kernel[(2,)](x, out, 4, BLOCK_SIZE=4)
print(f"Row sums: {out.tolist()}")
# Output: [10.0, 100.0]
```

------

## 16. `tl.max(x, axis)` — Max Reduction (最大值归约)

Reduces a Tensor (张量) by finding the maximum along the given Axis (轴) — essential for numerically stable Softmax.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def max_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + pid * n + offs, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    tl.store(out_ptr + pid, row_max)

x = torch.tensor([[3.0, 1.0, 4.0, 1.0],
                   [2.0, 7.0, 1.0, 8.0]], device='cuda')
out = torch.empty(2, device='cuda')
max_demo_kernel[(2,)](x, out, 4, BLOCK_SIZE=4)
print(f"Row maxes: {out.tolist()}")
# Output: [4.0, 8.0]
```

------

## 17. `tl.min(x, axis)` — Min Reduction (最小值归约)

Reduces a Tensor (张量) by finding the minimum along the given Axis (轴).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def min_demo_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + pid * n + offs, mask=mask, other=float('inf'))
    row_min = tl.min(x, axis=0)
    tl.store(out_ptr + pid, row_min)

x = torch.tensor([[3.0, 1.0, 4.0, 1.0],
                   [2.0, 7.0, 1.0, 8.0]], device='cuda')
out = torch.empty(2, device='cuda')
min_demo_kernel[(2,)](x, out, 4, BLOCK_SIZE=4)
print(f"Row mins: {out.tolist()}")
# Output: [1.0, 1.0]
```

------

## 18. `tl.dot(a, b)` — Block Matrix Multiply (块矩阵乘法)

Performs a Block-level Matrix Multiplication (块级矩阵乘法) — triggers Tensor Core (张量核心) usage automatically; requires `BLOCK >= 16`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def dot_demo_kernel(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
    offs_i = tl.arange(0, BLOCK)
    offs_j = tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offs_i[:, None] * BLOCK + offs_j[None, :])
    b = tl.load(b_ptr + offs_i[:, None] * BLOCK + offs_j[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_i[:, None] * BLOCK + offs_j[None, :], c)

B = 16
a = torch.randn(B, B, device='cuda', dtype=torch.float32)
b = torch.randn(B, B, device='cuda', dtype=torch.float32)
c = torch.empty(B, B, device='cuda', dtype=torch.float32)
dot_demo_kernel[(1,)](a, b, c, BLOCK=B)
print(f"Match PyTorch matmul: {torch.allclose(c, a @ b, atol=1e-2)}")
```

------

## 19. `triton.cdiv(a, b)` — Ceiling Division (上取整除法)

Computes $\lceil a / b \rceil$ — used to calculate how many Blocks (块) are needed to cover all elements.

```python
import triton

n_blocks = triton.cdiv(1000, 256)
print(f"Blocks needed: {n_blocks}")  # 4 (256*4=1024 >= 1000)
```

------

## 20. `tl.constexpr` — Compile-time Constant (编译期常量)

Marks a Kernel argument as a Compile-time Constant (编译期常量) — the compiler unrolls loops and optimizes layout; `BLOCK_SIZE` must always be `tl.constexpr`.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def constexpr_demo(out_ptr, BLOCK_SIZE: tl.constexpr, DEPTH: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(DEPTH):   # Compiler unrolls this loop
        val += 1.0
    tl.store(out_ptr + offs, val)

out = torch.empty(8, device='cuda')
constexpr_demo[(2,)](out, BLOCK_SIZE=4, DEPTH=10)
print(f"Values: {out.tolist()}")  # [10.0, 10.0, ...] (loop ran 10 times)
```

------

# II. Tier-1 Combination Patterns (Tier-1 组合模式)

## 1. Elementwise Skeleton (逐元素骨架)

**APIs:** `program_id` + `arange` + `load` + `store` + `where` + math ops

Every Elementwise Kernel shares this exact 5-step structure.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                    # Step 1
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)        # Step 2
    mask = offs < n                                            # Step 3
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)           # Step 4
    # GELU math
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    e2x = tl.exp(2.0 * inner)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    result = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, result, mask=mask)                # Step 5

n = 512
x = torch.randn(n, device='cuda')
out = torch.empty_like(x)
gelu_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK_SIZE=256)
ref = torch.nn.functional.gelu(x, approximate='tanh')
print(f"GELU match: {torch.allclose(out, ref, atol=1e-4)}")
```

## 2. Row Reduction Skeleton (行归约骨架)

**APIs:** `program_id` + `arange` + `load` + `max` + `sum` + `exp` + `store`

Softmax, LogSumExp, LayerNorm all follow: load row → reduce → transform → store.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(out_ptr + row * n_cols + offs, result, mask=mask)

rows, cols = 8, 64
x = torch.randn(rows, cols, device='cuda')
out = torch.empty_like(x)
softmax_kernel[(rows,)](x, out, cols, BLOCK_SIZE=64)
print(f"Softmax match: {torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-5)}")
```

------

# III. Tier-1 Quick Reference (速查表)

| API                          | Purpose (用途)               | CUDA Equivalent      |
| ---------------------------- | ---------------------------- | -------------------- |
| `tl.program_id(axis)`        | Block Index (块索引)         | `blockIdx.x`         |
| `tl.num_programs(axis)`      | Grid Size (网格大小)         | `gridDim.x`          |
| `tl.arange(start, end)`      | Index Range (索引范围)       | `threadIdx.x`        |
| `tl.load(ptr, mask, other)`  | Load Memory (加载内存)       | `__ldg()`            |
| `tl.store(ptr, val, mask)`   | Store Memory (写回内存)      | Direct assign        |
| `tl.zeros(shape, dtype)`     | Zero Tensor (零张量)         | `memset`             |
| `tl.full(shape, val, dtype)` | Constant Tensor (常量张量)   | Manual init          |
| `tl.where(cond, a, b)`       | Conditional (条件选择)       | Ternary `?:`         |
| `tl.exp(x)`                  | Exponential (指数)           | `expf()`             |
| `tl.log(x)`                  | Logarithm (对数)             | `logf()`             |
| `tl.sqrt(x)`                 | Square Root (平方根)         | `sqrtf()`            |
| `tl.abs(x)`                  | Absolute Value (绝对值)      | `fabsf()`            |
| `tl.sin(x)` / `tl.cos(x)`    | Trigonometric (三角函数)     | `sinf`/`cosf`        |
| `tl.minimum(a, b)`           | Elementwise Min (逐元素最小) | `fminf()`            |
| `tl.maximum(a, b)`           | Elementwise Max (逐元素最大) | `fmaxf()`            |
| `tl.sum(x, axis)`            | Sum Reduction (求和归约)     | `__shfl_down`        |
| `tl.max(x, axis)`            | Max Reduction (最大值归约)   | Manual reduction     |
| `tl.min(x, axis)`            | Min Reduction (最小值归约)   | Manual reduction     |
| `tl.dot(a, b)`               | Matrix Multiply (矩阵乘)     | `wmma` / Tensor Core |
| `triton.cdiv(a, b)`          | Ceiling Division (上取整)    | `(a+b-1)/b`          |
| `tl.constexpr`               | Compile Constant (编译常量)  | Template param       |
