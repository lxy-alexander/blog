---
title: "Triton Math Operations"
published: 2026-04-27
description: "Triton Math Operations"
image: ""
tags: ["triton","api","Triton Math Operations"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:19:25.354.194865047Z"
---

# Triton Math Operations (数学运算)

Triton supports element-wise math (逐元素运算) on block tensors (块张量) with a NumPy-like (类 NumPy) syntax (语法), covering arithmetic (算术), exponential (指数), trigonometric (三角), and casting (类型转换) operations.

## 1. Basic Arithmetic (`+ - * / %`)

Standard arithmetic operators (算术运算符) work element-wise (逐元素) on block tensors via Python operator overloading (运算符重载).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def arith_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x * y + 1.0)              # fused multiply-add

BLOCK = 4
x = torch.tensor([1., 2., 3., 4.], device='cuda')
y = torch.tensor([10., 20., 30., 40.], device='cuda')
out = torch.empty(BLOCK, device='cuda')

arith_kernel[(1,)](x, y, out, BLOCK=BLOCK)
print(out)
# tensor([ 11.,  41.,  91., 161.], device='cuda:0')
```

<br>

## 2. `tl.abs`

`tl.abs(x)` returns the absolute value (绝对值) element-wise for both integer (整型) and float (浮点) tensors.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def abs_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.abs(x))

BLOCK = 4
x = torch.tensor([-1.5, 2.0, -3.5, 4.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

abs_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([1.5000, 2.0000, 3.5000, 4.0000], device='cuda:0')
```

<br>

## 3. `tl.minimum` and `tl.maximum`

`tl.minimum` and `tl.maximum` compute the element-wise (逐元素) min and max between two block tensors, used for clipping (截断) values.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def clip_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    x = tl.maximum(x, 0.0)                             # clip lower
    x = tl.minimum(x, 5.0)                             # clip upper
    tl.store(out_ptr + offs, x)

BLOCK = 4
x = torch.tensor([-2.0, 3.0, 7.0, 1.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

clip_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([0., 3., 5., 1.], device='cuda:0')
```

<br>

## 4. `tl.exp` and `tl.exp2`

`tl.exp` computes $$e^x$$ and `tl.exp2` computes $$2^x$$ element-wise, used in softmax (softmax 函数) and other normalization (归一化) operations.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def exp_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.exp(x))

BLOCK = 4
x = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

exp_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([ 1.0000,  2.7183,  7.3891, 20.0855], device='cuda:0')
```

<br>

## 5. `tl.log` and `tl.log2`

`tl.log` computes the natural logarithm (自然对数) and `tl.log2` computes the base-2 logarithm (以 2 为底的对数) element-wise.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.log(x))

BLOCK = 4
x = torch.tensor([1.0, 2.7183, 7.3891, 20.0855], device='cuda')
out = torch.empty(BLOCK, device='cuda')

log_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00], device='cuda:0')
```

<br>

## 6. `tl.sqrt` and `tl.rsqrt`

`tl.sqrt` is the square root (平方根) and `tl.rsqrt` is the reciprocal square root (平方根倒数), commonly used in RMSNorm (均方根归一化) and LayerNorm.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_kernel(x_ptr, sqrt_out, rsqrt_out, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(sqrt_out + offs, tl.sqrt(x))
    tl.store(rsqrt_out + offs, tl.rsqrt(x))

BLOCK = 4
x = torch.tensor([1.0, 4.0, 9.0, 16.0], device='cuda')
s = torch.empty(BLOCK, device='cuda')
r = torch.empty(BLOCK, device='cuda')

sqrt_kernel[(1,)](x, s, r, BLOCK=BLOCK)
print("sqrt :", s)
# sqrt : tensor([1., 2., 3., 4.], device='cuda:0')
print("rsqrt:", r)
# rsqrt: tensor([1.0000, 0.5000, 0.3333, 0.2500], device='cuda:0')
```

<br>

## 7. `tl.sin` and `tl.cos`

`tl.sin` and `tl.cos` compute element-wise sine (正弦) and cosine (余弦), used in positional encoding (位置编码) like RoPE (旋转位置编码).

```python
import torch, math
import triton
import triton.language as tl

@triton.jit
def trig_kernel(x_ptr, sin_out, cos_out, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(sin_out + offs, tl.sin(x))
    tl.store(cos_out + offs, tl.cos(x))

BLOCK = 4
x = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], device='cuda')
s = torch.empty(BLOCK, device='cuda')
c = torch.empty(BLOCK, device='cuda')

trig_kernel[(1,)](x, s, c, BLOCK=BLOCK)
print("sin:", s)
# sin: tensor([ 0.0000e+00,  1.0000e+00, -8.7423e-08, -1.0000e+00], device='cuda:0')
print("cos:", c)
# cos: tensor([ 1.0000e+00, -4.3711e-08, -1.0000e+00,  1.1925e-08], device='cuda:0')
```

<br>

## 8. `tl.sigmoid`

`tl.sigmoid(x)` computes $$\sigma(x) = \frac{1}{1+e^{-x}}$$ element-wise, the classic activation function (激活函数) for binary classification (二分类).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.sigmoid(x))

BLOCK = 4
x = torch.tensor([-2.0, 0.0, 1.0, 2.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

sigmoid_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([0.1192, 0.5000, 0.7311, 0.8808], device='cuda:0')
```

<br>

## 9. `tl.where`

`tl.where(cond, a, b)` is the conditional select (条件选择) — equivalent to `a if cond else b` per element, used for masked operations (掩码运算).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    out = tl.where(x > 0.0, x, 0.0)                    # ReLU
    tl.store(out_ptr + offs, out)

BLOCK = 4
x = torch.tensor([-1.0, 2.0, -3.0, 4.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

relu_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([0., 2., 0., 4.], device='cuda:0')
```

<br>

## 10. `tl.cast` and `.to()`

`tl.cast(x, dtype)` and `x.to(dtype)` convert (转换) a block tensor's data type (数据类型), used for mixed precision (混合精度) computation.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cast_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)                          # float32
    x_fp16 = x.to(tl.float16)                          # cast to fp16
    x_back = x_fp16.to(tl.float32)                     # cast back
    tl.store(out_ptr + offs, x_back)

BLOCK = 4
x = torch.tensor([1.5, 2.25, 3.125, 4.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

cast_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([1.5000, 2.2500, 3.1250, 4.0000], device='cuda:0')
```

<br>

## 11. `tl.fdiv` (Float Division)

`tl.fdiv(x, y)` performs floating-point division (浮点除法) with a fast-math (快速数学) path, useful when full IEEE compliance (IEEE 合规性) is not required.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fdiv_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, tl.fdiv(x, y))

BLOCK = 4
x = torch.tensor([10., 20., 30., 40.], device='cuda')
y = torch.tensor([2., 4., 5., 8.], device='cuda')
out = torch.empty(BLOCK, device='cuda')

fdiv_kernel[(1,)](x, y, out, BLOCK=BLOCK)
print(out)
# tensor([5., 5., 6., 5.], device='cuda:0')
```

<br>

## 12. `tl.cdiv` (Ceiling Division)

`tl.cdiv(a, b)` is the in-kernel ceiling division (向上取整除法) on scalars, used to compute loop bounds (循环上界) inside kernels.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cdiv_kernel(out_ptr, n, BLOCK: tl.constexpr):
    num_iters = tl.cdiv(n, BLOCK)                       # scalar ceil(n/BLOCK)
    offs = tl.arange(0, BLOCK)
    val = num_iters.to(tl.float32) + tl.zeros([BLOCK], tl.float32)
    tl.store(out_ptr + offs, val)

BLOCK = 4
n = 10                                                  # ceil(10/4) = 3
out = torch.empty(BLOCK, device='cuda')

cdiv_kernel[(1,)](out, n, BLOCK=BLOCK)
print(out)
# tensor([3., 3., 3., 3.], device='cuda:0')
```

<br>

## 13. Comparison Operators (`< > == !=`)

Comparison operators (比较运算符) return boolean (布尔) block tensors, often combined with `tl.where` or used as masks (掩码).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def cmp_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    is_positive = x > 0.0                               # boolean tensor
    tl.store(out_ptr + offs, is_positive.to(tl.float32))

BLOCK = 4
x = torch.tensor([-1.0, 0.5, -2.0, 3.0], device='cuda')
out = torch.empty(BLOCK, device='cuda')

cmp_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([0., 1., 0., 1.], device='cuda:0')
```

<br>

## 14. `tl.zeros` and `tl.full`

`tl.zeros(shape, dtype)` and `tl.full(shape, value, dtype)` create constant block tensors (常量块张量) inside the kernel, useful as accumulators (累加器).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def init_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    z = tl.zeros([BLOCK], dtype=tl.float32)             # all zeros
    f = tl.full([BLOCK], 7.0, dtype=tl.float32)         # all sevens
    tl.store(out_ptr + offs, z + f)

BLOCK = 4
out = torch.empty(BLOCK, device='cuda')

init_kernel[(1,)](out, BLOCK=BLOCK)
print(out)
# tensor([7., 7., 7., 7.], device='cuda:0')
```

<br>

## 15. `tl.dot` (Matrix Multiply)

`tl.dot(a, b)` is the core matrix multiplication (矩阵乘法) primitive, mapped to Tensor Cores (张量核心) on modern GPUs for peak performance (峰值性能).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # load full A (M,K) and B (K,N)
    a = tl.load(a_ptr + tl.arange(0, M)[:, None] * K + tl.arange(0, K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
    c = tl.dot(a, b)                                    # (M,K) @ (K,N) = (M,N)
    tl.store(c_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], c)

M, N, K = 16, 16, 16                                    # tl.dot needs >=16
a = torch.ones(M, K, device='cuda', dtype=torch.float32)
b = torch.ones(K, N, device='cuda', dtype=torch.float32) * 2.0
c = torch.empty(M, N, device='cuda', dtype=torch.float32)

matmul_kernel[(1,)](a, b, c, M=M, N=N, K=K)
print(c[0, :5])                                         # each = K * 1 * 2 = 32
# tensor([32., 32., 32., 32., 32.], device='cuda:0')
print(c.shape)
# torch.Size([16, 16])
```

<br> <br>
