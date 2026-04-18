---
title: "Triton Overview"
published: 2026-04-15
description: "Triton Overview"
image: ""
tags: ["triton","Triton Overview"]
category: triton
draft: false
lang: ""
---

# I. Triton Overview (概述)

## 1. What is Triton? (它是做什么的)

### 1) Definition (定义)

**Triton** is a GPU programming language (GPU编程语言) and compiler (编译器).

👉 It is used to write custom GPU kernels (自定义GPU算子) using Python.

---

### 2) Core Function (核心功能)

Triton allows you to:

- Control GPU computation (控制GPU计算)
- Optimize memory access (优化内存访问)
- Improve performance (提升性能)

---

## 2. Why Learn Triton? (为什么学习)

### 1) Performance Optimization (性能优化)

Deep learning systems (深度学习系统) rely heavily on GPU computation (GPU计算).

👉 Triton helps:
- Reduce runtime (减少运行时间)
- Improve efficiency (提升效率)

---

### 2) Simpler than CUDA (比CUDA简单)

- CUDA: low-level programming (底层编程)
- Triton: high-level abstraction (高层抽象)

👉 Easier to write and maintain (更易编写和维护)

---

## 3. Comparison (对比)

### 1) Triton vs CUDA

- Triton:
  - Python-based (基于Python)
  - Automatic optimization (自动优化)

- CUDA:
  - C++-based (基于C++)
  - Manual optimization (手动优化)

---

### 2) Triton vs PyTorch

- Triton:
  - Kernel-level programming (算子级编程)
  - Used for optimization (用于优化)

- PyTorch:
  - Model-level framework (模型级框架)
  - Used for training (用于训练)

---

## 4. Runnable Example (可运行示例)

### 1) Vector Addition (向量加法)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x, y):
    n = x.numel()
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output

# Test
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

z = add(x, y)
print(z[:5])
