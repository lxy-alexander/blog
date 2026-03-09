---
title: "pytorch 120 APIs"
published: 2026-03-08
description: "pytorch 120 APIs"
image: ""
tags: ["pytorch","pytorch 120 APIs"]
category: pytorch
draft: false
lang: ""
---



# **I. PyTorch Must-Know 120 APIs — Learning Notes**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> This handbook covers <strong>120 core PyTorch APIs</strong> across 15 chapters, spanning Tensor operations (张量操作), automatic differentiation (自动微分), neural network modules (神经网络模块), optimizers (优化器), data loading (数据加载), model deployment (模型部署), and GPU acceleration (GPU加速). Each entry provides a definition, runnable code example, and key pitfalls. Suitable for beginners, practitioners preparing for interviews, and researchers working with PyTorch 2.x. </div>

------

## 1. Tensor Creation & Basic Operations (张量创建与基础操作)

### 1) `torch.tensor()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates a Tensor (张量) directly from a Python list or NumPy array. You can specify the Data Type (数据类型) and Device (设备) at creation time. </div>

```python
import torch
x = torch.tensor(
    [[1.0, 2.0], [3.0, 4.0]],
    dtype=torch.float32
)
print(x.shape)  # torch.Size([2, 2])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Every call <strong>copies</strong> the data. To share memory with the source array, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.as_tensor()</code> instead.</div>

------

### 2) `torch.zeros()` / `torch.ones()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates all-zero or all-one Tensors (全零/全一张量). Commonly used for bias initialization (偏置初始化) and mask generation (掩码生成). </div>

```python
z = torch.zeros(3, 4)                    # 3×4 all zeros
o = torch.ones(2, 3, dtype=torch.int32) # 2×3 all ones, int type
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">out=</code> parameter writes results into an existing Tensor, avoiding extra memory allocation (内存分配).</div>

------

### 3) `torch.arange()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Generates an Arithmetic Sequence Tensor (等差数列张量), analogous to Python's <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">range()</code>. Supports float step sizes. </div>

```python
t = torch.arange(0, 10, 2)       # tensor([0, 2, 4, 6, 8])
f = torch.arange(0.0, 1.0, 0.25) # tensor([0.00, 0.25, 0.50, 0.75])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Floating-point steps may cause Boundary Precision Issues (边界精度问题). For exact equal-interval sampling, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.linspace()</code>.</div>

------

### 4) `torch.linspace()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Uniformly generates <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">steps</code> points in the interval [start, end]. More precise than <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">arange</code> for float ranges. </div>

```python
t = torch.linspace(0, 1, steps=5)
# tensor([0.00, 0.25, 0.50, 0.75, 1.00])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Commonly used for plotting function curves (函数曲线) and generating uniformly sampled frequency axes (均匀采样频率轴).</div>

------

### 5) `torch.rand()` / `torch.randn()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">rand</code>: Uniform Distribution (均匀分布) U(0,1). <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">randn</code>: Standard Normal Distribution (标准正态分布) N(0,1). </div>

```python
u = torch.rand(2, 3)   # Uniform distribution
n = torch.randn(2, 3)  # Normal distribution

# Fix seed for reproducibility (可复现性)
torch.manual_seed(42)
x = torch.rand(3)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Neural network Weight Initialization (权重初始化) typically uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">randn</code>; Dropout Mask (Dropout掩码) generation uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rand</code>.</div>

------

### 6) `torch.eye()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates an Identity Matrix (单位矩阵) with ones on the diagonal and zeros elsewhere. Commonly used in Linear Algebra (线性代数) and regularization (正则化). </div>

```python
I = torch.eye(3)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Pass <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">n, m</code> to create a Non-square Identity Matrix (非方阵单位矩阵), e.g. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.eye(3, 4)</code>.</div>

------

### 7) `torch.full()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates a Tensor of specified shape where all elements equal a given Fill Value (填充值). </div>

```python
t = torch.full((2, 3), fill_value=7.0)
# tensor([[7., 7., 7.],
#         [7., 7., 7.]])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> More efficient than <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.zeros() * 7</code>. Ideal for creating padding masks (填充掩码).</div>

------

### 8) `torch.from_numpy()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts a NumPy ndarray to a Tensor. The two <strong>share memory (共享内存)</strong> — modifying one affects the other. </div>

```python
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
arr[0] = 99
print(t)  # tensor([99., 2., 3.])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Shared memory is a <span style="color:#C0392B;font-weight:600">double-edged sword</span>: it saves memory but can cause unintended modifications to the source array.</div>

------

### 9) `Tensor.numpy()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts a CPU Tensor back to a NumPy ndarray. Also shares the underlying memory (底层内存). </div>

```python
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()
t[0] = 100
print(arr)  # [100.  2.  3.]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">GPU Tensors must call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.cpu()</code> first</span>, and Tensors with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">requires_grad=True</code> must call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.detach()</code> first.</div>

------

### 10) `torch.empty()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Allocates Uninitialized Memory (未初始化内存) for a Tensor — the fastest allocation method. Values are whatever remains in memory. </div>

```python
t = torch.empty(3, 3)  # Values are undefined
t.fill_(0.5)           # Must fill before reading
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never read values before filling</span>. Best for performance-sensitive scenarios where you immediately overwrite the buffer.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.tensor()</code> for known data, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.zeros/ones/rand/randn()</code> for initialized buffers, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.empty()</code> only when you'll immediately overwrite every element.</div>

------

## 2. Tensor Shape & Dimension Transforms (张量形状与维度变换)

### 1) `Tensor.view()` / `Tensor.reshape()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Reshapes a Tensor without changing its data. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">view</code> requires Contiguous Memory (连续内存); <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">reshape</code> handles non-contiguous cases automatically. </div>

```python
x = torch.arange(12)   # shape [12]
y = x.view(3, 4)       # shape [3, 4]
z = x.reshape(2, 6)    # shape [2, 6]
w = x.reshape(-1, 3)   # -1 auto-infers → shape [4, 3]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Prefer <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">reshape</code> by default; switch to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">view</code> only when you need to guarantee zero-copy memory sharing.</div>

------

### 2) `torch.squeeze()` / `torch.unsqueeze()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">squeeze</code>: Removes dimensions of size 1. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">unsqueeze</code>: Inserts a size-1 dimension at a specified position. </div>

```python
x = torch.zeros(1, 3, 1, 5)
y = x.squeeze()       # [3, 5]

z = torch.zeros(3, 5)
w = z.unsqueeze(0)    # [1, 3, 5]
v = z.unsqueeze(-1)   # [3, 5, 1]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">unsqueeze</code> is frequently used for Broadcasting (广播): add a batch dimension (批次维度) or channel dimension (通道维度) to a vector.</div>

------

### 3) `torch.cat()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Concatenates multiple Tensors along an <strong>existing dimension (已有维度)</strong>. Does <span style="color:#C0392B;font-weight:600">not</span> create a new axis. </div>

```python
a = torch.zeros(2, 3)
b = torch.ones(4, 3)
c = torch.cat([a, b], dim=0)  # shape [6, 3]

d = torch.cat([torch.zeros(2, 2), torch.ones(2, 4)], dim=1)  # shape [2, 6]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> All Tensors must have the same shape on every dimension except the concatenation axis (拼接轴).</div>

------

### 4) `torch.stack()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Stacks Tensors along a <strong>new dimension (新维度)</strong>. All input Tensors must be exactly the same shape. </div>

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = torch.stack([a, b])        # [2, 3]  — new dim=0
d = torch.stack([a, b], dim=1) # [3, 2]  — new dim=1
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Key difference from <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cat</code>: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">stack</code> <span style="color:#C0392B;font-weight:600">requires identical shapes and always creates a new axis</span>.</div>

------

### 5) `Tensor.permute()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Reorders dimensions according to a specified axis order. Equivalent to NumPy's <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">transpose(axes)</code>. </div>

```python
# Convert NCHW → NHWC
x = torch.zeros(8, 3, 224, 224)
y = x.permute(0, 2, 3, 1)  # shape [8, 224, 224, 3]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> After <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">permute</code>, the Tensor becomes Non-contiguous (非连续). Call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.contiguous()</code> if a subsequent op requires contiguous memory.</div>

------

### 6) `Tensor.transpose()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Swaps exactly two specified dimensions. A simplified version of <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">permute</code> for axis swapping. </div>

```python
x = torch.zeros(4, 5, 6)
y = x.transpose(1, 2)   # shape [4, 6, 5]

m = torch.rand(3, 4)
mt = m.t()              # 2D matrix transpose → shape [4, 3]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.transpose(-1, -2)</code> for Batched Matrix Transpose (批量矩阵转置), valid over any batch dimension.</div>

------

### 7) `torch.split()` / `torch.chunk()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">split</code>: Splits by specified sizes. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">chunk</code>: Splits into equal pieces; the last chunk may be smaller. </div>

```python
x = torch.arange(10)
parts = torch.split(x, 3)   # (tensor([0,1,2]), tensor([3,4,5]), tensor([6,7,8]), tensor([9]))
chunks = torch.chunk(x, 3)  # 3 chunks: [0–3], [4–6], [7–9]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Multi-GPU Sharding (多GPU分片) and DataLoader batch splitting internally rely on <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">chunk</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">split</code> logic.</div>

------

### 8) `Tensor.flatten()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Flattens a Tensor to 1D, or flattens a specific range of dimensions. </div>

```python
x = torch.zeros(2, 3, 4)
y = x.flatten()       # shape [24]
z = x.flatten(1, 2)   # shape [2, 12]  — only flatten dims 1–2
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Most commonly used at the CNN → Fully Connected (全连接) transition. Equivalent to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">x.view(x.size(0), -1)</code>.</div>

------

### 9) `torch.broadcast_to()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Broadcasts (广播) a Tensor to a target shape as a read-only view. No data is copied. </div>

```python
x = torch.tensor([1, 2, 3])      # shape [3]
y = torch.broadcast_to(x, (4, 3)) # shape [4, 3]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Broadcasting is the underlying mechanism of most PyTorch arithmetic operations. Understanding it helps avoid Shape Errors (形状错误).</div>

------

### 10) `Tensor.expand()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Expands size-1 dimensions to a specified size. Shares storage (no memory copy), unlike <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">repeat</code>. </div>

```python
x = torch.zeros(3, 1)
y = x.expand(3, 4)   # shape [3, 4] — zero memory copy
z = x.repeat(1, 4)   # shape [3, 4] — actual data copy
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">expand</code> results in a Non-contiguous Tensor; call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.contiguous()</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.clone()</code> before writing.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Master <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">reshape</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">squeeze/unsqueeze</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cat/stack</code>, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">permute</code> — together they cover 90% of all shape manipulation needs.</div>

------

## 3. Math & Statistical Operations (数学与统计运算)

### 1) `torch.matmul()` / `@`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> General Matrix Multiplication (通用矩阵乘法). Supports 2D matrices, batched matrix multiplication (批量矩阵乘), and mixed broadcasting. </div>

```python
a = torch.rand(3, 4)
b = torch.rand(4, 5)
c = torch.matmul(a, b)  # [3, 5]
d = a @ b               # equivalent

# Batched matmul
x = torch.rand(8, 3, 4)
y = torch.rand(8, 4, 5)
z = x @ y               # [8, 3, 5]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The core of Transformer Attention Computation (注意力计算). <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">mm</code> is 2D-only; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">matmul</code> is more general.</div>

------

### 2) `torch.sum()` / `torch.mean()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Computes the sum or mean over all elements or a specified axis. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">keepdim=True</code> preserves the reduced dimension. </div>

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(x.sum())                        # 21.0
print(x.sum(dim=0))                   # [5, 7, 9]
print(x.mean(dim=1, keepdim=True))    # [[2.], [5.]]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">keepdim=True</code> avoids Dimension Alignment Issues (维度对齐问题) during broadcasting.</div>

------

### 3) `torch.max()` / `torch.min()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns the maximum/minimum value. When a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">dim</code> is specified, returns both values and indices (argmax/argmin). </div>

```python
x = torch.tensor([3., 1., 4., 1., 5.])
print(x.max())                 # tensor(5.)
vals, idx = x.max(dim=0)       # vals=5.0, idx=4
idx2 = x.argmax()              # tensor(4)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Classification Prediction Labels (分类网络预测标签): <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">preds = logits.argmax(dim=1)</code>.</div>

------

### 4) `torch.abs()` / `torch.sqrt()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Element-wise absolute value or square root. Used in loss computation (损失计算) and feature normalization (特征归一化). </div>

```python
x = torch.tensor([-1., 4., -9.])
print(torch.abs(x))   # [1., 4., 9.]
y = torch.tensor([1., 4., 9.])
print(torch.sqrt(y))  # [1., 2., 3.]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600"><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.sqrt</code> returns NaN for negative inputs</span>. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.clamp(x, min=0)</code> first.</div>

------

### 5) `torch.clamp()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Clips values to the range [min, max]. Values outside the range are truncated to the boundary. </div>

```python
x = torch.tensor([-2., 0., 3., 8.])
y = torch.clamp(x, min=0., max=5.)  # tensor([0., 0., 3., 5.])
z = torch.clamp(x, min=0.)          # equivalent to ReLU
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The go-to tool for Gradient Clipping (梯度裁剪), normalization, and avoiding <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">log(0)</code>.</div>

------

### 6) `torch.pow()` / `torch.exp()` / `torch.log()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Element-wise power, natural exponent, and natural logarithm. </div>

```python
x = torch.tensor([1., 2., 3.])
print(torch.pow(x, 2))    # [1., 4., 9.]
print(torch.exp(x))       # [e^1, e^2, e^3]
print(torch.log(x))       # [0., 0.693, 1.099]
print(torch.log1p(x))     # Numerically stable log(1+x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Cross-entropy already uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">log_softmax</code> internally. When computing manually, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">log_softmax</code> for Numerical Stability (数值稳定性).</div>

------

### 7) `torch.dot()` / `torch.cross()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">dot</code>: Inner product of 1D vectors. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">cross</code>: Cross product (叉积) of 3D vectors (physics / 3D graphics). </div>

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
print(torch.dot(a, b))  # 32.0
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> For batched inner products, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(a * b).sum(-1)</code> — more efficient than looping <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dot</code>.</div>

------

### 8) `torch.norm()` / `torch.linalg.norm()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Computes vector/matrix norms: L1, L2, Frobenius Norm (Frobenius范数), etc. </div>

```python
x = torch.tensor([3., 4.])
print(torch.linalg.norm(x))          # L2: 5.0
print(torch.linalg.norm(x, ord=1))   # L1: 7.0
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600"><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.norm</code> is deprecated</span>. New code should use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.linalg.norm</code>.</div>

------

### 9) `torch.topk()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns the top-k largest (or smallest) values and their indices from a Tensor. </div>

```python
x = torch.tensor([3., 1., 4., 1., 5., 9.])
vals, idx = torch.topk(x, k=3)
# vals: tensor([9., 5., 4.])
# idx:  tensor([5, 4, 2])

_, top5 = logits.topk(5, dim=1)  # Top-5 accuracy evaluation
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Standard approach for Top-5 Accuracy (Top-5准确率) evaluation. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">largest=False</code> to get the smallest k values.</div>

------

### 10) `torch.unique()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns unique elements from a Tensor, with optional sorting, counting, and inverse mapping (逆映射). </div>

```python
x = torch.tensor([1, 2, 2, 3, 1, 4])
u, cnt = torch.unique(x, return_counts=True)
# u:   tensor([1, 2, 3, 4])
# cnt: tensor([2, 2, 1, 1])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Commonly used for processing Category Labels (类别标签) and deduplicating tokens.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">matmul/@</code> powers Transformers, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clamp</code> guards numerical safety, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">topk</code> drives classification evaluation.</div>

------

## 4. Automatic Differentiation — Autograd (自动微分)

### 1) `Tensor.requires_grad`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Marks whether gradient computation (梯度计算) is needed for this Tensor. It is the entry switch of the Autograd System (自动微分系统). </div>

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2x+3 = 7
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Only Leaf Nodes (叶子节点) created by the user can directly set <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">requires_grad</code>. Intermediate nodes propagate it automatically.</div>

------

### 2) `Tensor.backward()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Triggers Backpropagation (反向传播) from a scalar (or with a gradient tensor argument), computing gradients for all leaf nodes. </div>

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)  # tensor([2., 2., 2.])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Gradients <strong>accumulate</strong> by default. <span style="color:#C0392B;font-weight:600">Call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">optimizer.zero_grad()</code> before each iteration</span>.</div>

------

### 3) `torch.no_grad()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Context manager that disables gradient computation — saves memory and speeds up inference (推理) / evaluation (评估). </div>

```python
model.eval()
with torch.no_grad():
    output = model(x)
    loss = criterion(output, labels)

# Also usable as a decorator
@torch.no_grad()
def predict(x):
    return model(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Always enable this during inference</span>, otherwise inference is slow and VRAM usage is high.</div>

------

### 4) `Tensor.detach()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns a new Tensor disconnected from the Computation Graph (计算图), sharing data but not propagating gradients. </div>

```python
x = torch.tensor([1., 2.], requires_grad=True)
y = x * 3
z = y.detach()       # no gradient tracking
arr = y.detach().numpy()  # must detach before .numpy()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In GAN training, freeze the Generator by calling <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">fake_img.detach()</code> before passing it to the Discriminator (判别器).</div>

------

### 5) `torch.autograd.grad()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Explicitly computes gradients of outputs w.r.t. inputs. Supports Higher-order Gradients (高阶梯度) like Hessians. </div>

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
dy_dx, = torch.autograd.grad(y, x, create_graph=True)  # 1st order
d2y,   = torch.autograd.grad(dy_dx, x)                 # 2nd order
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Core API for MAML (Model-Agnostic Meta-Learning, 模型无关元学习) and Physics-Informed Neural Networks (物理信息神经网络, PINN).</div>

------

### 6) `Tensor.grad` / `Tensor.grad_fn`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">grad</code>: stores the accumulated gradient. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">grad_fn</code>: points to the Backward Function (反向传播函数) that created this Tensor. </div>

```python
x = torch.tensor([1., 2.], requires_grad=True)
y = x * x
print(y.grad_fn)   # <MulBackward0 ...>
y.sum().backward()
print(x.grad)      # tensor([2., 4.])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">grad_fn=None</code> indicates a leaf node. Check with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.is_leaf</code>.</div>

------

### 7) `torch.enable_grad()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Re-enables gradient tracking inside a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">no_grad</code> context, enabling fine-grained control. </div>

```python
with torch.no_grad():
    x = model.encode(data)
    with torch.enable_grad():
        x.requires_grad_(True)
        loss = head(x)  # only this part tracked
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Useful for Partial Freeze Training (部分冻结训练), e.g., fine-tuning only the last layer.</div>

------

### 8) `register_hook()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Registers a hook function on a Tensor's backward pass, enabling inspection or modification of intermediate gradients. </div>

```python
grads = []
def save_grad(g):
    grads.append(g.clone())

x = torch.rand(3, requires_grad=True)
y = (x**2).sum()
x.register_hook(save_grad)
y.backward()
print(grads[0])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Invaluable for debugging Gradient Vanishing/Explosion (梯度消失/爆炸) and implementing gradient penalties like WGAN-GP.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Always pair <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">backward()</code> with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">zero_grad()</code>, wrap inference in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">no_grad()</code>, and use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">detach()</code> to stop gradients from crossing module boundaries.</div>

------

## 5. Neural Network Modules — `nn.Module` (神经网络模块)

### 1) `nn.Module`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> The base class for all neural networks in PyTorch. Manages parameters (参数), sub-modules (子模块), and defines the forward pass (前向传播) logic. </div>

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Only implement <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__init__</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">forward</code>. The backward pass is handled automatically by autograd.</div>

------

### 2) `nn.Linear()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Fully Connected Layer (全连接层) / Affine Transformation (仿射变换): y = xW<sup>T</sup> + b. The most fundamental learnable layer. </div>

```python
fc = nn.Linear(in_features=128, out_features=64, bias=True)
x = torch.rand(32, 128)
out = fc(x)  # shape [32, 64]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Weight shape is <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[out, in]</code>. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">bias=False</code> is commonly paired with BatchNorm.</div>

------

### 3) `nn.Conv2d()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> 2D Convolutional Layer (二维卷积层). Extracts local spatial features; the core building block of CNNs (卷积神经网络). </div>

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
x = torch.rand(8, 3, 224, 224)
out = conv(x)  # [8, 64, 224, 224]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">padding=kernel_size//2</code> preserves feature map size (Same Padding, 等尺寸填充).</div>

------

### 4) `nn.BatchNorm2d()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Normalizes each channel of a mini-batch (小批量归一化). Accelerates training and mitigates gradient vanishing (梯度消失). </div>

```python
bn = nn.BatchNorm2d(num_features=64)
x = torch.rand(8, 64, 28, 28)
out = bn(x)
# Standard order: Conv → BN → ReLU
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">BN is unstable when batch_size=1</span>. Switch to GroupNorm or LayerNorm in that case.</div>

------

### 5) `nn.Dropout()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> During training, randomly zeros out a fraction of neurons — a Regularization (正则化) technique to prevent Overfitting (过拟合). </div>

```python
dropout = nn.Dropout(p=0.5)
x = torch.rand(4, 128)
out = dropout(x)       # 50% elements zeroed during train mode

dropout.eval()
out_eval = dropout(x)  # identical to x in eval mode
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Forgetting <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">model.eval()</code></span> is the #1 most common bug causing non-deterministic inference results.</div>

------

### 6) `nn.Sequential()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Chains a series of layers in order, executing each <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">forward</code> call sequentially. Simplifies model definition. </div>

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)
out = model(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">OrderedDict</code> to name layers: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nn.Sequential(OrderedDict([('fc', nn.Linear(...))]))</code>.</div>

------

### 7) `nn.ModuleList()` / `nn.ModuleDict()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Registers sub-modules as a list or dictionary so that their parameters are correctly tracked and saved. </div>

```python
layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])
for layer in layers:
    x = torch.relu(layer(x))

heads = nn.ModuleDict({
    'cls': nn.Linear(64, 10),
    'reg': nn.Linear(64, 1)
})
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Plain Python <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">list</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code> are not registered</span> — <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">parameters()</code> will miss them!</div>

------

### 8) `nn.Embedding()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Maps integer indices to dense vectors (稠密向量). The standard Word Embedding Lookup Table (词向量查找表) in NLP. </div>

```python
vocab_size, embed_dim = 10000, 128
emb = nn.Embedding(vocab_size, embed_dim)
ids = torch.randint(0, vocab_size, (16, 50))  # [batch, seq_len]
out = emb(ids)  # [16, 50, 128]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">padding_idx</code> specifies a padding token whose embedding is excluded from gradient updates.</div>

------

### 9) `nn.LSTM()` / `nn.GRU()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Long Short-Term Memory (长短时记忆) and Gated Recurrent Unit (门控循环单元) — classic recurrent layers for sequence data (序列数据). </div>

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
x = torch.rand(8, 50, 128)  # [batch, seq, feat]
out, (h, c) = lstm(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">batch_first=True</code> sets the input format to [B, T, F], which is more intuitive. The default is [T, B, F].</div>

------

### 10) `nn.MultiheadAttention()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Multi-head Self-Attention (多头自注意力机制) — the core component of the Transformer Architecture (Transformer架构). </div>

```python
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.rand(4, 100, 512)
out, weights = attn(query=x, key=x, value=x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">key_padding_mask</code> to mask padding tokens; use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">attn_mask</code> for Causal Masking (因果掩码) in decoders.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Every custom network inherits from <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nn.Module</code>; use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ModuleList/Dict</code> (not plain lists) to ensure parameters are tracked.</div>

------

## 6. Activation Functions & Loss Functions (激活函数与损失函数)

### 1) `nn.ReLU()` / `F.relu()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Rectified Linear Unit (修正线性单元): max(0, x). Alleviates gradient vanishing; the most widely used activation function. </div>

```python
import torch.nn.functional as F
x = torch.randn(4, 64)
out1 = F.relu(x)                     # functional call
relu = nn.ReLU(inplace=True)
out2 = relu(x)                       # module call (can go in Sequential)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">inplace=True</code> saves memory but modifies the original tensor — <span style="color:#C0392B;font-weight:600">be careful when using autograd hooks</span>.</div>

------

### 2) `nn.GELU()` / `nn.SiLU()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">GELU</code>: the Transformer standard activation. <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">SiLU</code> (Swish): used in EfficientNet and mobile models. </div>

```python
gelu = nn.GELU()
silu = nn.SiLU()
x = torch.randn(4, 64)
print(gelu(x).shape)  # [4, 64]
print(silu(x).shape)  # [4, 64]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> BERT/GPT default to GELU; SiLU performs better on mobile-end models.</div>

------

### 3) `nn.Softmax()` / `F.softmax()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts logits to a Probability Distribution (概率分布) summing to 1. Output layer for multi-class classification (多分类任务). </div>

```python
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)   # tensor([0.659, 0.242, 0.099])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Do NOT manually add Softmax when using <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CrossEntropyLoss</code></span> — it already includes it internally.</div>

------

### 4) `nn.CrossEntropyLoss()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Multi-class Cross-Entropy Loss (多分类交叉熵损失) — internally fuses LogSoftmax + NLLLoss for numerical stability. </div>

```python
criterion = nn.CrossEntropyLoss()
logits = torch.rand(8, 10)
labels = torch.randint(0, 10, (8,))
loss = criterion(logits, labels)
loss.backward()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">label_smoothing</code> parameter (≥ PyTorch 1.8) effectively prevents Overconfidence (过度自信) and improves generalization.</div>

------

### 5) `nn.BCEWithLogitsLoss()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Binary / Multi-label Classification Loss (二分类/多标签分类损失). More numerically stable than applying Sigmoid then BCE. </div>

```python
criterion = nn.BCEWithLogitsLoss()
logits = torch.rand(8, 1)                        # no sigmoid needed
targets = torch.randint(0, 2, (8, 1)).float()
loss = criterion(logits, targets)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> For multi-label classification, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">targets</code> is a float matrix with each bit independent, not a class index.</div>

------

### 6) `nn.MSELoss()` / `nn.L1Loss()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Mean Squared Error (均方误差) and Mean Absolute Error (平均绝对误差) — for continuous value prediction (连续值预测) / regression (回归). </div>

```python
mse = nn.MSELoss()
mae = nn.L1Loss()
pred = torch.rand(4, 1)
target = torch.rand(4, 1)
print(mse(pred, target))
print(mae(pred, target))
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> L1 is more robust to outliers. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">SmoothL1Loss</code> (Huber Loss) combines both — recommended for object detection (目标检测).</div>

------

### 7) `nn.KLDivLoss()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> KL Divergence Loss (KL散度损失) — measures the difference between two probability distributions. Used in knowledge distillation (知识蒸馏) and VAE. </div>

```python
kl = nn.KLDivLoss(reduction='batchmean')
log_p = F.log_softmax(student_logits, dim=-1)   # input: log prob
q = F.softmax(teacher_logits, dim=-1)            # target: prob
loss = kl(log_p, q)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Input must be log-probabilities; target must be probabilities</span>. This matches the mathematical definition.</div>

------

### 8) `nn.LayerNorm()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Normalizes over the last N dimensions. Independent of batch size — the standard normalization in Transformers (Transformer标配). </div>

```python
ln = nn.LayerNorm(normalized_shape=512)
x = torch.rand(4, 100, 512)
out = ln(x)  # shape [4, 100, 512]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Outperforms BatchNorm for variable-length NLP sequences (可变长序列) and small batch sizes.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CrossEntropyLoss</code> for multi-class, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BCEWithLogitsLoss</code> for multi-label, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">SmoothL1</code> for regression — and never apply Softmax before CrossEntropy.</div>

------

## 7. Optimizers & Learning Rate Schedulers (优化器与学习率调度)

### 1) `torch.optim.SGD()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Stochastic Gradient Descent (随机梯度下降). Supports momentum (动量), weight decay (权重衰减), and Nesterov momentum. </div>

```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9,
    weight_decay=1e-4, nesterov=True
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> SGD+momentum is still common in CV; final accuracy sometimes surpasses Adam.</div>

------

### 2) `torch.optim.Adam()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Adaptive Moment Estimation (自适应矩估计). Combines AdaGrad and RMSProp. The default optimizer for most tasks. </div>

```python
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4
)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> For NLP/Transformer scenarios, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AdamW</code> (decoupled weight decay).</div>

------

### 3) `torch.optim.AdamW()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Improved Adam with correctly decoupled L2 regularization (解耦L2正则). The go-to optimizer for training Transformers. </div>

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-5, weight_decay=0.01
)  # Standard config for BERT/GPT fine-tuning
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In original Adam, L2 regularization is entangled with adaptive learning rate scaling. AdamW fixes this by decoupling them.</div>

------

### 4) `optimizer.zero_grad()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Clears all parameter gradient buffers. <strong>Must be called before each backward pass</strong>. </div>

```python
for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()   # 1. clear
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()         # 3. backward
        optimizer.step()        # 4. update
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">zero_grad(set_to_none=True)</code> uses less memory and is recommended for PyTorch ≥ 1.7.</div>

------

### 5) `lr_scheduler.StepLR()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Multiplies the learning rate by <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">gamma</code> every <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">step_size</code> epochs — stepwise decay (阶梯式衰减). </div>

```python
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Call at end of each epoch:
scheduler.step()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In modern PyTorch, call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">optimizer.step()</code> <em>before</em> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scheduler.step()</code>.</div>

------

### 6) `lr_scheduler.CosineAnnealingLR()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Cosine Annealing Decay (余弦退火调度): LR oscillates between [eta_min, lr] following a cosine curve. Excellent convergence. </div>

```python
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Combine with Warm Restarts (热重启, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CosineAnnealingWarmRestarts</code>) to escape local optima.</div>

------

### 7) `lr_scheduler.OneCycleLR()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Super-Convergence Training Strategy (超融合训练策略): LR rises then falls in a single cycle. Significantly reduces convergence time. </div>

```python
scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01,
    steps_per_epoch=len(loader), epochs=10
)
scheduler.step()  # call after every step (not epoch)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Set <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">max_lr</code> to the highest stable LR found by a learning rate finder.</div>

------

### 8) `torch.optim.LBFGS()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Quasi-Newton second-order optimizer (拟牛顿二阶优化器). Suited for small datasets. Requires a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">closure</code> function. </div>

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1)

def closure():
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    return loss

optimizer.step(closure)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Preferred for Neural Style Transfer (神经风格迁移) and other small-scale, high-precision convergence tasks.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Default to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AdamW</code> + <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CosineAnnealingLR</code> for Transformers, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">SGD</code> + <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">StepLR</code> for classic CNN image tasks.</div>

------

## 8. Data Loading & Preprocessing (数据加载与预处理)

### 1) `torch.utils.data.Dataset`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Abstract base class for custom datasets. Must implement <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__len__</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__getitem__</code>. </div>

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Put all preprocessing / augmentation inside <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__getitem__</code> for Lazy Loading (懒加载).</div>

------

### 2) `torch.utils.data.DataLoader`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Wraps a Dataset into an iterable batch loader with parallel reading (并行读取) and data shuffling (数据打乱). </div>

```python
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset=train_ds, batch_size=32,
    shuffle=True, num_workers=4, pin_memory=True
)
for x, y in loader:
    ...
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> On Windows, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">num_workers > 0</code> <span style="color:#C0392B;font-weight:600">requires <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">if __name__ == '__main__':</code></span> guard.</div>

------

### 3) `torchvision.transforms`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Image preprocessing and data augmentation (数据增强) library. Chain multiple transforms with <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">Compose</code>. </div>

```python
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The Normalize parameters are ImageNet statistics. Keep them consistent when using Transfer Learning (迁移学习).</div>

------

### 4) `torchvision.datasets.ImageFolder`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Automatically builds an image classification dataset from directory structure — subdirectory names become class labels (类别标签). </div>

```python
from torchvision.datasets import ImageFolder
# data/train/cat/*.jpg, data/train/dog/*.jpg
ds = ImageFolder(root='data/train', transform=transform)
print(ds.classes)   # ['cat', 'dog']
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Save the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">class_to_idx</code> dictionary alongside the model checkpoint.</div>

------

### 5) `torch.utils.data.random_split()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Randomly splits a dataset into train/validation subsets by specified lengths. </div>

```python
from torch.utils.data import random_split
n_val = int(len(dataset) * 0.2)
train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Pass <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">generator=torch.Generator().manual_seed(42)</code> for reproducible splits.</div>

------

### 6) `torchvision.models` (pretrained)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Provides many pre-trained models: ResNet, VGG, ViT, etc. Enables rapid Transfer Learning (迁移学习). </div>

```python
import torchvision.models as models
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(2048, 10)  # replace head for fine-tuning
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Freeze the backbone: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">for p in model.parameters(): p.requires_grad = False</code>.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>The data pipeline is: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Dataset</code> (what) → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">transforms</code> (how to augment) → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">DataLoader</code> (how to batch).</div>

------

## 9. Model Saving, Loading & Deployment (模型保存、加载与部署)

### 1) `torch.save()` / `torch.load()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Serializes / deserializes any Python object (model, tensor, dict) to/from a file. </div>

```python
# Recommended: save only the weights dict
torch.save(model.state_dict(), 'model_weights.pth')

# Load
state = torch.load('model_weights.pth', map_location='cpu')
model.load_state_dict(state)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Saving the entire model object couples code paths. <span style="color:#C0392B;font-weight:600">Always save only <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">state_dict</code></span>.</div>

------

### 2) `model.state_dict()` / `load_state_dict()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Gets / loads an ordered dictionary of model parameters. The core interface for Transfer Learning (迁移学习) and checkpoint resuming (断点续训). </div>

```python
checkpoint = {
    'epoch': epoch,
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'loss': best_loss
}
torch.save(checkpoint, 'ckpt.pth')
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">strict=False</code> allows partial loading (skips missing keys) — common in transfer learning.</div>

------

### 3) `torch.jit.script()` / `torch.jit.trace()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Compiles a model to TorchScript for deployment in Python-free environments (C++, mobile). </div>

```python
# trace: follow execution path (no control flow)
traced = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
traced.save('traced.pt')

# script: supports dynamic control flow
scripted = torch.jit.script(model)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Models with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">if/for</code> branches → use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">script</code>. Pure forward-pass models → use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">trace</code> (faster).</div>

------

### 4) `torch.onnx.export()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Exports a PyTorch model to ONNX format for cross-framework deployment (TensorRT, OpenVINO). </div>

```python
torch.onnx.export(
    model, torch.rand(1, 3, 224, 224), 'model.onnx',
    opset_version=17,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dynamic_axes</code> enables Dynamic Batch Size (动态批次大小) — essential for production deployment. Validate with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">onnxruntime</code>.</div>

------

### 5) `model.parameters()` / `named_parameters()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Iterates over all learnable parameters. The <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">named_</code> version also returns parameter names. </div>

```python
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total/1e6:.1f}M')

for name, p in model.named_parameters():
    print(name, p.shape)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Set layer-specific learning rates by passing <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[{'params': p, 'lr': lr}]</code> list to the optimizer.</div>

------

### 6) `model.train()` / `model.eval()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Switches between training and evaluation modes — affects Dropout and BatchNorm behavior. </div>

```python
model.train()
for x, y in train_loader:
    loss = criterion(model(x), y)
    loss.backward(); optimizer.step()

model.eval()
with torch.no_grad():
    for x, y in val_loader:
        pred = model(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Forgetting <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">model.eval()</code></span> is the most common reason for unstable inference results.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Always save <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">state_dict</code> (not the model object), and remember the deploy path: PyTorch → TorchScript / ONNX → Runtime.</div>

------

## 10. GPU Acceleration & Distributed Training (GPU加速与分布式训练)

### 1) `Tensor.to()` / `.cuda()` / `.cpu()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Moves Tensors or models to a specified device (GPU/CPU). The fundamental operation for GPU training (GPU训练). </div>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
result = output.cpu().numpy()  # move back to CPU
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Model and data must be on the same device</span>. Mixing CPU/GPU Tensors throws a runtime error.</div>

------

### 2) `torch.cuda.amp` — Automatic Mixed Precision (自动混合精度)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Automatically switches between FP16 and FP32, reducing VRAM usage and accelerating training. </div>

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(x)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">GradScaler</code> prevents FP16 Gradient Underflow (梯度下溢). Recommended for all modern GPU training.</div>

------

### 3) `nn.DataParallel()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Single-machine multi-GPU Data Parallel (数据并行) training. Automatically splits batches and aggregates gradients. </div>

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to('cuda')

# Access original model
sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">DataParallel efficiency is limited by Python GIL</span>. For large-scale training, use DistributedDataParallel (DDP).</div>

------

### 4) `nn.parallel.DistributedDataParallel()` — DDP

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Distributed Data Parallel: one process per GPU. Communication efficiency far exceeds DataParallel. </div>

```python
import torch.distributed as dist
dist.init_process_group('nccl')
local_rank = int(os.environ['LOCAL_RANK'])
model = model.to(local_rank)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Launch with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun --nproc_per_node=4 train.py</code>. Pair with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">DistributedSampler</code>.</div>

------

### 5) `torch.cuda.memory_summary()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Prints detailed GPU VRAM usage to help diagnose Out-of-Memory (OOM, 显存溢出) issues. </div>

```python
print(torch.cuda.memory_summary())

alloc = torch.cuda.memory_allocated()
total = torch.cuda.get_device_properties(0).total_memory
print(f'{alloc/1e9:.1f}GB used')
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> After OOM, call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.cuda.empty_cache()</code> to release cached memory — but it cannot free memory in active use.</div>

------

### 6) `torch.compile()` (PyTorch 2.0+)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Compiles the model into optimized kernels using graph capture and Triton operators, dramatically accelerating training/inference. </div>

```python
model = torch.compile(model)

# Different modes
model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> First run has Compilation Overhead (编译开销) (warmup). <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">fullgraph=True</code> forbids graph breaks for maximum performance.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Use AMP for every GPU training job; prefer DDP over DataParallel for multi-GPU; add <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.compile()</code> as a one-line speed boost in PyTorch 2.x.</div>

------

## 11. Advanced Features & Utilities (高级特性与实用工具)

### 1) `torch.nn.utils.clip_grad_norm_()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Clips the global L2 norm of all parameter gradients to prevent Gradient Explosion (梯度爆炸). Essential for RNN/Transformer training. </div>

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">max_norm=1.0</code> is standard for Transformers. <span style="color:#C0392B;font-weight:600">Must call after <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">backward()</code>, before <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">step()</code></span>.</div>

------

### 2) `torch.nn.utils.weight_norm()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Decomposes parameters into direction and magnitude, accelerating convergence. Used in WaveNet and generative models. </div>

```python
from torch.nn.utils import weight_norm, remove_weight_norm
wn_conv = weight_norm(nn.Conv1d(64, 64, 3, padding=1))
remove_weight_norm(wn_conv)  # merge before deployment
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Always call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">remove_weight_norm</code> before deployment to merge decomposed parameters.</div>

------

### 3) `torch.nn.functional.interpolate()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Upsamples or downsamples feature maps with bilinear, nearest, bicubic, etc. interpolation modes. </div>

```python
x = torch.rand(1, 64, 28, 28)
up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
print(up.shape)  # [1, 64, 56, 56]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">align_corners=False</code> matches TensorFlow's default behavior — important when migrating models.</div>

------

### 4) `torch.nn.functional.grid_sample()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Samples from a feature map at normalized grid coordinates. The core of Spatial Transformer Networks (空间变换网络, STN). </div>

```python
theta = torch.eye(2, 3, dtype=torch.float).unsqueeze(0)
grid = F.affine_grid(theta, x.size())
out = F.grid_sample(x, grid, mode='bilinear')
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Coordinate range is [-1, 1]. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">padding_mode='reflection'</code> is more natural for image borders.</div>

------

### 5) `torch.einsum()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Einstein Summation (爱因斯坦求和): expresses complex tensor operations as a concise string equation. </div>

```python
c = torch.einsum('ij,jk->ik', a, b)  # matrix multiply

# Attention scores: Q:[B,H,L,D], K:[B,H,L,D]
scores = torch.einsum('bhld,bhmd->bhlm', Q, K)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Mastering <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">einsum</code> dramatically simplifies Transformer and graph neural network code.</div>

------

### 6) `torch.profiler.profile()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Performance profiler that records per-operator CPU/GPU time and memory usage to locate bottlenecks. </div>

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(x)
print(prof.key_averages().table(sort_by='cuda_time_total'))
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Export a Chrome trace with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">prof.export_chrome_trace()</code> for visual bottleneck analysis in a browser.</div>

------

### 7) `torch.nn.init.*`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Provides Xavier, Kaiming, Orthogonal and other Parameter Initialization (参数初始化) strategies. Directly impacts training stability. </div>

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Sigmoid/Tanh → Xavier; ReLU family → Kaiming; Transformer → Orthogonal initialization.</div>

------

### 8) `torch.Tensor.item()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts a single-element Tensor to a Python scalar. Commonly used to log loss values. </div>

```python
loss = criterion(output, label)
loss_val = loss.item()        # detaches from graph
print(f'Loss: {loss_val:.4f}')

# WRONG: total_loss += loss  ← graph grows → OOM
# CORRECT:
total_loss += loss.item()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Accumulating <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">loss</code> tensors directly causes the computation graph to grow unboundedly → OOM</span>. Always use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.item()</code>.</div>

------

### 9) `torch.Tensor.clone()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates a deep copy (深拷贝) of a Tensor with fully independent data, while preserving gradient propagation. </div>

```python
x = torch.rand(3, requires_grad=True)
y = x.clone()           # gradient can still propagate
z = x.detach().clone()  # gradient detached

buf = torch.empty(3)
buf.copy_(x)            # in-place copy
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clone().detach()</code> for Target Network (目标网络) parameter copying in RL / momentum update.</div>

------

### 10) `torch.where()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Conditional selection: returns values from <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">x</code> where condition is True, else from <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">y</code>. Vectorized if-else. </div>

```python
x = torch.tensor([-1., 2., -3., 4.])
y = torch.zeros_like(x)
out = torch.where(x > 0, x, y)  # tensor([0., 2., 0., 4.])  — manual ReLU
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Both branches participate in gradient computation; the gradient of the unselected branch is zero.</div>

------

### 11) `torch.gather()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Gathers values from a source Tensor by an index Tensor — enables irregular indexing (不规则索引) like NMS. </div>

```python
logits = torch.rand(4, 10)
targets = torch.tensor([3, 7, 1, 5]).unsqueeze(1)  # [4, 1]
scores = logits.gather(dim=1, index=targets)         # [4, 1]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Core tool for sequence decoding, top-k sampling, and Q-value selection in Reinforcement Learning (强化学习).</div>

------

### 12) `torch.scatter_()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Scatters values from <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">src</code> into <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">self</code> at positions specified by <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">index</code> (in-place). </div>

```python
y = torch.zeros(4, 5)
labels = torch.tensor([[2], [0], [4], [1]])
y.scatter_(dim=1, index=labels, value=1.0)  # one-hot encoding
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scatter_add_</code> implements segment sum — the fundamental primitive for Graph Neural Network (图神经网络) message aggregation.</div>

------

### 13) `torch.masked_fill()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Fills positions where the mask is True with a specified value. Essential for Attention Masking (注意力掩码). </div>

```python
L = 5
mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = torch.rand(L, L)
scores = scores.masked_fill(mask, float('-inf'))  # causal mask
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Transformer decoder's Causal Self-Attention (因果自注意力) must use this to block future information.</div>

------

### 14) `torch.nn.utils.rnn.pad_sequence()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Pads a list of variable-length sequences to a uniform-length tensor for NLP batch processing. </div>

```python
from torch.nn.utils.rnn import pad_sequence
seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
padded = pad_sequence(seqs, batch_first=True)  # [3, 3]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Combine with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pack_padded_sequence</code> to skip LSTM computation on padding positions.</div>

------

### 15) `nn.TransformerEncoder()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Multi-layer Transformer Encoder with built-in Multi-head Attention + FFN + Residual Normalization. </div>

```python
enc_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
encoder = nn.TransformerEncoder(enc_layer, num_layers=6)
out = encoder(src, src_key_padding_mask=m)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">norm_first=True</code> (Pre-LN) trains more stably — recommended for large models.</div>

------

### 16) `torch.Tensor.contiguous()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns a Contiguous Memory (连续内存) copy of the Tensor. Returns itself (zero overhead) if already contiguous. </div>

```python
x = torch.rand(4, 5, 6)
y = x.permute(2, 0, 1)         # non-contiguous
print(y.is_contiguous())        # False
z = y.contiguous()
w = z.view(6, -1)               # safe to view now
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In most cases, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">reshape</code> handles this automatically. Call manually only when contiguous memory is truly required.</div>

------

### 17) `torch.Tensor.type()` / `.to(dtype)`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts the Tensor's Data Type (数据类型): float32 ↔ float16 ↔ int64, etc. </div>

```python
x = torch.tensor([1, 2, 3])
f = x.float()   # int → float32
h = x.half()    # float32 → float16
l = x.long()    # → int64
y = x.to(dtype=torch.float32)  # recommended
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Cross-entropy labels need <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">long()</code>; normalized image pixels need <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">float()</code>; inference acceleration uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">half()</code>.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>The advanced toolkit: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clip_grad_norm_</code> (stability), <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">einsum</code> (clarity), <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather/scatter</code> (indexing), <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">masked_fill</code> (attention), <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">item()</code> (memory safety).</div>

------

## 12. Convolution, Pooling & Normalization Layers (卷积、池化与正则化层)

### 1) `nn.MaxPool2d()` / `nn.AvgPool2d()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> 2D Max / Average Pooling (最大/平均池化). Downsamples feature maps using a sliding window, reducing spatial size. </div>

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.rand(8, 64, 28, 28)
out = pool(x)  # [8, 64, 14, 14]

gap = nn.AdaptiveAvgPool2d((1, 1))
feat = gap(out)  # [8, 64, 1, 1] — global avg pool
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AdaptiveAvgPool2d((1,1))</code> is the standard Global Average Pooling (全局平均池化) in ResNet's classification head.</div>

------

### 2) `nn.ConvTranspose2d()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Transposed Convolution (转置卷积 / 反卷积) for upsampling. Core layer in U-Net and GAN Generators. </div>

```python
deconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
x = torch.rand(4, 64, 14, 14)
out = deconv(x)  # [4, 32, 28, 28]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">kernel=4, stride=2, padding=1</code> is the classic recipe that exactly doubles the spatial size.</div>

------

### 3) `nn.GroupNorm()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Splits channels into groups and normalizes within each group. Independent of batch size — outperforms BN in small-batch scenarios. </div>

```python
gn = nn.GroupNorm(num_groups=8, num_channels=32)
x = torch.rand(2, 32, 64, 64)
out = gn(x)  # shape unchanged
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Recommended for object detection / instance segmentation (small batch). <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">num_groups=1</code> ≡ LayerNorm.</div>

------

### 4) `nn.InstanceNorm2d()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Normalizes each sample and each channel independently. Standard normalization for Image Style Transfer (图像风格迁移). </div>

```python
inst = nn.InstanceNorm2d(num_features=64, affine=True)
x = torch.rand(4, 64, 256, 256)
out = inst(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">affine=True</code> adds learnable scale/shift parameters for better style adaptation.</div>

------

### 5) `nn.Upsample()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Module-form wrapper around <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">F.interpolate</code>. No learnable parameters; can be placed in <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">Sequential</code>. </div>

```python
up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
x = torch.rand(4, 32, 14, 14)
out = up(x)  # [4, 32, 28, 28]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Combine with ConvTranspose2d for learnable upsampling control (learned vs fixed).</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Normalization choice: BN (large batch) → GN (small batch/detection) → LN (NLP) → IN (style transfer).</div>

------

## 13. Indexing, Selection & Advanced Operations (索引、选择与高级操作)

### 1) `torch.nonzero()` / `torch.argwhere()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Returns the coordinates of all non-zero (or True) elements. Used for Sparse Operations (稀疏操作). </div>

```python
x = torch.tensor([[0, 1, 0], [2, 0, 3]])
idx = torch.nonzero(x)           # tensor([[0,1],[1,0],[1,2]])
idx2 = torch.argwhere(x > 0)     # PyTorch 1.9+
```

------

### 2) `torch.index_select()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Selects elements along a dimension by index tensor. Similar to NumPy fancy indexing. </div>

```python
x = torch.rand(5, 4)
idx = torch.tensor([0, 2, 4])
out = torch.index_select(x, dim=0, index=idx)  # shape [3, 4]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Index must be a 1D LongTensor; more efficient than boolean masking for this case.</div>

------

### 3) `torch.masked_select()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Selects elements by boolean mask. Returns a flattened 1D Tensor. </div>

```python
x = torch.randn(3, 3)
mask = x > 0
pos_vals = torch.masked_select(x, mask)  # all positive values, 1D
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Result is always 1D. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">masked_fill</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scatter_</code> to reconstruct shape.</div>

------

### 4) `torch.sort()` / `torch.argsort()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Sorts a Tensor along a dimension, returning sorted values and original indices. </div>

```python
x = torch.tensor([3., 1., 4., 1., 5., 9.])
vals, idx = torch.sort(x, descending=True)  # [9,5,4,3,1,1]
order = torch.argsort(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Key step in NMS (Non-Maximum Suppression, 非极大抑制): sort boxes by confidence descending.</div>

------

### 5) `torch.cumsum()` / `torch.cumprod()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Cumulative sum (累积和) or cumulative product (累积积) along a dimension. </div>

```python
x = torch.tensor([1., 2., 3., 4.])
print(torch.cumsum(x, dim=0))   # tensor([1., 3., 6., 10.])
print(torch.cumprod(x, dim=0))  # tensor([1., 2., 6., 24.])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cumsum</code> is an efficient alternative for generating causal attention masks (lower triangular).</div>

------

### 6) `torch.flip()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Flips a Tensor along specified dimensions — mirror flip augmentation or reverse operation. </div>

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
h = torch.flip(x, dims=[1])  # horizontal: [[3,2,1],[6,5,4]]
v = torch.flip(x, dims=[0])  # vertical: [[4,5,6],[1,2,3]]
```

------

### 7) `torch.bucketize()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Assigns continuous values to discrete buckets (离散化) by given boundaries. Analogous to NumPy's <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">digitize</code>. </div>

```python
boundaries = torch.tensor([0.0, 0.5, 1.0])
x = torch.tensor([-0.1, 0.3, 0.7, 1.5])
bins = torch.bucketize(x, boundaries)  # tensor([0, 1, 2, 3])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Useful for Feature Engineering (特征工程) — binning continuous features and custom quantile normalization.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather</code> picks values by index; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scatter_</code> puts values by index; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">masked_fill</code> overwrites by condition.</div>

------

## 14. Randomness & Reproducibility (随机性与可复现性)

### 1) `torch.manual_seed()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Sets the global random seed to ensure consistent results across runs (实验可复现性). </div>

```python
import random, numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Also set <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.backends.cudnn.deterministic=True</code> for fully deterministic behavior.</div>

------

### 2) `torch.Generator()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Creates an independent Random Number Generator (随机数生成器) object, avoiding interference with the global seed. </div>

```python
g = torch.Generator()
g.manual_seed(42)
x = torch.rand(3, generator=g)  # independent state

loader = DataLoader(ds, shuffle=True, generator=g)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Safer than global seeds in multi-thread / multi-process scenarios.</div>

------

### 3) `torch.distributions.*`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Probability distribution library supporting sampling and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">log_prob</code> computation. Foundation for VAE and Reinforcement Learning (强化学习). </div>

```python
from torch.distributions import Normal, Categorical

# VAE reparameterization trick (重参数化技巧)
dist = Normal(loc=mu, scale=torch.exp(logvar))
z = dist.rsample()      # differentiable sampling
log_p = dist.log_prob(z)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rsample()</code> is differentiable (reparameterization); <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sample()</code> is not (for policy gradients).</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Always call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set_seed()</code> at the start of every experiment and use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rsample()</code> for differentiable stochastic layers.</div>

------

## 15. Utilities & Performance Tips (实用工具与性能技巧)

### 1) `torch.no_grad()` vs `torch.inference_mode()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">inference_mode</code> is more aggressive than <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">no_grad</code>: it skips version counting entirely for faster pure inference. </div>

```python
with torch.no_grad():
    out1 = model(x)

@torch.inference_mode()
def predict(x):
    return model(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> PyTorch 1.9+ recommends <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">inference_mode</code> for pure inference — faster than <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">no_grad</code>.</div>

------

### 2) `torch.Tensor.pin_memory()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Pins CPU Tensors to Page-locked Memory (页锁定内存), dramatically accelerating CPU→GPU data transfer. </div>

```python
loader = DataLoader(dataset, pin_memory=True, num_workers=4)
for x, y in loader:
    x = x.to('cuda', non_blocking=True)  # async transfer
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pin_memory=True</code> + <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">non_blocking=True</code> enables CPU data loading to overlap with GPU computation (Pipeline, 流水线并行).</div>

------

### 3) `torch.utils.checkpoint.checkpoint()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Gradient Checkpointing (梯度检查点): trades recomputation for VRAM savings. Can save 50%+ VRAM for very large models. </div>

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.heavy_block, x)  # no intermediate activations saved
    return self.head(x)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Trades ~30% extra training time for drastically reduced VRAM — enables training models that would otherwise OOM.</div>

------

### 4) `torch.nn.functional.one_hot()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Converts integer class indices to One-hot Encoding (独热编码) tensors. </div>

```python
labels = torch.tensor([0, 2, 1, 3])
one_hot = F.one_hot(labels, num_classes=4).float()
# tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Output defaults to LongTensor. Call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.float()</code> before participating in loss computation.</div>

------

### 5) `torch.nn.functional.cosine_similarity()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Computes Cosine Similarity (余弦相似度) between two groups of vectors. Core metric in Contrastive Learning (对比学习) — SimCLR, CLIP. </div>

```python
a = torch.randn(8, 128)
b = torch.randn(8, 128)
sim = F.cosine_similarity(a, b, dim=-1)  # shape [8], range [-1, 1]

# Similarity matrix for contrastive learning
mat = a @ b.T  # [8, 8]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Equivalent to (a/||a||) · (b/||b||). In contrastive learning, L2-normalize first then use dot product.</div>

------

### 6) `nn.SyncBatchNorm.convert_sync_batchnorm()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Replaces all BatchNorm layers with cross-GPU Synchronized BatchNorm (跨GPU同步批归一化). Essential for DDP training with BN. </div>

```python
model = MyModel()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # before DDP wrap
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Must convert before wrapping with DDP</span>. Without SyncBN, each GPU computes its own BN statistics — inaccurate.</div> <div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Performance stack: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pin_memory + non_blocking</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AMP</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.compile</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gradient_checkpoint</code> (if OOM).</div>

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:20px 24px;margin-top:32px"> <span style="color:#3B5BDB;font-weight:700;font-size:16px">🎯 Master Summary — 120 APIs in 6 Core Concepts</span><br><br> <strong>1. Tensor Ops</strong>: Create (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tensor/zeros/rand</code>) → Shape (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">reshape/permute/cat</code>) → Math (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">matmul/clamp/topk</code>)<br> <strong>2. Autograd</strong>: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">requires_grad</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">backward()</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">no_grad / detach</code><br> <strong>3. Networks</strong>: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nn.Module</code> → Layers (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Linear/Conv2d/LSTM/Attention</code>) → Norm + Dropout<br> <strong>4. Training</strong>: Loss → Optimizer (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AdamW</code>) → Scheduler → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clip_grad_norm_</code><br> <strong>5. Data</strong>: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Dataset</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">transforms</code> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">DataLoader</code> → pretrained models<br> <strong>6. Deploy</strong>: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">state_dict</code> → TorchScript / ONNX → AMP / DDP / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.compile</code> </div>
