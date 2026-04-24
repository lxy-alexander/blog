---
title: "PyTorch APIs"
published: 2026-04-20
description: "PyTorch APIs"
image: ""
tags: ["pytorch","PyTorch APIs"]
category: pytorch
draft: false
lang: ""
---

# I. PyTorch Overview (PyTorch 概述)

PyTorch is an end-to-end Deep Learning Framework (深度学习框架) covering data loading, model building, training, and deployment — with 1500+ APIs, mastering ~150 core ones lets you build production-grade AI systems.

## 1. API Scale Comparison (API 规模对比)

| Framework   | Total APIs | For Peak Performance | To Get Started |
| ----------- | ---------- | -------------------- | -------------- |
| Triton      | ~35        | ~35                  | ~15            |
| CuTe DSL    | ~80+       | ~60-80               | ~25            |
| **PyTorch** | **1500+**  | **~150-200**         | **~50**        |

## 2. Learning Strategy (学习策略)

PyTorch APIs are layered — learn Tier-1 first (2 weeks), then expand as projects demand; never try to memorize everything upfront.

------

# II. Tier-1: Tensor Basics (张量基础)

The Tensor (张量) is PyTorch's fundamental data structure — a multi-dimensional array with GPU acceleration and Automatic Differentiation (自动微分) support.

## 1. `torch.tensor(data)` — Create Tensor from Data (从数据创建张量)

Creates a Tensor from a Python list, NumPy array, or scalar — the most direct way to get data into PyTorch.

```python
import torch

# From list
a = torch.tensor([1.0, 2.0, 3.0])
print(a)  # tensor([1., 2., 3.])

# From nested list (2D)
b = torch.tensor([[1, 2], [3, 4]])
print(b.shape)  # torch.Size([2, 2])

# With specific dtype
c = torch.tensor([1, 2, 3], dtype=torch.float16)
print(c.dtype)  # torch.float16
```

## 2. `torch.zeros / ones / randn / rand / empty` — Factory Functions (工厂函数)

Create Tensors with specific initialization patterns — `randn` for Normal Distribution (正态分布), `rand` for Uniform (均匀分布), `zeros/ones` for constants.

```python
z = torch.zeros(3, 4)          # All zeros, shape (3, 4)
o = torch.ones(2, 3)           # All ones
r = torch.randn(2, 3)          # Normal distribution N(0,1)
u = torch.rand(2, 3)           # Uniform [0, 1)
e = torch.empty(2, 3)          # Uninitialized (fast)

# Like variants: match shape/dtype/device of existing tensor
x = torch.randn(3, 4, device='cuda')
y = torch.zeros_like(x)        # Same shape, dtype, device as x
```

## 3. `torch.arange / linspace / logspace` — Sequence Generators (序列生成器)

Generate evenly spaced values — `arange` for integer-like steps, `linspace` for fixed count between endpoints.

```python
a = torch.arange(0, 10, 2)            # tensor([0, 2, 4, 6, 8])
b = torch.linspace(0, 1, steps=5)     # tensor([0.0, 0.25, 0.5, 0.75, 1.0])
c = torch.logspace(0, 3, steps=4)     # tensor([1, 10, 100, 1000])
```

## 4. `tensor.shape / .dtype / .device` — Tensor Attributes (张量属性)

Every Tensor has three key attributes: Shape (形状), Data Type (数据类型), and Device (设备).

```python
x = torch.randn(2, 3, 4, device='cuda', dtype=torch.float32)
print(x.shape)    # torch.Size([2, 3, 4])
print(x.dtype)    # torch.float32
print(x.device)   # cuda:0
print(x.ndim)     # 3 (number of dimensions)
print(x.numel())  # 24 (total number of elements)
```

## 5. `.to(device/dtype)` / `.cuda()` / `.cpu()` — Device & Type Transfer (设备与类型转换)

Move Tensors between CPU and GPU, or convert Data Types — essential for GPU training.

```python
x = torch.randn(3, 4)
x_gpu = x.to('cuda')              # CPU -> GPU
x_cpu = x_gpu.to('cpu')           # GPU -> CPU
x_fp16 = x.to(torch.float16)      # Cast to half precision
x_gpu2 = x.cuda()                 # Shortcut for .to('cuda')

# Move model + data to same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

------

# III. Tier-1: Tensor Operations (张量运算)

## 1. Arithmetic — Element-wise (逐元素算术)

Standard math operators work element-wise on Tensors — supports Broadcasting (广播).

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)       # tensor([5., 7., 9.])
print(a * b)       # tensor([4., 10., 18.])
print(a / b)       # tensor([0.25, 0.4, 0.5])
print(a ** 2)      # tensor([1., 4., 9.])

# Broadcasting: (3,4) + (4,) works automatically
x = torch.randn(3, 4)
bias = torch.randn(4)
print((x + bias).shape)  # torch.Size([3, 4])
```

## 2. `torch.matmul / @ / mm / bmm` — Matrix Multiplication (矩阵乘法)

`@` or `torch.matmul` is the universal matrix multiply — handles batched, 2D, and vector cases.

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                  # (3,4) @ (4,5) -> (3,5)
C = torch.matmul(A, B)     # Same as above

# Batched matmul
A_batch = torch.randn(8, 3, 4)
B_batch = torch.randn(8, 4, 5)
C_batch = A_batch @ B_batch   # (8,3,4) @ (8,4,5) -> (8,3,5)

# torch.mm: strictly 2D only
# torch.bmm: strictly 3D batched only
```

## 3. `torch.sum / mean / max / min / argmax / argmin` — Reduction Operations (归约运算)

Reduce Tensors along specified Dimensions (维度) — `keepdim=True` preserves the reduced dimension.

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(torch.sum(x))            # tensor(21.) — global sum
print(torch.sum(x, dim=1))     # tensor([6., 15.]) — row sums
print(torch.mean(x, dim=0))    # tensor([2.5, 3.5, 4.5]) — column means
print(torch.max(x, dim=1))     # values=tensor([3., 6.]), indices=tensor([2, 2])
print(torch.argmax(x, dim=1))  # tensor([2, 2]) — index of max per row

# keepdim preserves shape for broadcasting
row_max = x.max(dim=1, keepdim=True).values   # shape: (2, 1)
normalized = x - row_max                      # Broadcasting works
```

## 4. `torch.cat / stack / split / chunk` — Concatenation & Splitting (拼接与分割)

-   **torch.cat(tensors, dim=0, *, out=None) -> Tensor**

-   **torch.stack(tensors, dim=0, *, out=None) -> Tensor**
-   torch.split(tensor, split_size_or_sections, dim=0) -> List[Tensor]
-   torch.chunk(input, chunks, dim=0) -> List[Tensor]

Combine or split Tensors along existing or new dimensions.

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# cat: join tensors along existing dim
c = torch.cat([a, b], dim=0)      # (4, 3) — stack vertically

d = torch.cat([a, b], dim=1)      # (2, 6) — stack horizontally

# stack: join tensors along NEW dim
s = torch.stack([a, b], dim=0)    # (2, 2, 3) — new batch dim
x = torch.stack([a, b], dim=1)

# split/chunk: reverse of cat
parts = torch.split(c, 2, dim=0)  # Split into chunks of size 2
chunks = torch.chunk(c, 2, dim=0) # Split into 2 equal chunks
```

## 5. `view / reshape / permute / transpose / squeeze / unsqueeze` — Shape Manipulation (形状变换)

Reshape Tensors without copying data — `view` requires contiguous memory, `reshape` handles any case.

```python
x = torch.randn(2, 3, 4)

# reshape / view: change this tensor's shape (do not change total elements)
# view requires the contiguous memory
y = x.view(6, 4)              # (2,3,4) -> (6,4)
y = x.reshape(2, 12)          # (2,3,4) -> (2,12)
y = x.view(-1)                # Flatten: (24,)

# permute: reorder dimensions, only change the stride not the data
y = x.permute(2, 0, 1)        # (2,3,4) -> (4,2,3)

# transpose: swap two dims
y = x.transpose(0, 2)         # (2,3,4) -> (4,3,2)

# squeeze / unsqueeze: remove / add dim of size 1
y = x.unsqueeze(0)             # (2,3,4) -> (1,2,3,4)
y = y.squeeze(0)               # (1,2,3,4) -> (2,3,4)
```

```python
torch.Tensor.view()
Returns a new tensor with the same data but of a different shape.
The returned tensor shares the same data and must have the same number of elements.
Requires contiguous memory. Throws RuntimeError if not contiguous.

torch.reshape()
Returns a tensor with the same data and number of elements, but with the specified shape.
When possible, the returned tensor will be a view of input. Otherwise, it will be a copy.
Do not depend on the copying vs. viewing behavior.

torch.as_strided()
Create a view of an existing tensor with specified size, stride and storage_offset.
⚠️ More than one element may refer to a single memory location.
In-place operations may result in incorrect behavior. Clone first if you need to write.

torch.Tensor.transpose() / torch.transpose()
Returns a tensor that is a transposed version of input.
The given dimensions dim0 and dim1 are swapped.
The resulting tensor shares its underlying storage with the input tensor.

torch.Tensor.permute() / torch.permute()
Returns a view of the original tensor with its dimensions permuted.

torch.Tensor.squeeze() / torch.squeeze()
Returns a tensor with all specified dimensions of size 1 removed.
The returned tensor shares the storage with the input tensor.

torch.Tensor.unsqueeze() / torch.unsqueeze()
Returns a new tensor with a dimension of size one inserted at the specified position.
The returned tensor shares the same underlying data with this tensor.

torch.Tensor.expand()
Returns a new view of the tensor with singleton dimensions expanded to a larger size.
⚠️ More than one element of an expanded tensor may refer to a single memory location.

torch.narrow()
Returns a new tensor that is a narrowed version of self tensor.
The dimension dim is input from start to start + length.
The returned tensor and self tensor share the same underlying storage.

torch.Tensor.select()
Slices the self tensor along the selected dimension at the given index.
This function returns a view of the original tensor with the given dimension removed.

torch.Tensor.diagonal() / torch.diagonal()
Returns a partial view of input with the its diagonal elements with respect to dim1 and dim2
appended as a dimension at the end of the shape.

torch.flatten()
Flattens input by reshaping it into a one-dimensional tensor.
If input is already one-dimensional, it is returned as is.
Can return either a view or a copy. Do not depend on either behavior.

torch.chunk()
Attempts to split a tensor into the specified number of chunks along the given dimension.
Each chunk is a view of the input tensor.

torch.split()
Splits the tensor into chunks. Each chunk is a view of the original tensor.

torch.unbind()
Removes a tensor dimension. Returns a tuple of all slices along a given dimension,
already without it. Each slice is a view of the input tensor.

torch.Tensor.contiguous()
Returns a contiguous in memory tensor containing the same data as self tensor.
If self tensor is already in the specified memory format, this function returns self tensor.
Otherwise, it returns a copy.

torch.Tensor.clone()
Returns a copy of the self tensor. Unlike copy_(), this function is recorded in the
computation graph. Gradients propagating to the cloned tensor will propagate to the
original tensor.

torch.repeat() / torch.Tensor.repeat()
Repeats this tensor along the specified dimensions.
Unlike expand(), this function copies the tensor's data.
```

```python
═══════════════════════════════════════════════════════
PYTORCH TENSOR VIEW / RESHAPE / MEMORY — WHEN TO USE WHAT
═══════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. view() vs reshape() — 最常见的选择
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 正常情况：两者一样
x = torch.randn(4, 6)
x.view(2, 12)     # ✓
x.reshape(2, 12)  # ✓ 结果相同

# 坑：transpose之后用view
x = torch.randn(4, 6)
y = x.transpose(0, 1)   # y是非连续的
y.view(24)               # ❌ RuntimeError: tensor is not contiguous
y.reshape(24)            # ✓ 但会悄悄拷贝一份数据！
y.contiguous().view(24)  # ✓ 显式拷贝，意图清晰

# 结论：
# - 确定内存连续 → view()，性能最好，出错立刻报错
# - 不确定        → reshape()，但注意它可能悄悄拷贝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. expand() vs repeat() — 扩展维度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bias = torch.randn(1, 8)

# expand：stride=0，不拷贝，只是"假装"有32份
a = bias.expand(32, 8)   # ✓ 零拷贝，shape=(32,8)
                          # 底层还是1×8的数据

# repeat：真实拷贝32份
b = bias.repeat(32, 1)   # shape=(32,8)，真实32×8的数据

# 坑：对expand结果做in-place操作
a = bias.expand(32, 8)
a[0, 0] = 999   # ❌ 或得到意外结果
                 # 因为所有行共享同一内存，改一行全变

# 结论：
# - 只读场景（广播计算）→ expand()，省显存
# - 需要独立修改每份   → repeat()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. transpose() / permute() — 维度交换
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x = torch.randn(2, 3, 4)

# transpose：只能交换两个维度
x.transpose(0, 2)   # (4, 3, 2)

# permute：任意重排所有维度
x.permute(2, 0, 1)  # (4, 2, 3)

# 坑：transpose/permute后直接view
x = torch.randn(2, 3, 4)
y = x.permute(2, 0, 1)  # (4, 2, 3)，非连续
y.view(-1)               # ❌ RuntimeError
y.contiguous().view(-1)  # ✓

# 典型使用场景：NCHW → NHWC
img = torch.randn(8, 3, 224, 224)   # NCHW
img = img.permute(0, 2, 3, 1)       # NHWC (8, 224, 224, 3)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. chunk() / split() / unbind() — 切分
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x = torch.randn(12, 4)

# chunk：尽量等分，最后一块可能小一点
parts = x.chunk(5, dim=0)
# → (3,4), (3,4), (3,4), (3,4) 只有4块，因为12/5向上取整=3

# split：精确指定每块大小
a, b = x.split([8, 4], dim=0)   # (8,4) 和 (4,4)

# unbind：完全拆开，去掉该维度
rows = x.unbind(dim=0)   # 12个 (4,) tensor

# 坑：chunk不一定给你想要的块数
x = torch.randn(10, 4)
parts = x.chunk(3, dim=0)
len(parts)   # 3，但大小是 (4,4), (4,4), (2,4)，不等分！

# 坑：这些都是视图，修改会影响原tensor
parts = x.chunk(3, dim=0)
parts[0][0, 0] = 999   # x也被修改了！
# 需要独立副本时：[p.clone() for p in parts]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. squeeze() / unsqueeze() — 维度增删
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x = torch.randn(3, 1, 4)

x.squeeze(1)      # (3, 4)，删掉size=1的维度
x.squeeze()       # 删掉所有size=1的维度，小心！

x.unsqueeze(0)    # (1, 3, 1, 4)
x.unsqueeze(-1)   # (3, 1, 4, 1)

# 坑：squeeze()不加dim，batch_size=1时出事
x = torch.randn(1, 8)   # batch=1
x.squeeze()              # (8,) ← batch维度也没了！
x.squeeze(1)             # (1, 8) ← 只删dim=1，安全


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. clone() vs contiguous() — 什么时候真正需要拷贝
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x = torch.randn(4, 6)
y = x.transpose(0, 1)

# contiguous：仅为了让内存连续，梯度正常传播
z = y.contiguous()        # 新内存，但grad_fn保留

# clone：完整独立副本，梯度也能传播
z = y.clone()             # 新内存，新tensor，grad_fn保留

# detach：断开梯度，不拷贝数据
z = y.detach()            # 同一内存，不参与反向传播

# 坑：只想要数值不想要梯度，忘记detach
z = y  # z和y共享梯度图，修改z会影响backward
z = y.detach().clone()  # ✓ 安全的独立副本，无梯度


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总结：选择决策树
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

改形状？
  ├─ 确定连续                    → view()
  └─ 不确定                      → reshape()

扩展维度？
  ├─ 只读/广播                   → expand()
  └─ 需要独立修改                → repeat()

交换维度？
  ├─ 两个维度                    → transpose()
  └─ 多个维度任意重排             → permute()
     └─ 之后要view？             → 先 .contiguous()

切分tensor？
  ├─ 等分N块                     → chunk()
  ├─ 指定每块大小                → split()
  └─ 完全拆开                    → unbind()
     └─ 需要独立修改？           → .clone() 每一块

需要拷贝？
  ├─ 仅为了连续性                → contiguous()
  ├─ 完整独立副本（有梯度）      → clone()
  └─ 完整独立副本（无梯度）      → detach().clone()
```







## 6. Indexing & Slicing (索引与切片)

PyTorch supports NumPy-style Advanced Indexing (高级索引) — including boolean masks and fancy indexing.

```python
x = torch.randn(4, 5)

# Basic slicing
row0 = x[0]            # First row
col2 = x[:, 2]         # Third column
block = x[1:3, 2:4]    # Sub-matrix

# Boolean masking
mask = x > 0
positives = x[mask]    # All positive elements (1D)

# Fancy indexing
indices = torch.tensor([0, 2, 3])
selected = x[indices]  # Select rows 0, 2, 3

# gather: advanced index-based selection
idx = torch.tensor([[0, 1], [2, 3]])
gathered = torch.gather(x, dim=1, index=idx)
```

------

# IV. Tier-1: Autograd — Automatic Differentiation (自动微分)

==Autograd (自动微分) is PyTorch's engine for computing Gradients (梯度) — it records operations on Tensors and auto-computes backward passes.==

## 1. `requires_grad=True` — Enable Gradient Tracking (启用梯度追踪)

Tells PyTorch to track all operations on this Tensor for backward Differentiation (微分).

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
z = y.sum()

z.backward()          # Compute gradients
print(x.grad)         # tensor([7., 9.])  — dz/dx = 2x + 3
```

## 2. `.backward()` — Compute Gradients (计算梯度)

Runs Backpropagation (反向传播) from the scalar output, filling `.grad` attributes of all leaf Tensors.

```python
model_params = torch.randn(3, requires_grad=True)
loss = (model_params ** 2).sum()
loss.backward()
print(model_params.grad)  # 2 * model_params
```

## 3. `torch.no_grad()` — Disable Gradient Context (禁用梯度上下文)

Temporarily disables gradient computation — used during Inference (推理) to save memory and speed up computation.

```python
model = torch.nn.Linear(10, 5)
x = torch.randn(3, 10)

with torch.no_grad():
    output = model(x)    # No gradient tracking, faster
    print(output.requires_grad)  # False

# Also useful: torch.inference_mode() — even faster
with torch.inference_mode():
    output = model(x)
```

## 4. `.detach()` — Detach from Computation Graph (从计算图分离)

Creates a new Tensor sharing the same data but detached from the Computation Graph (计算图) — stops gradient flow.

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.detach()    # z shares data with y but has no grad_fn
print(z.requires_grad)  # False
```

## 5. `.grad.zero_()` / `optimizer.zero_grad()` — Zero Gradients (梯度清零)

Gradients accumulate by default — you must zero them before each backward pass.

```python
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    loss = x ** 2
    loss.backward()
    print(f"Step {i}: grad = {x.grad}")  # Accumulates!
    x.grad.zero_()  # Reset for next iteration
```

------

# V. Tier-1: `nn.Module` — Model Building (模型构建)

==`nn.Module` is PyTorch's base class for all neural network components== — every model is a Module containing other Modules.

## 1. `nn.Module` — Base Class (基类)

Subclass `nn.Module`, define layers in `__init__`, define forward logic in `forward()`.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(784, 256, 10)
print(model)
```

## 2. `nn.Linear(in, out)` — Fully Connected Layer (全连接层)

Applies a Linear Transformation (线性变换): $y = xW^T + b$.

-   `x`: (128,)
-   `W`: (64, 128) **64 表示“输出特征数”（out_features）**。
-   `b`: (64,)
-   `y`: (64,)

```python
layer = nn.Linear(128, 64)
x = torch.randn(32, 128)      # Batch of 32, 128-dim input
y = layer(x)                   # (32, 64)
print(layer.weight.shape)      # (64, 128)
print(layer.bias.shape)        # (64,)
```

## 3. `nn.Conv2d(in_ch, out_ch, kernel_size)` — 2D Convolution (二维卷积)

Applies a 2D Convolution (卷积) — the building block of CNNs (卷积神经网络).

```python
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
x = torch.randn(8, 3, 224, 224)   # (batch, channels, H, W)
y = conv(x)                        # (8, 64, 224, 224)
```

## 4. Activation Functions (激活函数)

Non-linear functions applied after linear layers — `ReLU` is the most common.

```python
relu = nn.ReLU()
gelu = nn.GELU()
silu = nn.SiLU()           # Also called Swish (Sigmoid Linear Unit)
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

# Functional API (no state, use in forward())
import torch.nn.functional as F
y = F.relu(x)
y = F.gelu(x)
y = F.softmax(x, dim=-1)
```

## 5. `nn.Embedding(num, dim)` — Embedding Layer (嵌入层)

Lookup table mapping integer indices to dense Vectors (向量) — used for words, tokens, categorical features.

```python
embed = nn.Embedding(num_embeddings=10000, embedding_dim=256)
token_ids = torch.tensor([1, 42, 999, 7])
vectors = embed(token_ids)     # (4, 256)
```

## 6. Normalization Layers (归一化层)

Normalize activations for stable training — `LayerNorm` for Transformers, `BatchNorm` for CNNs.

```python
# LayerNorm: normalize across features (used in Transformers)
ln = nn.LayerNorm(256)
x = torch.randn(32, 10, 256)  # (batch, seq, features)
y = ln(x)                      # Same shape, normalized

# BatchNorm: normalize across batch (used in CNNs)
bn = nn.BatchNorm2d(64)
x = torch.randn(8, 64, 32, 32)  # (batch, channels, H, W)
y = bn(x)
```

## 7. `nn.Dropout(p)` — Dropout Regularization (随机失活正则化)

Randomly zeros elements with Probability (概率) `p` during training — prevents Overfitting (过拟合).

```python
dropout = nn.Dropout(p=0.1)
x = torch.randn(32, 256)
y = dropout(x)  # ~10% of elements zeroed during training
```

## 8. `nn.Sequential(...)` — Sequential Container (顺序容器)

Chains modules sequentially — input flows through each in order.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
y = model(torch.randn(32, 784))  # (32, 10)
```

## 9. `model.parameters()` / `model.named_parameters()` — Access Parameters (访问参数)

Iterates over all learnable Parameters (参数) — passed to the Optimizer (优化器).

```python
model = MLP(784, 256, 10)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## 10. `model.train()` / `model.eval()` — Training vs Eval Mode (训练与评估模式)

Switches the model between Training Mode (训练模式, dropout active, batchnorm uses batch stats) and Eval Mode (评估模式, dropout off, batchnorm uses running stats).

```python
model.train()   # Enable dropout, update batchnorm stats
# ... training loop ...

model.eval()    # Disable dropout, use saved batchnorm stats
with torch.no_grad():
    predictions = model(test_input)
```

------

# VI. Tier-1: Training Loop (训练循环)

## 1. Loss Functions (损失函数)

Measure the difference between predictions and targets — the value being minimized during training.

```python
# Classification
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)      # (batch, num_classes)
targets = torch.randint(0, 10, (32,))
loss = ce_loss(logits, targets)

# Regression
mse_loss = nn.MSELoss()
pred = torch.randn(32, 1)
target = torch.randn(32, 1)
loss = mse_loss(pred, target)

# Other common losses
bce = nn.BCEWithLogitsLoss()       # Binary classification
l1 = nn.L1Loss()                   # Mean absolute error
huber = nn.HuberLoss()             # Smooth L1
```

## 2. Optimizers (优化器)

Update model Parameters (参数) based on computed Gradients (梯度).

```python
import torch.optim as optim

# Adam: most popular, good default
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW: Adam with decoupled weight decay (preferred for Transformers)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## 3. Learning Rate Schedulers (学习率调度器)

Adjust the Learning Rate (学习率) during training — warm up, decay, or cycle.

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=1000)

# Use in training loop
for epoch in range(100):
    train_one_epoch()
    scheduler.step()
```

## 4. Complete Training Loop (完整训练循环)

The canonical PyTorch training pattern — **memorize this skeleton**.

```python
model = MLP(784, 256, 10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Backward pass
        optimizer.zero_grad()    # 1. Zero gradients
        loss.backward()          # 2. Compute gradients
        optimizer.step()         # 3. Update parameters

    # Evaluation
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_logits = model(val_x.to(device))
```

------

# VII. Tier-1: Data Loading (数据加载)

## 1. `Dataset` — Define Your Data (定义数据集)

Subclass `Dataset` with `__len__` and `__getitem__` to wrap any data source.

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
```

## 2. `DataLoader` — Batching & Shuffling (批处理与打乱)

Wraps a Dataset with batching, shuffling, multi-process loading, and prefetching.

```python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,         # Shuffle each epoch
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster CPU->GPU transfer
    drop_last=True,       # Drop incomplete last batch
)

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)  # (64, 784)
    # ... training step ...
```

------

# VIII. Tier-1: Save & Load (保存与加载)

## 1. `torch.save / torch.load` — Checkpoint (检查点)

Save and load model weights, optimizer state, or any Python object.

```python
# Save model weights only
torch.save(model.state_dict(), 'model.pth')

# Load model weights
model = MLP(784, 256, 10)
model.load_state_dict(torch.load('model.pth', weights_only=True))

# Save full checkpoint (model + optimizer + epoch)
checkpoint = {
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'loss': loss.item(),
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume training
ckpt = torch.load('checkpoint.pth', weights_only=False)
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
```

------

# IX. Tier-2: Transformer Components (Transformer 组件)

## 1. `nn.MultiheadAttention` — Multi-Head Attention (多头注意力)

The core attention mechanism of Transformers — computes Scaled Dot-Product Attention (缩放点积注意力).

```python
mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
x = torch.randn(32, 50, 256)      # (batch, seq_len, embed_dim)
attn_out, attn_weights = mha(x, x, x)  # Self-attention: Q=K=V=x
print(attn_out.shape)              # (32, 50, 256)
```

## 2. `nn.Transformer` — Full Transformer (完整 Transformer)

Pre-built Encoder-Decoder Transformer — useful for seq2seq tasks.

```python
transformer = nn.Transformer(
    d_model=256, nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    batch_first=True,
)
src = torch.randn(32, 50, 256)    # Source sequence
tgt = torch.randn(32, 30, 256)    # Target sequence
out = transformer(src, tgt)        # (32, 30, 256)
```

## 3. `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` — Encoder Only (仅编码器)

For BERT-style models — Self-Attention + FFN with residual connections.

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256, nhead=8, dim_feedforward=1024,
    dropout=0.1, batch_first=True, norm_first=True,  # Pre-norm
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
x = torch.randn(32, 128, 256)
out = encoder(x)  # (32, 128, 256)
```

## 4. `F.scaled_dot_product_attention` — FlashAttention Kernel (闪存注意力核函数)

PyTorch's built-in efficient attention — auto-selects FlashAttention, Memory-Efficient, or Math backend.

```python
q = torch.randn(32, 8, 128, 64, device='cuda')  # (B, H, S, D)
k = torch.randn(32, 8, 128, 64, device='cuda')
v = torch.randn(32, 8, 128, 64, device='cuda')

# Automatically uses FlashAttention when available
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape)  # (32, 8, 128, 64)
```

------

# X. Tier-2: Performance Optimization (性能优化)

## 1. `torch.compile(model)` — Graph Compilation (图编译)

The single most impactful optimization — compiles your model into optimized fused kernels via TorchDynamo + TorchInductor, often giving 1.5-2x speedup.

```python
model = MLP(784, 256, 10).to('cuda')

# One-line speedup
compiled_model = torch.compile(model)

# With options
compiled_model = torch.compile(model, mode='reduce-overhead')  # Low latency
compiled_model = torch.compile(model, mode='max-autotune')     # Max throughput

# Use exactly like original model
output = compiled_model(input_tensor)
```

## 2. `torch.amp` — Automatic Mixed Precision (自动混合精度)

Train in FP16/BF16 for ~2x speedup on modern GPUs while maintaining FP32 Accuracy (精度).

```python
scaler = torch.amp.GradScaler('cuda')

for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # Forward pass in mixed precision
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

    # Backward pass with gradient scaling
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 3. `torch.nn.utils.clip_grad_norm_` — Gradient Clipping (梯度裁剪)

Clips gradient norms to prevent Gradient Explosion (梯度爆炸) — essential for Transformer training.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 4. `torch.backends.cudnn.benchmark` — cuDNN Autotuning (cuDNN 自动调优)

Auto-selects fastest convolution algorithm for fixed input sizes — set once at the beginning.

```python
torch.backends.cudnn.benchmark = True  # Enable for fixed-size inputs
```

## 5. `torch.set_float32_matmul_precision` — TensorFloat-32 (张量浮点32)

Controls whether matrix multiplications use TF32 on Ampere+ GPUs — `'high'` enables it for ~3x GEMM speedup.

```python
torch.set_float32_matmul_precision('high')  # Enable TF32
```

------

# XI. Tier-2: Recurrent & Sequence Layers (循环与序列层)

## 1. `nn.LSTM / nn.GRU` — Recurrent Layers (循环层)

Process Sequential Data (序列数据) with hidden state — `LSTM` for long-range, `GRU` for simpler alternative.

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, bidirectional=True)
x = torch.randn(32, 50, 128)       # (batch, seq_len, features)
output, (h_n, c_n) = lstm(x)
print(output.shape)                 # (32, 50, 512) — bidirectional doubles hidden
```

## 2. `nn.RNN` — Basic Recurrent Layer (基础循环层)

Simplest recurrent unit — rarely used in practice due to Vanishing Gradients (梯度消失).

------

# XII. Tier-2: Computer Vision Layers (计算机视觉层)

## 1. Pooling Layers (池化层)

Downsample feature maps — reduce spatial dimensions while retaining important features.

```python
# Max pooling
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 64, 32, 32)
y = pool(x)                        # (8, 64, 16, 16)

# Adaptive: output size is fixed regardless of input
adapt_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
y = adapt_pool(x)                  # (8, 64, 1, 1) — global average pooling
```

## 2. `nn.ConvTranspose2d` — Transposed Convolution (转置卷积)

Upsamples feature maps — used in Decoder (解码器) and Generative Models (生成模型).

```python
up_conv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
x = torch.randn(8, 64, 16, 16)
y = up_conv(x)                     # (8, 32, 32, 32)
```

------

# XIII. Tier-2: Utilities (工具函数)

## 1. `torch.einsum(equation, *tensors)` — Einstein Summation (爱因斯坦求和)

Expresses complex tensor operations in a single equation — batched matmul, contractions, traces, etc.

```python
# Batched matrix multiply
A = torch.randn(8, 3, 4)
B = torch.randn(8, 4, 5)
C = torch.einsum('bik,bkj->bij', A, B)  # (8, 3, 5)

# Attention: (B,H,S,D) x (B,H,D,S) -> (B,H,S,S)
q = torch.randn(2, 8, 10, 64)
k = torch.randn(2, 8, 10, 64)
attn = torch.einsum('bhsd,bhtd->bhst', q, k)
```

## 2. `torch.where(condition, x, y)` — Conditional Select (条件选择)

Element-wise selection based on condition — like `np.where`.

```python
x = torch.randn(5)
y = torch.zeros(5)
result = torch.where(x > 0, x, y)   # Keep positives, zero negatives
```

## 3. `torch.clamp(input, min, max)` — Value Clamping (值截断)

Clamps all elements to `[min, max]` range.

```python
x = torch.randn(5)
clamped = torch.clamp(x, min=-1.0, max=1.0)
```

## 4. `torch.topk(input, k)` — Top-K Selection (前 K 选择)

Returns the `k` largest elements and their indices — used in beam search and top-k sampling.

```python
logits = torch.randn(32, 50000)   # Vocabulary logits
values, indices = torch.topk(logits, k=50, dim=-1)  # Top 50 tokens
```

## 5. `torch.nn.functional.one_hot` — One-Hot Encoding (独热编码)

Converts class indices to One-Hot Vectors (独热向量).

```python
labels = torch.tensor([0, 2, 1, 4])
one_hot = F.one_hot(labels, num_classes=5)  # (4, 5)
```

## 6. Gradient Checkpointing (梯度检查点)

Trades compute for memory — recomputes intermediate activations during backward instead of storing them.

```python
from torch.utils.checkpoint import checkpoint

class BigModel(nn.Module):
    def forward(self, x):
        # Recompute layer1's output during backward to save memory
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = self.layer2(x)
        return x
```

------

# XIV. Tier-3: Distributed Training (分布式训练)

## 1. `DistributedDataParallel (DDP)` — Multi-GPU Training (多 GPU 训练)

Replicates model on each GPU, synchronizes gradients via All-Reduce (全规约) — the standard for multi-GPU training.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Training loop is the same, just use DDP-wrapped model
# Launch: torchrun --nproc_per_node=4 train.py
```

## 2. `FullyShardedDataParallel (FSDP)` — Memory-Efficient Distributed (内存高效分布式)

Shards model parameters across GPUs — enables training models larger than single GPU memory.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(MyModel().to(device))
```

## 3. `DeviceMesh` / `DTensor` — Advanced Parallelism (高级并行)

For Tensor Parallelism (张量并行), Pipeline Parallelism (流水线并行), and mixed strategies.

------

# XV. Tier-3: Deployment & Export (部署与导出)

## 1. `torch.export.export` — Export for Deployment (导出部署)

Captures the full model graph for ahead-of-time compilation and deployment.

```python
from torch.export import export

exported = export(model, (sample_input,))
```

## 2. `torch.onnx.export` — ONNX Export (ONNX 导出)

Export model to ONNX format for cross-platform Inference (推理) — runs on TensorRT, ONNX Runtime, etc.

```python
torch.onnx.export(model, sample_input, "model.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}})
```

## 3. `torch.jit.script / torch.jit.trace` — TorchScript (TorchScript 脚本化)

Converts Python model to a serializable, optimizable IR — legacy approach, prefer `torch.export` for new code.

------

# XVI. Tier-3: Quantization & Efficiency (量化与效率)

## 1. `torchao` — Advanced Quantization (高级量化)

PyTorch's first-class quantization library — supports INT8, INT4, and mixed-precision for efficient Inference (推理).

```python
import torchao
# Quantize model to INT8 for 2x inference speedup
quantized_model = torchao.quantize(model, 'int8_weight_only')
```

## 2. `torch.quantization` — Legacy Quantization API (传统量化 API)

Older quantization API — supports dynamic, static, and quantization-aware training.

------

# XVII. API Combination Patterns (API 组合模式)

## 1. Standard Training Pattern (标准训练模式)

**APIs used together:** `nn.Module` + `DataLoader` + `Loss` + `Optimizer` + `.backward()` + `.step()` + `torch.compile`

This is the pattern you'll write 90% of the time.

```python
# Setup
model = torch.compile(MyModel().to(device))
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Loop
for epoch in range(100):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
```

## 2. Mixed Precision Training Pattern (混合精度训练模式)

**APIs used together:** Pattern 1 + `torch.amp.autocast` + `GradScaler`

```python
scaler = torch.amp.GradScaler('cuda')
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

## 3. Transformer Block Pattern (Transformer 块模式)

**APIs used together:** `nn.MultiheadAttention` + `nn.LayerNorm` + `nn.Linear` + `nn.GELU` + `nn.Dropout` + `F.scaled_dot_product_attention`

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x
```

## 4. Evaluation / Inference Pattern (评估/推理模式)

**APIs used together:** `model.eval()` + `torch.no_grad()` / `torch.inference_mode()` + `torch.compile`

```python
model.eval()
compiled_model = torch.compile(model, mode='reduce-overhead')

with torch.inference_mode():
    for x, y in test_loader:
        x = x.to(device)
        preds = compiled_model(x)
        accuracy = (preds.argmax(dim=1) == y.to(device)).float().mean()
```

## 5. Custom Autograd Function Pattern (自定义自动微分函数模式)

**APIs used together:** `torch.autograd.Function` + `.apply()` + `ctx.save_for_backward`

```python
class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
output = CustomReLU.apply(input_tensor)
```

------

# XVIII. API Combination Quick Reference (API 组合速查表)

## 1. Which APIs Go Together? (哪些 API 一起使用)

| Pattern (模式)                 | Core APIs                                                    | Add for Performance              |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| **Basic Training (基础训练)**  | `nn.Module`, `DataLoader`, `Loss`, `Optimizer`, `.backward()`, `.step()` | `torch.compile`, `pin_memory`    |
| **Mixed Precision (混合精度)** | + `torch.amp.autocast`, `GradScaler`                         | `bfloat16` on Ampere+            |
| **Transformer (变换器)**       | + `MultiheadAttention`, `LayerNorm`, `GELU`, `Embedding`     | `F.scaled_dot_product_attention` |
| **CNN (卷积网络)**             | + `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `AdaptiveAvgPool2d`  | `cudnn.benchmark`                |
| **Multi-GPU (多 GPU)**         | + `DDP`, `DistributedSampler`, `init_process_group`          | `FSDP` for large models          |
| **Inference (推理)**           | `model.eval()`, `torch.inference_mode()`, `torch.compile`    | `torchao` quantization           |
| **Checkpoint (检查点)**        | `torch.save`, `torch.load`, `state_dict()`                   | `gradient_checkpointing`         |

## 2. Performance Stack — Apply in Order (性能堆栈 — 按顺序应用)

| Priority | Technique (技术)    | API                                    | Typical Speedup |
| -------- | ------------------- | -------------------------------------- | --------------- |
| 1        | GPU transfer        | `.to('cuda')`                          | 10-100x         |
| 2        | Graph compilation   | `torch.compile`                        | 1.5-2x          |
| 3        | Mixed precision     | `torch.amp.autocast`                   | 1.5-2x          |
| 4        | Efficient attention | `F.scaled_dot_product_attention`       | 2-4x            |
| 5        | TF32 matmul         | `set_float32_matmul_precision('high')` | 1.3x            |
| 6        | Gradient clipping   | `clip_grad_norm_`                      | Stability       |
| 7        | Data loading        | `num_workers`, `pin_memory`            | 1.2-1.5x        |

------

# XIX. Interview Quick Answers (面试速答)

## 1. Rapid-Fire Q&A

**Q: `torch.compile` in one sentence?** It traces your model into a graph and fuses operations via TorchInductor, often giving 1.5-2x speedup with a single line of code.

**Q: `view` vs `reshape`?** `view` requires contiguous memory and shares storage; `reshape` works on any tensor but may copy data if non-contiguous.

**Q: `model.train()` vs `model.eval()`?** `train()` enables Dropout and uses batch statistics for BatchNorm; `eval()` disables Dropout and uses running statistics — always set before forward pass.

**Q: `torch.no_grad()` vs `torch.inference_mode()`?** Both disable gradient computation; `inference_mode` is stricter and faster because it also disables autograd dispatch and version counting.

**Q: Why `optimizer.zero_grad()` before `backward()`?** PyTorch accumulates gradients by default (useful for gradient accumulation); without zeroing, gradients from previous steps add up incorrectly.

**Q: `nn.CrossEntropyLoss` expects what input?** Raw logits (unnormalized scores), NOT softmax probabilities — it internally applies `log_softmax + NLLLoss`.

**Q: DDP vs FSDP?** DDP replicates the full model on each GPU and syncs gradients; FSDP shards parameters across GPUs so each GPU only stores a fraction, enabling larger models.

**Q: How to speed up training with one line?** `model = torch.compile(model)` — this alone often gives the biggest single improvement.

**Q: What's FlexAttention?** PyTorch 2.5+ API that lets you define custom attention masks in pure Python (causal, sliding window, etc.) and auto-compiles them into fused FlashAttention kernels.

**Q: `state_dict()` vs saving the whole model?** Always save `state_dict()` (just the parameters) — saving the whole model with `pickle` is fragile and breaks when code changes.
