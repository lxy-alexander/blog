---
title: "torch.compile"
published: 2026-04-27
description: "torch.compile"
image: ""
tags: ["pytorch","compile","torch.compile"]
category: pytorch / compile
draft: false
lang: ""
createdAt: "2026-04-27T21:35:39.517.246616702Z"
---

# PyTorch `torch.compile`

`torch.compile` is the JIT compiler (即时编译器) introduced in PyTorch 2.0 that traces (追踪) and optimizes (优化) Python models into fused (融合) low-level kernels (底层内核), typically delivering 1.3x-2x speedup (加速) with a single line of code.

`torch.compile(fn)` triggers TorchDynamo, which uses the CPython Frame Evaluation API to intercept(拦截) Python execution, capture PyTorch operations as an FX Graph, and pass that graph to Inductor, TensorRT, or another backend for optimization and compilation.



## 1. Basic Usage

`torch.compile(model)` returns a compiled (编译后) version of the model that runs faster (更快) on subsequent calls (后续调用).

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).cuda()

compiled_model = torch.compile(model)        # one-line speedup

x = torch.randn(32, 128, device='cuda')
out = compiled_model(x)
print(out.shape)
# torch.Size([32, 10])
```

<br>

## 2. Compile Modes

The `mode` argument selects the compilation strategy (编译策略), trading compile time (编译时间) for runtime performance (运行时性能).

| Mode                | Purpose                                                      |
| ------------------- | ------------------------------------------------------------ |
| `"default"`         | Balanced compile time and speed                              |
| `"reduce-overhead"` | Use CUDA Graphs (CUDA 图) to reduce launch overhead (启动开销) |
| `"max-autotune"`    | Aggressive autotuning (自动调优), best perf but slow compile |

```python
import torch

def fn(x):
    return x.sin().cos()

compiled = torch.compile(fn, mode="reduce-overhead")

x = torch.randn(1024, device='cuda')
y = compiled(x)
print(y.shape)
# torch.Size([1024])
```

<br>

## 3. Compile Backends

The `backend` argument picks the lowering backend (下沉后端); `"inductor"` is the default (默认), generating Triton (Triton) and C++ code (C++ 代码).

```python
import torch

def fn(x, y):
    return (x + y).relu()

compiled = torch.compile(fn, backend="inductor")     # default

a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
out = compiled(a, b)
print(out.shape, out.dtype)
# torch.Size([1024]) torch.float32
```

<br>

## 4. Decorator Style

`@torch.compile` can be used as a decorator (装饰器) directly on functions, simplifying syntax (简化语法).

```python
import torch

@torch.compile
def my_fn(x):
    return x * x + 1.0

x = torch.arange(5, dtype=torch.float32, device='cuda')
print(my_fn(x))
# tensor([ 1.,  2.,  5., 10., 17.], device='cuda:0')
```

<br>

## 5. `fullgraph=True`

`fullgraph=True` forces a single graph (单图) without graph breaks (图中断), maximizing fusion (融合) opportunities at the cost of strictness.

```python
import torch

@torch.compile(fullgraph=True)
def fn(x):
    return x.sin() + x.cos()

x = torch.randn(8, device='cuda')
out = fn(x)
print(out.shape)
# torch.Size([8])
# Note: any unsupported Python ops would raise an error instead of breaking the graph
```

<br>

## 6. `dynamic=True`

`dynamic=True` enables dynamic shapes (动态形状), avoiding recompilation (重编译) when input sizes vary (输入尺寸变化).

```python
import torch

@torch.compile(dynamic=True)
def fn(x):
    return x.sum(dim=-1)

# different shapes share the same compiled graph
print(fn(torch.randn(4, 8, device='cuda')).shape)
# torch.Size([4])
print(fn(torch.randn(16, 32, device='cuda')).shape)
# torch.Size([16])
```

<br>

## 7. How It Works (TorchDynamo + Inductor)

`torch.compile` uses TorchDynamo (动态图捕获) to capture an FX graph (FX 图), then Inductor (Inductor 后端) lowers it to fused Triton/C++ kernels (融合内核).

```python
import torch

def fn(x):
    return (x * 2 + 1).relu()

# Inspect the captured graph
explanation = torch._dynamo.explain(fn)(torch.randn(4, device='cuda'))
print("graph count:", explanation.graph_count)
print("break reasons:", len(explanation.break_reasons))
# graph count: 1
# break reasons: 0
```

<br>

## 8. Avoiding Graph Breaks

Graph breaks (图中断) split the model into multiple subgraphs (多个子图), reducing fusion (融合); avoid Python side effects (副作用) like `print` and unsupported ops.

```python
import torch

@torch.compile
def good_fn(x):
    return x.sin() + x.cos()                      # pure tensor ops -> no breaks

@torch.compile
def bad_fn(x):
    print("hello")                                 # graph break!
    return x + 1

x = torch.randn(4, device='cuda')
print(good_fn(x).shape)                            # torch.Size([4])
print(bad_fn(x).shape)                             # torch.Size([4])  (still works, just slower)
```

<br>

## 9. Inspecting Generated Code

Setting `TORCH_LOGS="output_code"` (环境变量) prints the generated Triton/C++ source (生成代码), useful for performance debugging (性能调试).

```python
import os
os.environ["TORCH_LOGS"] = "output_code"          # set BEFORE importing torch internals

import torch

@torch.compile
def fn(x):
    return x.sin().cos()

x = torch.randn(1024, device='cuda')
y = fn(x)
print(y.shape)
# torch.Size([1024])
# Stdout will also dump the generated Triton kernel source code
```

<br>

## 10. Use with `nn.Module`

Calling `torch.compile(model)` on a full `nn.Module` (完整模块) compiles the `forward` method (前向方法), supporting both training (训练) and inference (推理).

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP().cuda()
compiled = torch.compile(model)

x = torch.randn(8, 64, device='cuda')
y = compiled(x)
loss = y.sum()
loss.backward()                                    # autograd works through compile
print(y.shape, model.fc1.weight.grad.shape)
# torch.Size([8, 10]) torch.Size([128, 64])
```

<br>

## 11. Common Pitfalls

| Pitfall (陷阱)                              | Solution (解决方案)                          |
| ------------------------------------------- | -------------------------------------------- |
| Recompiles on shape change (形状变化重编译) | Use `dynamic=True`                           |
| Graph breaks (图中断) on Python ops         | Use `fullgraph=True` to detect               |
| First call slow (首次调用慢)                | Compile cost is one-time (一次性)            |
| `.item()` causes break                      | Avoid CPU sync (CPU 同步) inside compiled fn |
| Custom CUDA ops (自定义 CUDA 算子)          | Register with `torch.library`                |

<br>

## 12. Summary

`torch.compile` is a one-line drop-in (一行替换) optimizer that uses TorchDynamo (TorchDynamo) + Inductor (Inductor) to fuse PyTorch ops (PyTorch 算子) into Triton/C++ kernels — easiest way (最简单方式) to get 1.3x-2x speedup on modern GPUs.

| Feature                      | Value                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| Default backend (默认后端)   | `inductor`                                                   |
| Default mode (默认模式)      | `default`                                                    |
| Generated code (生成代码)    | Triton + C++                                                 |
| Supports training (支持训练) | Yes (autograd compatible 兼容)                               |
| Best for (适用于)            | Stable shapes (稳定形状), large models (大模型)              |
| Avoid for (不适用于)         | Heavy Python control flow (繁重控制流), shape-varying tiny ops |

<br> <br>
