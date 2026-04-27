---
title: "AOTAutograd"
published: 2026-04-27
description: "AOTAutograd"
image: ""
tags: ["pytorch","compile","AOTAutograd"]
category: pytorch / compile
draft: false
lang: ""
createdAt: "2026-04-27T22:02:31.858.697951804Z"
---

# PyTorch AOTAutograd

AOTAutograd (Ahead-of-Time Autograd, 提前自动微分) is the middle layer (中间层) of `torch.compile` that traces both the forward graph (前向图) and backward graph (反向图) ahead of time (提前), enabling end-to-end fusion (端到端融合) across training steps.

## 1. What AOTAutograd Is

AOTAutograd is a **graph-level autograd compiler (图级自动微分编译器)** that pre-computes the backward pass (反向传播) as another FX graph (FX 图), instead of building it dynamically (动态构建) at runtime.

| Aspect                              | Eager Autograd                    | AOTAutograd                        |
| ----------------------------------- | --------------------------------- | ---------------------------------- |
| Backward construction (反向构建)    | Built during forward (前向时构建) | Traced ahead of time (提前追踪)    |
| Backward representation (反向表示)  | Tape (磁带) of grad_fn nodes      | FX Graph (FX 图)                   |
| Fusion across fwd/bwd (前后向融合)? | ❌ No                              | ✅ Yes                              |
| Saved tensors (保存张量)            | Implicit                          | Explicit graph inputs (显式图输入) |
| Used by                             | Standard PyTorch                  | `torch.compile`                    |

<br>

## 2. Position in the Compile Pipeline

AOTAutograd sits between Dynamo (Dynamo) and Inductor (Inductor), turning a forward FX graph (前向 FX 图) into joint (forward + backward) graphs (联合图) ready for backend compilation (后端编译).

```
┌────────────────┐
│  Python code   │
└───────┬────────┘
        ▼
┌────────────────┐
│   TorchDynamo  │   captures forward FX graph (捕获前向图)
└───────┬────────┘
        ▼
┌────────────────┐
│  AOTAutograd   │   adds backward graph (添加反向图)
│                │   ── runs the forward symbolically (符号执行)
│                │   ── records autograd ops (记录 autograd 操作)
│                │   ── builds backward FX graph (构建反向 FX 图)
└───────┬────────┘
        │ emits TWO graphs:
        │   • forward graph (前向图)
        │   • backward graph (反向图)
        ▼
┌────────────────┐
│   Inductor     │   compiles each graph to fused kernels (融合内核)
└────────────────┘
```

<br>

## 3. Why It Matters

By exposing the backward (反向) as an explicit graph (显式图), AOTAutograd enables backward fusion (反向融合), recomputation (重计算), and operator scheduling (算子调度) that eager autograd cannot do.

| Benefit                               | Effect                                              |
| ------------------------------------- | --------------------------------------------------- |
| Backward fusion (反向融合)            | Combine backward kernels into one                   |
| Activation recomputation (激活重计算) | Trade compute for memory (以算换存)                 |
| Joint scheduling (联合调度)           | Reorder fwd/bwd ops for memory peak (峰值内存)      |
| Operator decomposition (算子分解)     | Break complex ops into simple primitives (基础原语) |
| Functionalization (函数化)            | Remove in-place ops (移除原地操作)                  |

<br>

## 4. Tracing Forward and Backward Together

The core API `aot_function(fn, fw_compiler, bw_compiler)` traces both passes (两次传播) and hands each graph to a user-provided compiler (用户编译器).

```python
import torch
from functorch.compile import aot_function
from torch.fx import GraphModule

def fw_compiler(gm: GraphModule, example_inputs):
    print("=== Forward Graph ===")
    gm.graph.print_tabular()
    return gm.forward

def bw_compiler(gm: GraphModule, example_inputs):
    print("=== Backward Graph ===")
    gm.graph.print_tabular()
    return gm.forward

def fn(x, y):
    return (x * y).sin().sum()

compiled = aot_function(fn, fw_compiler, bw_compiler)

x = torch.randn(4, requires_grad=True)
y = torch.randn(4, requires_grad=True)
out = compiled(x, y)
out.backward()                                # triggers backward graph compile

print("x.grad shape:", x.grad.shape)
# === Forward Graph ===
# (mul, sin, sum nodes)
# === Backward Graph ===
# (cos, mul, sum_backward nodes)
# x.grad shape: torch.Size([4])
```

<br>

## 5. Joint Forward+Backward Graph

Internally, AOTAutograd first builds a **joint graph (联合图)** containing both forward and backward (前向与反向), then partitions it (划分) into separate fw/bw graphs.

```
                Joint FX Graph
   ┌──────────────────────────────────────┐
   │  inputs: x, y, grad_out              │
   │                                      │
   │  ┌────────── forward ─────────┐      │
   │  │ mul = x * y                │      │
   │  │ sin = sin(mul)             │      │
   │  │ sum = sum(sin)             │      │
   │  └────────────────────────────┘      │
   │  ┌────────── backward ────────┐      │
   │  │ cos = cos(mul)             │      │
   │  │ gx  = grad_out * cos * y   │      │
   │  │ gy  = grad_out * cos * x   │      │
   │  └────────────────────────────┘      │
   │                                      │
   │  outputs: sum, gx, gy                │
   └──────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   forward graph           backward graph
```

<br>

## 6. Min-Cut Partitioner (Activation Recomputation)

AOTAutograd's default partitioner (分区器) uses min-cut (最小割) on the joint graph (联合图) to decide which activations (激活) to save versus recompute (重计算), trading compute for memory (以算换存).

```python
import torch
from functorch.compile import aot_function, min_cut_rematerialization_partition
from torch.fx import GraphModule

def fw_compiler(gm: GraphModule, _):
    n_save = sum(1 for n in gm.graph.nodes if n.op == 'output')
    print(f"Forward outputs (saved + result): {len(list(gm.graph.nodes)[-1].args[0])}")
    return gm.forward

def bw_compiler(gm: GraphModule, _):
    return gm.forward

def fn(x):
    return x.sin().cos().sum()                 # 2 intermediate activations

compiled = aot_function(
    fn, fw_compiler, bw_compiler,
    partition_fn=min_cut_rematerialization_partition,
)

x = torch.randn(4, requires_grad=True)
compiled(x).backward()
print("x.grad shape:", x.grad.shape)
# Forward outputs (saved + result): 2     <- min_cut picks minimal save set
# x.grad shape: torch.Size([4])
```

<br>

## 7. Functionalization (Remove In-Place Ops)

AOTAutograd functionalizes (函数化) the graph (图), rewriting in-place ops (原地操作) like `add_` into pure functional ops (纯函数式操作) like `add`, simplifying compiler analysis (编译器分析).

```python
import torch

@torch.compile
def fn(x):
    y = x.clone()
    y.add_(1.0)                                # in-place op
    return y * 2

x = torch.tensor([1., 2., 3.], device='cuda')
print(fn(x))
# tensor([4., 6., 8.], device='cuda:0')
# Internally, AOTAutograd rewrites add_ -> add (pure functional form)
```

<br>

## 8. Operator Decomposition

AOTAutograd decomposes (分解) high-level ops (高层算子) like `torch.nn.functional.layer_norm` into primitive ops (原语), giving Inductor (Inductor) more fusion opportunity (融合机会).

```
    layer_norm(x, weight, bias)
                │
                ▼
       ┌────────────────────┐
       │ mean(x, dim=-1)    │
       │ var(x, dim=-1)     │
       │ rsqrt(var + eps)   │
       │ (x - mean) * rsqrt │
       │ * weight + bias    │
       └────────────────────┘
       (all simple primitives — easy to fuse)
import torch
import torch.nn.functional as F

@torch.compile
def fn(x, w, b):
    return F.layer_norm(x, (8,), w, b)

x = torch.randn(4, 8, device='cuda')
w = torch.ones(8, device='cuda')
b = torch.zeros(8, device='cuda')
print(fn(x, w, b).shape)
# torch.Size([4, 8])
# layer_norm is decomposed into mean/var/rsqrt/mul/add and fused into 1 kernel
```

<br>

## 9. Inspecting AOT Graphs in `torch.compile`

Setting `TORCH_LOGS="aot"` (环境变量) makes `torch.compile` print both the forward (前向) and backward (反向) AOT graphs (AOT 图).

```python
import os
os.environ["TORCH_LOGS"] = "aot_graphs"

import torch

@torch.compile
def fn(x, y):
    return (x * y).sin().sum()

x = torch.randn(4, requires_grad=True, device='cuda')
y = torch.randn(4, requires_grad=True, device='cuda')
fn(x, y).backward()

print("x.grad shape:", x.grad.shape)
# Stderr will show:
# === Forward graph ===
# def forward(primals_1, primals_2): ...
# === Backward graph ===
# def forward(tangents_1, mul, ...): ...
# x.grad shape: torch.Size([4])
```

<br>

## 10. Key Concepts Cheat Sheet

| Concept                            | Meaning                                    |
| ---------------------------------- | ------------------------------------------ |
| Joint graph (联合图)               | Single graph containing forward + backward |
| Partition (划分)                   | Split joint graph into fw/bw graphs        |
| Min-cut partitioner (最小割分区器) | Picks save vs recompute boundary           |
| Functionalization (函数化)         | Convert in-place ops to pure ops           |
| Decomposition (分解)               | Break complex ops into primitives          |
| Tangents (切向量)                  | Backward graph's grad_output inputs        |
| Primals (原向量)                   | Forward graph's tensor inputs              |

<br>

## 11. AOTAutograd vs Eager Autograd

The key shift (关键转变) is that AOTAutograd builds the backward statically (静态) once, while eager autograd rebuilds the backward dynamically (动态) on every iteration (每次迭代).

| Aspect                              | Eager                    | AOTAutograd                       |
| ----------------------------------- | ------------------------ | --------------------------------- |
| Backward built when (反向构建时机)? | Per iteration (每次迭代) | Once at compile time (编译期一次) |
| Memory peak (内存峰值)              | Hard to optimize         | Optimized via min-cut             |
| Backward fusion (反向融合)          | ❌                        | ✅                                 |
| Recomputation (重计算)              | Manual (`checkpoint`)    | Automatic (min-cut 自动)          |
| Overhead per step (每步开销)        | Tape construction        | Just kernel launch                |

<br>

## 12. Summary

AOTAutograd (AOTAutograd) is the **autograd-aware (autograd 感知) compiler stage** that pre-traces the backward graph (反向图) and partitions it (划分) for memory-efficient (内存高效) end-to-end (端到端) compilation in `torch.compile`.

| Component              | Role                                    |
| ---------------------- | --------------------------------------- |
| Trace (追踪)           | Symbolically run forward + backward     |
| Functionalize (函数化) | Remove in-place ops                     |
| Decompose (分解)       | Lower complex ops to primitives         |
| Partition (划分)       | Min-cut joint graph into fw/bw          |
| Compile (编译)         | Hand fw/bw graphs to backend (Inductor) |

<br> <br>
