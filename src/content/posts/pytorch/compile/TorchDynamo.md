---
title: "TorchDynamo"
published: 2026-04-27
description: "TorchDynamo"
image: ""
tags: ["pytorch","compile","TorchDynamo"]
category: pytorch / compile
draft: false
lang: ""
createdAt: "2026-04-27T21:36:42.050.478775347Z"
---

# PyTorch TorchDynamo

TorchDynamo (TorchDynamo) is the Python-level graph capture (图捕获) frontend (前端) of `torch.compile`, which hooks into CPython bytecode (字节码) to extract FX graphs (FX 图) from arbitrary Python code without rewriting it.

## 1. What TorchDynamo Is

TorchDynamo is a **bytecode-level tracer (字节码级追踪器)** that intercepts (拦截) Python frame execution (Python 帧执行) and rewrites it into compilable FX graphs (可编译 FX 图).

| Aspect                    | Description                               |
| ------------------------- | ----------------------------------------- |
| Layer (层级)              | Python frontend (Python 前端)             |
| Captures (捕获)           | FX Graph (FX 图)                          |
| Mechanism (机制)          | CPython frame evaluation API (帧求值 API) |
| Backend output (后端输出) | Inductor / TensorRT / etc.                |
| Triggered by (触发)       | `torch.compile(fn)`                       |

<br>

## 2. How It Works

TorchDynamo uses CPython's frame evaluation hook (帧求值钩子, PEP 523) to intercept each Python function call (函数调用) and trace tensor operations (张量操作) into an FX graph (FX 图).

```
Python source
     ↓ (CPython compiles to bytecode 字节码)
Bytecode
     ↓ (Dynamo intercepts via PEP 523 frame eval hook)
FX Graph + Guards (图 + 守卫)
     ↓ (passed to backend, e.g. Inductor)
Optimized kernels (优化内核)
```

<br>

## 3. Inspecting Captured Graphs

`torch._dynamo.explain(fn)(args)` reveals (揭示) how many graphs (图数量) Dynamo captured and where graph breaks (图中断) occurred.

```python
import torch

def fn(x, y):
    return (x + y).relu().sum()

explanation = torch._dynamo.explain(fn)(
    torch.randn(4, device='cuda'),
    torch.randn(4, device='cuda'),
)
print("graph count :", explanation.graph_count)
print("op count    :", explanation.op_count)
print("break count :", len(explanation.break_reasons))
# graph count : 1
# op count    : 3
# break count : 0
```

<br>

## 4. Graph Breaks

A graph break (图中断) happens when Dynamo encounters unsupported Python (不支持的 Python), forcing it to fall back to eager mode (回退到 eager 模式) and split the function into multiple subgraphs (多个子图).

```python
import torch

@torch.compile
def has_break(x):
    y = x + 1
    print("hello")              # <-- graph break: print is a side effect (副作用)
    return y * 2

x = torch.randn(4, device='cuda')
out = has_break(x)
print(out.shape)
# torch.Size([4])

# Inspect:
exp = torch._dynamo.explain(has_break)(x)
print("graph count:", exp.graph_count)        # >= 2 due to print
print("first reason:", exp.break_reasons[0].reason if exp.break_reasons else "none")
# graph count: 2
# first reason: ... (print causes break)
```

<br>

## 5. Common Graph Break Causes

Graph breaks (图中断) come from Python features (Python 特性) Dynamo can't statically reason about (静态推理); avoiding them maximizes fusion (融合).

| Cause (原因)                       | Example (示例)                       |
| ---------------------------------- | ------------------------------------ |
| `print()` side effect              | `print(x.shape)`                     |
| `.item()` / CPU sync (CPU 同步)    | `if x.item() > 0:`                   |
| Numpy / pandas calls (Numpy 调用)  | `np.sin(x)`                          |
| Unsupported builtins (不支持内建)  | `hash(tensor)`                       |
| Custom C extension (自定义 C 扩展) | calling unregistered op (未注册算子) |
| Dynamic Python class (动态类)      | `setattr(self, name, val)`           |

<br>

## 6. `fullgraph=True`

Setting `fullgraph=True` makes Dynamo raise an error (报错) on any graph break (图中断), enforcing a single fused graph (单个融合图).

```python
import torch

@torch.compile(fullgraph=True)
def strict_fn(x):
    return x.sin().cos()                   # pure tensor ops -> ok

x = torch.randn(4, device='cuda')
print(strict_fn(x).shape)
# torch.Size([4])

# This would raise an error:
# @torch.compile(fullgraph=True)
# def bad(x):
#     print("breaking")                   # raises Unsupported / graph break
#     return x + 1
```

<br>

## 7. Guards

Guards (守卫) are runtime checks (运行时检查) Dynamo inserts to validate (验证) that captured assumptions (捕获时假设) — like tensor shape (形状), dtype (数据类型), device (设备) — still hold on the next call.

```python
import torch

@torch.compile
def fn(x):
    return x * 2

# First call: compiles + installs guards on shape/dtype/device
fn(torch.randn(4, device='cuda'))

# Same shape/dtype/device -> guards pass, reuse compiled graph
fn(torch.randn(4, device='cuda'))

# Different shape -> guards fail, recompile (重编译)
fn(torch.randn(8, device='cuda'))

print("Guards trigger recompile when shape/dtype/device changes")
# Guards trigger recompile when shape/dtype/device changes
```

<br>

## 8. Recompilation

Recompilation (重编译) is triggered when guards fail (守卫失败); excessive recompilation (过度重编译) destroys speedup (破坏加速) and is a common pitfall (常见陷阱).

```python
import torch
import torch._dynamo as dynamo

@torch.compile
def fn(x):
    return x.sum()

# Track recompiles
dynamo.reset()
for n in [4, 8, 16, 32]:
    fn(torch.randn(n, device='cuda'))

# Each new shape triggers a recompile.
# Fix: use dynamic=True to share one graph across shapes
@torch.compile(dynamic=True)
def fn_dynamic(x):
    return x.sum()

dynamo.reset()
for n in [4, 8, 16, 32]:
    fn_dynamic(torch.randn(n, device='cuda'))
print("dynamic=True avoids per-shape recompile")
# dynamic=True avoids per-shape recompile
```

<br>

## 9. `torch._dynamo.reset`

`torch._dynamo.reset()` clears the compile cache (编译缓存) and guards (守卫), forcing fresh compilation (重新编译) — useful for benchmarking (基准测试) and debugging.

```python
import torch
import torch._dynamo as dynamo

@torch.compile
def fn(x):
    return x + 1

x = torch.randn(4, device='cuda')
fn(x)                                     # first call: compile
fn(x)                                     # second call: reuse cache

dynamo.reset()                            # wipe cache + guards
fn(x)                                     # forced to recompile
print("Cache reset; next call recompiles fresh")
# Cache reset; next call recompiles fresh
```

<br>

## 10. Inspecting the FX Graph

Using a custom backend (自定义后端) that just prints the captured FX graph (FX 图) is the easiest way to see what Dynamo extracted (提取).

```python
import torch
from torch.fx import GraphModule

def my_backend(gm: GraphModule, example_inputs):
    print("=== Captured FX Graph ===")
    gm.graph.print_tabular()
    return gm.forward                     # just run eager

@torch.compile(backend=my_backend)
def fn(x, y):
    return (x + y).relu()

fn(torch.randn(4, device='cuda'), torch.randn(4, device='cuda'))
# === Captured FX Graph ===
# opcode         name    target                    args             kwargs
# -------------  ------  ------------------------  ---------------  --------
# placeholder    l_x_    L_x_                      ()               {}
# placeholder    l_y_    L_y_                      ()               {}
# call_function  add     <built-in function add>   (l_x_, l_y_)     {}
# call_function  relu    <built-in method relu>    (add,)           {}
# output         output  output                    ((relu,),)       {}
```

<br>

## 11. Logging Dynamo Internals

Setting environment variables (环境变量) like `TORCH_LOGS="dynamo"` exposes (暴露) Dynamo's tracing decisions (追踪决策) and recompile reasons (重编译原因).

```python
import os
os.environ["TORCH_LOGS"] = "dynamo,graph_breaks,recompiles"

import torch

@torch.compile
def fn(x):
    return x.sin()

x = torch.randn(4, device='cuda')
fn(x)
# Logs (in stderr):
# [INFO] Step 1: torchdynamo start tracing fn ...
# [INFO] produced graph with 1 op
# Useful for finding hidden graph breaks and recompiles
print("done")
# done
```

<br>

## 12. Dynamo vs Other Tracers

Dynamo's bytecode-level capture (字节码级捕获) handles real Python code (真实 Python 代码) — including data-dependent control flow (数据依赖控制流) — that older tracers (旧追踪器) cannot.

| Method                    | Approach                                 | Handles Python control flow (Python 控制流)? |
| ------------------------- | ---------------------------------------- | -------------------------------------------- |
| `torch.jit.trace`         | Run-once tensor recording (一次运行记录) | ❌ No                                         |
| `torch.jit.script`        | Static AST parsing (静态 AST 解析)       | ⚠️ Subset only                                |
| `torch.fx.symbolic_trace` | Symbolic Proxy tracing (符号 Proxy 追踪) | ❌ No                                         |
| **TorchDynamo**           | **Bytecode + frame eval hook**           | **✅ Yes (with graph breaks)**                |

<br>

## 13. Summary

TorchDynamo (TorchDynamo) is the **graph capture frontend (图捕获前端)** of `torch.compile`, using CPython's frame evaluation hook (帧求值钩子) to extract FX graphs (FX 图) and emit guards (守卫) for safe reuse (安全复用).

| Concept                      | Role                                             |
| ---------------------------- | ------------------------------------------------ |
| Frame eval hook (帧求值钩子) | Intercepts every Python function call (拦截调用) |
| FX Graph (FX 图)             | Captured tensor operations (捕获的张量操作)      |
| Guard (守卫)                 | Runtime check on shape/dtype/device              |
| Graph break (图中断)         | Fallback to eager on unsupported Python          |
| Recompile (重编译)           | Triggered when guards fail                       |
| Backend (后端)               | Receives the FX graph (e.g. Inductor)            |

<br> <br>
