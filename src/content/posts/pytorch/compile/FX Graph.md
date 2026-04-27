---
title: "FX Graph"
published: 2026-04-27
description: "FX Graph"
image: ""
tags: ["pytorch","compile","FX Graph"]
category: pytorch / compile
draft: false
lang: ""
createdAt: "2026-04-27T21:57:36.004.413666974Z"
---

# PyTorch FX Graph

FX Graph (FX 图) is PyTorch's intermediate representation (中间表示, IR) — a directed graph (有向图) where nodes (节点) are operations (操作) and edges (边) are tensor data flow (张量数据流), produced by Dynamo (Dynamo) and consumed by backends (后端) like Inductor (Inductor).

## 1. What FX Graph Is

FX Graph is a **functional, side-effect-free IR (无副作用中间表示)** representing the captured tensor program (捕获的张量程序) as a list of nodes (节点列表), each describing one operation.

```
        ┌─────────────────────────────────────────────┐
        │              FX Graph (FX 图)                │
        │                                             │
        │   placeholder x ───┐                        │
        │                    ├──► add ──► relu ──► output
        │   placeholder y ───┘                        │
        │                                             │
        └─────────────────────────────────────────────┘
              ▲                                    │
              │ Dynamo captures                    │ Inductor lowers
              │                                    ▼
        Python source                       Triton / C++ kernels
```

<br>

## 2. Node Types

An FX Graph has **5 node types (5 种节点类型)** — each representing a different role (角色) in the computation (计算).

| Opcode          | Role                                | Example           |
| --------------- | ----------------------------------- | ----------------- |
| `placeholder`   | Function input (函数输入)           | `x`, `y`          |
| `get_attr`      | Read module attribute (读取属性)    | `self.weight`     |
| `call_function` | Call a free function (调用自由函数) | `torch.add(x, y)` |
| `call_method`   | Call a tensor method (调用张量方法) | `x.relu()`        |
| `call_module`   | Call a submodule (调用子模块)       | `self.linear(x)`  |
| `output`        | Final result (最终输出)             | `return out`      |

<br>

## 3. Visualizing a Simple FX Graph

A function `f(x, y) = relu(x + y)` produces a 4-node graph (4 节点图) — 2 inputs, 1 add, 1 relu, plus an output node.

```
   x ──┐
       ├──► add(x, y) ──► relu(add) ──► output
   y ──┘
import torch
from torch.fx import GraphModule

def my_backend(gm: GraphModule, example_inputs):
    print("=== FX Graph (tabular) ===")
    gm.graph.print_tabular()
    return gm.forward

@torch.compile(backend=my_backend)
def f(x, y):
    return (x + y).relu()

f(torch.randn(4, device='cuda'), torch.randn(4, device='cuda'))
# === FX Graph (tabular) ===
# opcode         name    target                    args         kwargs
# -------------  ------  ------------------------  -----------  --------
# placeholder    l_x_    L_x_                      ()           {}
# placeholder    l_y_    L_y_                      ()           {}
# call_function  add     <built-in function add>   (l_x_,l_y_)  {}
# call_method    relu    relu                      (add,)       {}
# output         output  output                    ((relu,),)   {}
```

<br>

## 4. Visualizing an MLP FX Graph

A 2-layer MLP (双层 MLP) with `Linear → ReLU → Linear` produces a graph (图) with `call_module` (调用子模块) and `call_function` nodes (节点).

```
   x ──► linear_1 ──► relu ──► linear_2 ──► output
        (call_module) (call_function) (call_module)
import torch
import torch.nn as nn
from torch.fx import GraphModule

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Use FX symbolic tracing to capture the graph
from torch.fx import symbolic_trace
traced = symbolic_trace(MLP())
traced.graph.print_tabular()
# opcode         name    target              args      kwargs
# -------------  ------  ------------------  --------  --------
# placeholder    x       x                   ()        {}
# call_module    fc1     fc1                 (x,)      {}
# call_function  relu    <built-in ... relu> (fc1,)    {}
# call_module    fc2     fc2                 (relu,)   {}
# output         output  output              (fc2,)    {}
```

<br>

## 5. Visualizing a Branch (Element-wise Sum)

A function with two branches (两个分支) `sin(x) + cos(x)` shows a diamond shape (菱形结构) where one input feeds two operations (两个操作).

```
              ┌──► sin ──┐
       x ─────┤          ├──► add ──► output
              └──► cos ──┘
import torch
from torch.fx import GraphModule

def my_backend(gm: GraphModule, example_inputs):
    print("=== Diamond FX Graph ===")
    gm.graph.print_tabular()
    return gm.forward

@torch.compile(backend=my_backend)
def f(x):
    return x.sin() + x.cos()

f(torch.randn(4, device='cuda'))
# === Diamond FX Graph ===
# opcode         name    target                    args        kwargs
# -------------  ------  ------------------------  ----------  --------
# placeholder    l_x_    L_x_                      ()          {}
# call_method    sin     sin                       (l_x_,)     {}
# call_method    cos     cos                       (l_x_,)     {}
# call_function  add     <built-in function add>   (sin,cos)   {}
# output         output  output                    ((add,),)   {}
```

<br>

## 6. Iterating Over Nodes

Each FX Graph (FX 图) is iterable (可迭代), exposing nodes (节点) as objects with `.op`, `.target`, `.args`, and `.users` for graph analysis (图分析) and rewriting (重写).

```python
import torch
from torch.fx import symbolic_trace

def f(x, y):
    return torch.relu(x + y)

gm = symbolic_trace(f)
for node in gm.graph.nodes:
    print(f"op={node.op:14s} name={node.name:8s} target={node.target}")
# op=placeholder    name=x        target=x
# op=placeholder    name=y        target=y
# op=call_function  name=add      target=<built-in function add>
# op=call_function  name=relu     target=<built-in method relu of type object>
# op=output         name=output   target=output
```

<br>

## 7. Visualizing as Image (with `pydot`)

`torch.fx.passes.graph_drawer.FxGraphDrawer` renders (渲染) the FX graph (FX 图) into a PNG/SVG image (图像), giving a clear visual diagram (可视化图).

```python
import torch
from torch.fx import symbolic_trace
from torch.fx.passes.graph_drawer import FxGraphDrawer

def f(x, y):
    return (x + y).relu().sum()

gm = symbolic_trace(f)
drawer = FxGraphDrawer(gm, "f")
drawer.get_dot_graph().write_svg("fx_graph.svg")
print("Saved fx_graph.svg")
# Saved fx_graph.svg
# (Open in browser to see boxes-and-arrows visualization)
#
# Diagram (textual):
#
#     ┌──────┐    ┌──────┐
#     │  x   │    │  y   │
#     └──┬───┘    └──┬───┘
#        │           │
#        └─────┬─────┘
#              ▼
#          ┌─────┐
#          │ add │
#          └──┬──┘
#             ▼
#          ┌──────┐
#          │ relu │
#          └──┬───┘
#             ▼
#          ┌─────┐
#          │ sum │
#          └──┬──┘
#             ▼
#          ┌────────┐
#          │ output │
#          └────────┘
```

<br>

## 8. Modifying an FX Graph

FX graphs are mutable (可变) — you can insert (插入), replace (替换), or erase (删除) nodes to implement custom passes (自定义编译 pass) like operator fusion (算子融合).

```python
import torch
from torch.fx import symbolic_trace

def f(x):
    return x + 1

gm = symbolic_trace(f)

# Insert a node that doubles the result before output
with gm.graph.inserting_before(list(gm.graph.nodes)[-1]):  # before 'output'
    add_node = [n for n in gm.graph.nodes if n.op == 'call_function'][0]
    new_node = gm.graph.call_function(torch.mul, args=(add_node, 2))

# Rewire the output to use the new node
output_node = list(gm.graph.nodes)[-1]
output_node.args = (new_node,)
gm.recompile()

print(gm(torch.tensor([1., 2., 3.])))
# tensor([4., 6., 8.])    -- (x+1) * 2
```

<br>

## 9. FX Graph in the Compile Pipeline

FX Graph sits between (位于) Dynamo (前端) and Inductor (后端), serving as the canonical exchange format (规范交换格式) for all PyTorch compilers.

```
   ┌────────────────┐
   │ Python source  │   user code
   └───────┬────────┘
           │
   ┌───────▼────────┐
   │   TorchDynamo  │   bytecode capture (字节码捕获)
   └───────┬────────┘
           │ emits
   ┌───────▼────────┐
   │   FX Graph     │ ◄── the IR every backend speaks (统一 IR)
   └───────┬────────┘
           │ consumed by
   ┌───────▼────────┐
   │ AOTAutograd    │   adds backward graph (反向图)
   └───────┬────────┘
           │
   ┌───────▼────────┐
   │   Inductor     │   lowers to Triton / C++ (下沉到 Triton/C++)
   └───────┬────────┘
           │
   ┌───────▼────────┐
   │ GPU kernels    │   actual execution (实际执行)
   └────────────────┘
```

<br>

## 10. Common FX Operations Cheat Sheet

These are the most common APIs (常用 API) you'll touch when writing custom compiler passes (自定义编译 pass) over FX graphs.

| API                                 | Purpose                             |
| ----------------------------------- | ----------------------------------- |
| `gm.graph.nodes`                    | Iterate nodes (遍历节点)            |
| `gm.graph.print_tabular()`          | Print graph as table (表格打印)     |
| `gm.graph.call_function(fn, args)`  | Insert function call (插入函数调用) |
| `gm.graph.erase_node(node)`         | Remove node (删除节点)              |
| `node.replace_all_uses_with(other)` | Replace node uses (替换节点使用)    |
| `gm.recompile()`                    | Regenerate forward (重新生成前向)   |
| `FxGraphDrawer(gm).get_dot_graph()` | Visualize graph (可视化)            |
| `symbolic_trace(fn)`                | Trace function to FX (追踪到 FX)    |

<br>

## 11. Summary

FX Graph (FX 图) is PyTorch's **functional, mutable, visualizable IR (函数式、可变、可视化的 IR)** — the universal language (通用语言) connecting graph capture (图捕获) to backend code generation (后端代码生成).

| Property               | Value                                    |
| ---------------------- | ---------------------------------------- |
| Type (类型)            | Functional IR (函数式 IR)                |
| Producers (生产者)     | Dynamo / `symbolic_trace`                |
| Consumers (消费者)     | Inductor / TensorRT / custom backends    |
| Mutable (可变)?        | Yes — supports custom passes             |
| Visualizable (可视化)? | Yes — via `FxGraphDrawer`                |
| Node types (节点类型)  | 5: placeholder, get_attr, call_*, output |

<br> <br>
