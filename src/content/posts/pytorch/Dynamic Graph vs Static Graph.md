---
title: "Dynamic Graph vs Static Graph"
published: 2026-04-15
description: "Dynamic Graph vs Static Graph"
image: ""
tags: ["pytorch","Dynamic Graph vs Static Graph"]
category: pytorch
draft: false
lang: ""
---

# I. Dynamic Graph vs Static Graph (动态图 vs 静态图)

## 1. What is Dynamic Graph? (动态图是什么)

### 1) Definition (定义)

A **Dynamic Computation Graph (动态图计算图)** is built during runtime (运行时构建).

👉 The graph changes as the program executes (执行时动态变化).

---

### 2) Characteristics (特点)

- Defined on-the-fly (即时定义)
- Flexible control flow (灵活控制流)
- Easy debugging (易调试)

---

### 3) Example (示例)

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3

y.backward()
print(x.grad)  # dy/dx = 2x = 4
```

👉 The graph is created dynamically when `y` is computed.

------

## 2. What is Static Graph? (静态图是什么)

### 1) Definition (定义)

A **Static Computation Graph (静态计算图)** is defined before execution (运行前定义).

👉 The graph structure does not change during runtime (运行时不变).

------

### 2) Characteristics (特点)

-   Predefined graph (预先定义)
-   Optimized before execution (执行前优化)
-   Better performance (更高性能)

------

### 3) Conceptual Example (概念示例)

```python
# Pseudo-code (伪代码)
x = placeholder()
y = x * x + 3

# Build graph first (先构建图)
graph = build_graph(x, y)

# Execute later (再执行)
run(graph, feed_dict={x: 2})
```

------

## 3. Key Differences (核心区别)

| Feature (特性)        | Dynamic Graph    | Static Graph              |
| --------------------- | ---------------- | ------------------------- |
| Build Time (构建时间) | Runtime (运行时) | Before execution (运行前) |
| Flexibility (灵活性)  | High (高)        | Low (低)                  |
| Debugging (调试)      | Easy (容易)      | Hard (困难)               |
| Performance (性能)    | Medium (中等)    | High (高)                 |

------

## 4. Mathematical View (数学视角)

A computation graph (计算图) represents operations:

$$
y = f(x)
$$

-   Dynamic graph: build $f(x)$ during execution (运行时构建函数)
-   Static graph: define $f(x)$ before execution (运行前定义函数)

------

## 5. When to Use (何时使用)

### 1) Dynamic Graph

-   Research (科研)
-   Prototyping (快速实验)

------

### 2) Static Graph

-   Production systems (生产环境)
-   Performance-critical tasks (高性能场景)

------

## 6. One-Line Summary (一句话总结)

👉 Dynamic graph (动态图) = flexible and easy (灵活易用)
👉 Static graph (静态图) = efficient and optimized (高效优化)

```

```
