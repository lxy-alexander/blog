---
title: "Inductor"
published: 2026-04-27
description: "Inductor"
image: ""
tags: ["pytorch","compile","Inductor"]
category: pytorch / compile
draft: false
lang: ""
createdAt: "2026-04-27T22:04:55.929.785246873Z"
---

# PyTorch Inductor

Inductor (Inductor 后端) is the default backend of `torch.compile` that lowers FX graphs  into fused Triton kernels for GPU and C++ kernels  for CPU, delivering most of `torch.compile`'s speedup (加速).

## 1. What Inductor Is

Inductor is a **define-by-run code generator (按需代码生成器)** that takes an FX graph (FX 图) from AOTAutograd (AOTAutograd) and emits optimized kernels (优化内核) — Triton on GPU, C++/OpenMP on CPU.

| Aspect                 | Description                                    |
| ---------------------- | ---------------------------------------------- |
| Layer (层级)           | Backend (后端)                                 |
| Input (输入)           | FX Graph (from AOTAutograd)                    |
| Output (输出)          | Triton kernels (GPU) / C++ kernels (CPU)       |
| Default for            | `torch.compile(..., backend="inductor")`       |
| Key feature (关键特性) | Vertical + horizontal fusion (纵向 + 横向融合) |

<br>

## 2. Position in the Compile Pipeline

Inductor sits at the bottom (最底层) of the `torch.compile` stack (栈), turning the AOT-partitioned FX graph (AOT 划分图) into actual executable kernels (可执行内核).

```
┌────────────────┐
│  Python code   │
└───────┬────────┘
        ▼
┌────────────────┐
│   TorchDynamo  │  bytecode → FX graph
└───────┬────────┘
        ▼
┌────────────────┐
│  AOTAutograd   │  joint fw/bw graph (联合图)
└───────┬────────┘
        ▼
┌────────────────┐
│    Inductor    │  ◄── this layer
│                │
│  ┌──────────┐  │
│  │ Lowering │  │  FX → Inductor IR (内部 IR)
│  └────┬─────┘  │
│       ▼        │
│  ┌──────────┐  │
│  │ Scheduling│ │  loop fusion (循环融合)
│  └────┬─────┘  │
│       ▼        │
│  ┌──────────┐  │
│  │ Codegen  │  │  emit Triton / C++
│  └──────────┘  │
└───────┬────────┘
        ▼
   GPU/CPU kernels
```

<br>

## 3. Lowering: FX → Inductor IR

Inductor first lowers (下沉) FX nodes (FX 节点) to its internal IR (内部 IR) — a low-level loop-based representation (循环式表示) suitable for fusion (融合) and code generation (代码生成).

| FX Op (FX 算子) | Inductor IR                                  |
| --------------- | -------------------------------------------- |
| `aten.add`      | `Pointwise(lambda i: load_x(i) + load_y(i))` |
| `aten.sum`      | `Reduction(combine_fn=add, init=0)`          |
| `aten.mm`       | `MatrixMultiply(lhs, rhs)` (template)        |
| `aten.relu`     | `Pointwise(lambda i: max(load_x(i), 0))`     |

<br>

## 4. Scheduling: Loop Fusion

The scheduler (调度器) merges (合并) compatible operations (兼容操作) into a single loop (单循环), which is the **#1 source (头号来源) of Inductor's speedup**.

```
   Before fusion:
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ for i: a*b  │→ │ for i: +c   │→ │ for i: relu │
   └─────────────┘  └─────────────┘  └─────────────┘
        3 kernel launches, 3× memory traffic (3 倍内存流量)

   After fusion:
   ┌─────────────────────────────────┐
   │ for i: relu((a[i]*b[i]) + c[i]) │
   └─────────────────────────────────┘
        1 kernel launch, 1× memory traffic
```

<br>

## 5. Code Generation: Triton (GPU)

For CUDA tensors (CUDA 张量), Inductor emits Triton kernels (Triton 内核); each fused subgraph (融合子图) becomes one Triton function (Triton 函数).

```python
import os
os.environ["TORCH_LOGS"] = "output_code"            # show generated code

import torch

@torch.compile
def fn(x, y):
    return (x * y + 1).relu()

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
out = fn(x, y)
print(out.shape)
# torch.Size([1024])
#
# Stderr also dumps generated Triton kernel like:
# @triton.jit
# def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
#     xoffset = tl.program_id(0) * XBLOCK
#     xindex = xoffset + tl.arange(0, XBLOCK)[:]
#     xmask = xindex < xnumel
#     tmp0 = tl.load(in_ptr0 + xindex, xmask)
#     tmp1 = tl.load(in_ptr1 + xindex, xmask)
#     tmp2 = tmp0 * tmp1
#     tmp3 = tmp2 + 1.0
#     tmp4 = tl.maximum(tmp3, 0.0)
#     tl.store(out_ptr0 + xindex, tmp4, xmask)
```

<br>

## 6. Code Generation: C++ (CPU)

For CPU tensors (CPU 张量), Inductor emits vectorized (向量化) C++ kernels using OpenMP (OpenMP) for multi-threading (多线程).

```python
import os
os.environ["TORCH_LOGS"] = "output_code"

import torch

@torch.compile
def fn(x, y):
    return (x * y + 1).relu()

x = torch.randn(1024)                                # CPU tensor
y = torch.randn(1024)
out = fn(x, y)
print(out.shape)
# torch.Size([1024])
#
# Generated C++ kernel like:
# extern "C" void kernel(const float* in0, const float* in1, float* out, long n) {
#     #pragma omp parallel for
#     for (long i = 0; i < n; ++i) {
#         float tmp = in0[i] * in1[i] + 1.0f;
#         out[i] = std::max(tmp, 0.0f);
#     }
# }
```

<br>

## 7. Compile Modes

Inductor exposes three compile modes (编译模式) that trade compile time (编译时间) for runtime performance (运行时性能).

| Mode                | Behavior                                               | Best for                                         |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `"default"`         | Balanced (平衡) compile time & speed                   | General use                                      |
| `"reduce-overhead"` | Adds CUDA Graphs (CUDA 图)                             | Small models, low-latency inference (低延迟推理) |
| `"max-autotune"`    | Aggressive autotuning (激进自动调优), templates (模板) | Production training (生产训练)                   |

```python
import torch

@torch.compile(mode="max-autotune")
def matmul_relu(a, b):
    return torch.relu(a @ b)

a = torch.randn(512, 512, device='cuda')
b = torch.randn(512, 512, device='cuda')
out = matmul_relu(a, b)
print(out.shape)
# torch.Size([512, 512])
# First call is slow (autotuning); subsequent calls are very fast.
```

<br>

## 8. CUDA Graphs (`reduce-overhead`)

`mode="reduce-overhead"` wraps (包装) the compiled kernel sequence in a CUDA Graph (CUDA 图), eliminating per-launch overhead (启动开销) for small kernels.

```python
import torch

@torch.compile(mode="reduce-overhead")
def fn(x):
    return x.sin().cos().sin()

x = torch.randn(128, device='cuda')

# Warm-up: compiles + records CUDA Graph
for _ in range(3):
    out = fn(x)

print(out.shape)
# torch.Size([128])
# Subsequent runs replay the recorded graph (~0 launch overhead)
```

<br>

## 9. Max-Autotune (Templates)

`mode="max-autotune"` enables matmul (矩阵乘) and conv (卷积) **templates (模板)** that benchmark multiple Triton configs (Triton 配置) and pick the fastest.

```python
import torch

@torch.compile(mode="max-autotune")
def gemm(a, b):
    return a @ b

a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

# First call benchmarks several Triton configs (BLOCK_M, BLOCK_N, etc.)
out = gemm(a, b)
print(out.shape, out.dtype)
# torch.Size([1024, 1024]) torch.float16
```

<br>

## 10. Vertical vs Horizontal Fusion

Inductor performs both fusion strategies (融合策略); horizontal fusion (横向融合) packs independent ops (独立操作) into one kernel to launch fewer kernels.

```
   Vertical Fusion (纵向融合):
   ┌──────┐    ┌──────┐    ┌──────┐
   │ a*b  │───►│ +c   │───►│ relu │     (chain of dependent ops)
   └──────┘    └──────┘    └──────┘
                    ▼
            ┌────────────────────┐
            │ relu(a*b + c)      │
            └────────────────────┘

   Horizontal Fusion (横向融合):
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ y = sin(x)│ │ z = cos(x)│ │ w = exp(x)│   (independent)
   └──────────┘  └──────────┘  └──────────┘
                    ▼
            ┌────────────────────┐
            │ y, z, w = ..(x)..  │   one kernel, three outputs
            └────────────────────┘
```

<br>

## 11. Inspecting Inductor Output

Setting environment variables (环境变量) like `TORCH_LOGS` and `TORCH_COMPILE_DEBUG` exposes (暴露) every stage of Inductor — IR (内部 IR), schedule (调度), and final code (最终代码).

| Variable                 | What It Shows                |
| ------------------------ | ---------------------------- |
| `TORCH_LOGS=output_code` | Final Triton/C++ source      |
| `TORCH_LOGS=inductor`    | Inductor pass logs           |
| `TORCH_LOGS=schedule`    | Fusion decisions (融合决策)  |
| `TORCH_COMPILE_DEBUG=1`  | Dumps all IRs to disk (磁盘) |

```python
import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"

import torch

@torch.compile
def fn(x):
    return x.sin().cos()

fn(torch.randn(8, device='cuda'))
print("Inductor wrote debug artifacts to ./torch_compile_debug/")
# Inductor wrote debug artifacts to ./torch_compile_debug/
# Inside: pre/post fusion IRs, generated Triton code, scheduling logs.
```

<br>

## 12. Inductor's Optimization Passes

Inductor applies many passes (编译 pass) on the way from FX graph to kernels — the most impactful (最有影响力) ones are listed below.

| Pass                                  | Purpose                              |
| ------------------------------------- | ------------------------------------ |
| Constant folding (常量折叠)           | Pre-compute static values            |
| Dead code elimination (死代码消除)    | Remove unused nodes                  |
| Memory planning (内存规划)            | Reuse buffers (复用缓冲)             |
| Loop fusion (循环融合)                | Combine compatible loops             |
| Tiling & vectorization (分块与向量化) | Use SIMD / Tensor Cores              |
| Layout optimization (布局优化)        | Choose contiguous strides (连续步长) |
| Pattern matching (模式匹配)           | Replace e.g. `softmax` with template |

<br>

## 13. Performance Tips

A few practical guidelines (实用准则) help Inductor produce faster code (更快代码) consistently across workloads.

| Tip                                             | Reason                                   |
| ----------------------------------------------- | ---------------------------------------- |
| Use `mode="max-autotune"` for matmul-heavy code | Picks best Triton config                 |
| Use `mode="reduce-overhead"` for small models   | Eliminates launch overhead               |
| Avoid `.item()` and `.cpu()` in hot path        | Causes graph break (图中断)              |
| Use `dynamic=True` for variable shapes          | Avoids recompiles (避免重编译)           |
| Prefer fused activations (融合激活)             | Inductor fuses pointwise ops             |
| Use `torch.float16` / `bfloat16`                | Tensor Cores activate (启用 Tensor Core) |

<br>

## 14. Summary

Inductor (Inductor 后端) is the **codegen-and-fusion engine (代码生成与融合引擎)** of `torch.compile` — it lowers FX graphs (FX 图) to a loop IR (循环 IR), fuses operations (融合操作), and emits Triton (GPU) or C++ (CPU) kernels.

| Stage              | Output                                       |
| ------------------ | -------------------------------------------- |
| Lowering (下沉)    | Inductor IR (loop-based)                     |
| Scheduling (调度)  | Fused op groups (融合组)                     |
| Codegen (代码生成) | Triton / C++ source                          |
| Compilation (编译) | `.so` / Triton-compiled cubin (cubin 二进制) |
| Execution (执行)   | GPU/CPU kernel launch                        |

| Key Feature                      | Why It Matters                          |
| -------------------------------- | --------------------------------------- |
| Loop fusion (循环融合)           | Reduces memory traffic (内存流量)       |
| Triton codegen (Triton 代码生成) | Generates GPU code without writing CUDA |
| CUDA Graphs (CUDA 图)            | Eliminates launch overhead              |
| Autotuning (自动调优)            | Picks best kernel config per shape      |
| Functional IR (函数式 IR)        | Enables aggressive analysis (激进分析)  |

<br> <br>
