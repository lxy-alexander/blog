---
title: "CuTe API Overview"
published: 2026-04-27
description: "CuTe API Overview"
image: ""
tags: ["cute","api","CuTe API Overview"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T17:35:11.470.196362686Z"
---
# CuTe API Overview

CuTe (CUDA Tensors) is NVIDIA's tensor abstraction library (张量抽象库) inside CUTLASS 3.x for writing high-performance GPU kernels (高性能 GPU 内核); mastering ~**40-60 core APIs** is enough to write production-grade (生产级) kernels.

## 1. Total API Scope

CuTe's API surface is larger than Triton (50-70) but smaller than full CUTLASS — the core (核心) is built around two concepts (概念): **Layout** (布局) and **Tensor** (张量).

| Category                            | API Count | Importance    | Difficulty (难度) |
| ----------------------------------- | --------- | ------------- | ----------------- |
| Layout & Shape (布局与形状)         | ~10       | ⭐⭐⭐ Must know | High              |
| Tensor Construction (张量构造)      | ~6        | ⭐⭐⭐ Must know | Medium            |
| Tiling & Partitioning (分块与划分)  | ~8        | ⭐⭐⭐ Must know | High              |
| Copy Operations (拷贝操作)          | ~6        | ⭐⭐⭐ Must know | High              |
| MMA (Matrix Multiply Accumulate)    | ~8        | ⭐⭐⭐ Must know | Very High         |
| Atom & TiledMMA/Copy                | ~8        | ⭐⭐ Important  | Very High         |
| Algorithm Helpers (算法辅助)        | ~6        | ⭐⭐ Important  | Medium            |
| Swizzle & Layout Algebra (布局代数) | ~6        | ⭐⭐ Important  | Very High         |
| Print & Debug (调试)                | ~4        | ⭐ Optional    | Low               |

<br>

## 2. Layout & Shape (Must Know)

Layout (布局) is CuTe's core abstraction (核心抽象) — a function (函数) mapping logical coordinates (逻辑坐标) to memory offsets (内存偏移), defined by `Shape` and `Stride`.

| API                          | Purpose                             |
| ---------------------------- | ----------------------------------- |
| `make_shape(...)`            | Create shape tuple (形状元组)       |
| `make_stride(...)`           | Create stride tuple (步长元组)      |
| `make_layout(shape, stride)` | Build a layout (构造布局)           |
| `Layout<Shape, Stride>`      | Static layout type (静态布局类型)   |
| `size(layout)`               | Total elements (总元素数)           |
| `rank(layout)`               | Number of modes (模态数)            |
| `shape(layout)`              | Get shape                           |
| `stride(layout)`             | Get stride                          |
| `Int<N>{}`                   | Compile-time integer (编译期整数)   |
| `_1, _2, _4, ...`            | Compile-time constants (编译期常量) |

```cpp
using namespace cute;
auto layout = make_layout(make_shape(_4{}, _8{}),    // 4x8 matrix
                          make_stride(_8{}, _1{})); // row-major
print(layout);   // (_4,_8):(_8,_1)
```

<br>

## 3. Tensor Construction (Must Know)

Tensor (张量) = pointer (指针) + Layout (布局); it represents a logical view (逻辑视图) of memory without owning data (不拥有数据).

| API                          | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `make_tensor(ptr, layout)`   | Wrap pointer with layout (包装指针)              |
| `make_tensor(layout)`        | Allocate register tensor (寄存器张量)            |
| `make_gmem_ptr(ptr)`         | Tag as global memory (全局内存)                  |
| `make_smem_ptr(ptr)`         | Tag as shared memory (共享内存)                  |
| `make_rmem_ptr<T>()`         | Register memory pointer (寄存器内存)             |
| `make_fragment_like(tensor)` | Allocate matching register fragment (寄存器片段) |

<br>

## 4. Tiling & Partitioning (Must Know)

Tiling (分块) divides a large tensor into smaller tiles (小分块), and partitioning (划分) assigns tiles to threads (线程) — the foundation (基础) of GPU parallelism (GPU 并行).

| API                                     | Purpose                                |
| --------------------------------------- | -------------------------------------- |
| `local_tile(tensor, tile_shape, coord)` | Extract one tile (提取分块)            |
| `local_partition(tensor, layout, tid)`  | Partition among threads (线程间划分)   |
| `tiled_divide(tensor, tile)`            | Tile a tensor (张量分块)               |
| `flat_divide(tensor, tile)`             | Flatten tile dims (展平分块维度)       |
| `zipped_divide(tensor, tile)`           | Zip tile/rest dims (压缩分块维度)      |
| `logical_divide(layout, tile)`          | Logical layout division (逻辑布局除法) |
| `composition(A, B)`                     | Compose two layouts (布局复合)         |
| `tile_to_shape(tile, shape)`            | Repeat tile to shape (重复分块)        |

<br>

## 5. Copy Operations (Must Know)

`cute::copy` is the unified async/sync (统一异步/同步) data movement (数据移动) API, with optional `Atom` (原子操作) controlling the underlying instruction (底层指令).

| API                       | Purpose                               |
| ------------------------- | ------------------------------------- |
| `copy(src, dst)`          | Generic copy (通用拷贝)               |
| `copy(atom, src, dst)`    | Copy using a specific atom (使用原子) |
| `copy_if(pred, src, dst)` | Predicated copy (谓词拷贝)            |
| `cp_async_fence()`        | Async copy fence (异步拷贝屏障)       |
| `cp_async_wait<N>()`      | Wait for N async copies (等待 N 个)   |
| `cp_async_wait_all()`     | Wait for all (等待全部)               |

```cpp
using namespace cute;
copy(gA, sA);                                    // global -> shared
cp_async_fence();
cp_async_wait<0>();                              // wait all in-flight
__syncthreads();
```

<br>

## 6. MMA Operations (Must Know)

MMA (Matrix Multiply Accumulate, 矩阵乘累加) maps to Tensor Cores (张量核心), the source (源头) of FLOPs (浮点运算) on Ampere/Hopper GPUs.

| API                               | Purpose                                |
| --------------------------------- | -------------------------------------- |
| `gemm(mma, A, B, C)`              | $$C = A \cdot B + C$$                  |
| `gemm(mma, alpha, A, B, beta, C)` | Scaled GEMM (缩放 GEMM)                |
| `make_tiled_mma(atom, layout)`    | Build a tiled MMA (构造分块 MMA)       |
| `MMA_Atom<MMA_Op>`                | Wrap a hardware MMA op (硬件 MMA 操作) |
| `SM80_16x8x16_F32F16F16F32_TN`    | Ampere fp16 MMA (安培 fp16)            |
| `SM90_64x128x16_F32F16F16_SS`     | Hopper WGMMA (Hopper 大规模 MMA)       |
| `tiled_mma.get_thread_slice(tid)` | Get per-thread slice (每线程切片)      |
| `partition_fragment_A/B/C`        | Allocate A/B/C fragments (分配片段)    |

<br>

## 7. Atom & TiledMMA/TiledCopy (Important)

Atoms (原子) describe a single hardware instruction (单条硬件指令); TiledMMA/TiledCopy compose multiple atoms (多个原子) into a thread-block-level (线程块级) operation.

| API                                             | Purpose                                |
| ----------------------------------------------- | -------------------------------------- |
| `Copy_Atom<Op, T>`                              | Wrap a copy instruction (包装拷贝指令) |
| `make_tiled_copy(atom, thr_layout, val_layout)` | Build tiled copy (构造分块拷贝)        |
| `tiled_copy.get_thread_slice(tid)`              | Get thread copy slice (线程拷贝切片)   |
| `SM80_CP_ASYNC_CACHEALWAYS<T>`                  | Ampere `cp.async` (cp.async 指令)      |
| `UniversalCopy<T>`                              | Generic copy atom (通用拷贝原子)       |
| `AutoVectorizingCopy`                           | Auto-vectorized (自动向量化)           |
| `MMA_Traits<Op>`                                | MMA shape/types metadata (MMA 元数据)  |
| `MMA_Atom<Op>`                                  | Wrap MMA instruction (包装 MMA 指令)   |

<br>

## 8. Algorithm Helpers (Important)

These helpers (辅助函数) handle common patterns (常见模式) like fill (填充), clear (清零), and axpby (axpby 运算).

| API                       | Purpose                            |
| ------------------------- | ---------------------------------- |
| `clear(tensor)`           | Zero the tensor (清零)             |
| `fill(tensor, value)`     | Fill with constant (填充常量)      |
| `axpby(a, X, b, Y)`       | $$Y = aX + bY$$                    |
| `transform(src, dst, fn)` | Element-wise function (逐元素函数) |
| `for_each(tensor, fn)`    | Iterate over elements (遍历元素)   |
| `cosize(layout)`          | Codomain size (值域大小)           |

<br>

## 9. Swizzle & Layout Algebra (Important)

Swizzle (混洗) avoids shared memory bank conflicts (共享内存 bank 冲突) by remapping addresses (重映射地址); layout algebra (布局代数) composes/inverts layouts symbolically (符号化).

| API                            | Purpose                                    |
| ------------------------------ | ------------------------------------------ |
| `Swizzle<B, M, S>`             | XOR-based swizzle pattern (基于异或的混洗) |
| `composition(swizzle, layout)` | Apply swizzle to layout (应用混洗)         |
| `inverse(layout)`              | Invert a layout (逆布局)                   |
| `right_inverse(layout)`        | Right inverse (右逆)                       |
| `complement(layout, cosize)`   | Layout complement (布局补)                 |
| `coalesce(layout)`             | Merge contiguous modes (合并连续模态)      |

<br>

## 10. Print & Debug (Optional)

CuTe provides rich printing (丰富打印) for layouts, tensors, and tiled operations — essential for debugging (调试) layout bugs (布局错误).

| API                    | Purpose                            |
| ---------------------- | ---------------------------------- |
| `print(layout)`        | Print layout text form (文本形式)  |
| `print_tensor(tensor)` | Print tensor values (张量值)       |
| `print_layout(layout)` | Visualize layout (可视化布局)      |
| `print_latex(layout)`  | LaTeX visualization (LaTeX 可视化) |

<br>

## 11. Learning Roadmap

A practical learning path (学习路径) — most kernels (90%) need only the **first 30 APIs** in the "Must Know" tier (必学层).

| Stage                                   | APIs to Master                                               | Time      |
| --------------------------------------- | ------------------------------------------------------------ | --------- |
| **Stage 1: Layout Basics (布局基础)**   | `make_shape`, `make_stride`, `make_layout`, `Int<N>`, `print` | 3-5 days  |
| **Stage 2: Tensor & Tile (张量与分块)** | `make_tensor`, `local_tile`, `local_partition`, `tiled_divide` | 1 week    |
| **Stage 3: Copy (拷贝)**                | `copy`, `Copy_Atom`, `make_tiled_copy`, `cp_async_*`         | 1-2 weeks |
| **Stage 4: GEMM (矩阵乘)**              | `MMA_Atom`, `make_tiled_mma`, `gemm`, `partition_fragment_*` | 2-3 weeks |
| **Stage 5: Optimization (优化)**        | `Swizzle`, pipelining (流水线), warp specialization (warp 特化) | 3-4 weeks |
| **Stage 6: Hopper (霍珀)**              | TMA (`SM90_TMA_*`), WGMMA (`SM90_*`), barrier (屏障)         | 2-4 weeks |

<br>

## 12. Key Difference vs Triton

CuTe is more powerful but harder (更难) than Triton — it exposes hardware details (硬件细节) like layouts and atoms, while Triton hides them.

| Aspect                         | Triton                         | CuTe                                   |
| ------------------------------ | ------------------------------ | -------------------------------------- |
| Language (语言)                | Python DSL                     | C++ template (C++ 模板)                |
| API count (API 数)             | 50-70                          | 40-60 core, ~100+ total                |
| Abstraction level (抽象层级)   | Block-level (块级)             | Thread-level + block-level             |
| Layout control (布局控制)      | Implicit (隐式)                | Explicit (显式)                        |
| Performance ceiling (性能上限) | ~95% of cuBLAS                 | 100% (matches cuBLAS)                  |
| Learning curve (学习曲线)      | 1-2 weeks                      | 2-3 months                             |
| Best for (适用于)              | Custom ops, prototyping (原型) | Production kernels (生产内核), CUTLASS |

<br>

## 13. Summary

To write **high-performance CuTe kernels** (高性能 CuTe 内核), master ~**30 core APIs** across Layout, Tensor, Tiling, Copy, and MMA; the full surface (完整范围) is around **80-100 APIs** including atoms, swizzles, and Hopper-specific (Hopper 特有) instructions.

| Tier (层级)          | Count | Coverage                                |
| -------------------- | ----- | --------------------------------------- |
| ⭐⭐⭐ Must Know (必学) | ~30   | Covers 90% kernels (覆盖 90% 内核)      |
| ⭐⭐ Important (重要)  | ~25   | Tiled MMA/Copy, swizzle, layout algebra |
| ⭐ Optional (可选)    | ~15   | Debug, advanced layout ops, Hopper TMA  |

**Realistic timeline (现实时间线):** Going from zero to writing FlashAttention-quality (FlashAttention 级别) CuTe kernels takes about **2-3 months** of focused study (专注学习).

<br> <br>