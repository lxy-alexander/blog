---
title: "CuTeDSL APIs"
published: 2026-04-20
description: "CuTeDSL APIs"
image: ""
tags: ["CuTe","CuTeDSL APIs"]
category: CuTe
draft: false
lang: ""
---

# CuTe DSL API Study Notes — Index (索引)

## Overview (概览)

CuTe DSL is NVIDIA's Python-native interface in CUTLASS 4 for writing peak-performance GPU Kernels — it exposes Layout Algebra (布局代数), hardware Atoms (原子), and explicit memory hierarchy control.

$$ \text{Total APIs} \approx 26 + 22 + 16 = \textbf{~64 core APIs} $$

------

## File Structure (文件结构)

| File                                 | Tier                        | APIs          | What You Can Build                                        |
| ------------------------------------ | --------------------------- | ------------- | --------------------------------------------------------- |
| `CuTeDSL_Tier1_Foundation_APIs.md`   | Tier-1: Foundation (基础)   | ~26           | Any kernel: elementwise, tiled copy, basic GEMM skeleton  |
| `CuTeDSL_Tier2_Performance_APIs.md`  | Tier-2: Performance (性能)  | ~22           | Tensor Core GEMM, async copy, mixed precision, pipelining |
| `CuTeDSL_Tier3_Architecture_APIs.md` | Tier-3: Architecture (架构) | ~16+ per arch | Peak perf on Ampere / Hopper / Blackwell specifically     |

------

## Learning Path (学习路径)

```
Week 1-2:  Tier-1 — Layout algebra fundamentals
                     make_layout, composition, coalesce, complement
                     Tensor creation & slicing
                     zipped_divide, local_tile, local_partition
                     Elementwise kernels with TV layout
                     Swizzle for bank-conflict-free SMEM

Week 3-4:  Tier-2 — Copy engine (make_copy_atom, make_tiled_copy_tv)
                     MMA engine (make_tiled_mma, cute.gemm)
                     Partition workflow (get_slice, partition_S/D, partition_A/B/C)
                     Synchronization (syncthreads, fences)
                     Mixed precision (.to(dtype))
                     Debugging (cute.printf, print_tensor)

Week 5-6:  Tier-3 — Choose your target architecture:
                     Ampere: cp.async pipeline + MMA atoms
                     Hopper: TMA + warpgroup MMA + pipeline + cluster
                     Blackwell: tcgen05 + TMEM + block-scaled MMA
```

------

## CuTe DSL vs Triton — Concept Mapping (概念对照)

| Concept (概念)   | Triton                          | CuTe DSL                                  |
| ---------------- | ------------------------------- | ----------------------------------------- |
| Kernel decorator | `@triton.jit`                   | `@cute.kernel` + `@cute.jit`              |
| Block ID         | `tl.program_id(0)`              | `cute.arch.block_idx()[0]`                |
| Thread ID        | Implicit (block-level)          | `cute.arch.thread_idx()[0]`               |
| Load data        | `tl.load(ptr+offs, mask)`       | `tensor[idx].load()`                      |
| Store data       | `tl.store(ptr+offs, val, mask)` | `tensor[idx].store(val)`                  |
| Matrix multiply  | `tl.dot(a, b)`                  | `cute.gemm(tiled_mma, rA, rB, rC)`        |
| Tiling           | `BLOCK_SIZE` + manual offsets   | `cute.zipped_divide(tensor, tile)`        |
| Shared memory    | Auto-managed                    | Explicit: `make_tensor(smem_ptr, layout)` |
| Async copy       | `num_stages` parameter          | `cp_async_commit/wait` or TMA             |
| Bank conflicts   | Auto-managed                    | Explicit: `cute.Swizzle(M, B, S)`         |
| Autotuning       | `@triton.autotune`              | Manual or custom autotuner                |
| Boundary check   | `mask = offs < n`               | `pred=` in copy, or manual                |
| Type cast        | `.to(tl.float16)`               | `.to(cutlass.Float16)`                    |

------

## API Count by Tier and Category (各层级各类别 API 统计)

### Tier-1: Foundation (~26 APIs)

| Category                     | Count | APIs                                                         |
| ---------------------------- | ----- | ------------------------------------------------------------ |
| Decorators & Launch (装饰器) | 4     | `@kernel`, `@jit`, `compile`, `launch`                       |
| Thread Indexing (线程索引)   | 5     | `thread_idx`, `block_idx`, `block_dim`, `grid_dim`, `cluster_idx/dim` |
| Layout System (布局系统)     | 7     | `make_layout`, `make_layout_tv`, `size`, `coalesce`, `composition`, `complement`, `logical_product` |
| Tensor APIs (张量)           | 5     | `make_tensor`, `.load/.store`, slicing, `make_fragment_like`, `TensorSSA` |
| Tiling (分块)                | 4     | `zipped_divide`, `local_tile`, `local_partition`, `group_modes` |
| Swizzle                      | 1     | `cute.Swizzle`                                               |

### Tier-2: Performance (~22 APIs)

| Category               | Count | APIs                                                         |
| ---------------------- | ----- | ------------------------------------------------------------ |
| Copy Engine (拷贝引擎) | 8     | `cute.copy` ×2, `make_copy_atom`, `make_tiled_copy_tv`, `get_slice`, `partition_S/D`, `cp_async_commit/wait` |
| MMA Engine (MMA 引擎)  | 5     | `make_tiled_mma`, `cute.gemm`, `get_slice`, `partition_A/B/C`, `partition_shape_*` |
| Synchronization (同步) | 3     | `syncthreads`, `fence_view_async_shared`, `get_dyn_smem_ptr` |
| Type System (类型系统) | 3     | Numeric types, `.to(dtype)`, TensorSSA arithmetic            |
| Debugging (调试)       | 3     | `cute.printf`, `cute.print_tensor`, `print(layout)`          |

### Tier-3: Architecture (~16+ per arch)

| Architecture      | Count | Key APIs                                                     |
| ----------------- | ----- | ------------------------------------------------------------ |
| Ampere (SM80)     | ~6    | `CpAsyncOp`, `cp_async_commit/wait`, Ampere MMA atoms        |
| Hopper (SM90)     | ~8    | `tma_partition`, `warpgroup.*`, `Pipeline`, `warpgroup_fence`, Cluster |
| Blackwell (SM100) | ~8    | `tcgen05.MmaOp/CopyOp`, `tmem.*`, Block-Scaled, `set/get(Field)` |
| Control Flow      | 3     | `cutlass.range`, `cutlass.const_expr`, `get_dyn_smem_size`   |

------

## Combination Cheat Sheet (组合速查)

| Pattern (模式)                  | Tier-1 APIs                                                  | + Tier-2                                                     | + Tier-3                      |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| **Elementwise (逐元素)**        | `kernel/jit` + `thread/block_idx` + `make_tensor` + `zipped_divide` + `.load/.store` | —                                                            | —                             |
| **Vectorized Elem (向量化)**    | + `make_layout_tv` + slicing `[None, (mi,ni)]`               | —                                                            | —                             |
| **Tiled Copy (分块拷贝)**       | + `Swizzle` + `make_tensor(smem)`                            | `make_copy_atom` + `make_tiled_copy_tv` + `partition_S/D` + `cute.copy` | `CpAsyncOp` / TMA             |
| **GEMM (矩阵乘法)**             | Layout + Tensor + Tiling                                     | `make_tiled_mma` + `cute.gemm` + `partition_A/B/C` + `syncthreads` | Architecture MMA atoms        |
| **Pipelined GEMM (流水线)**     | All above                                                    | + multi-stage SMEM                                           | `cp_async_*` / Pipeline / TMA |
| **FlashAttention (闪存注意力)** | All above                                                    | All above                                                    | Warpgroup + Cluster + TMA     |

------

## Performance Optimization Priority (性能优化优先级)

| #    | Action                      | Impact                     | Tier     |
| ---- | --------------------------- | -------------------------- | -------- |
| 1    | **Master Layout algebra**   | Foundation for everything  | Tier-1   |
| 2    | **Swizzle SMEM layouts**    | Up to 32x SMEM bandwidth   | Tier-1   |
| 3    | **Use hardware Copy Atoms** | Async / zero-thread copies | Tier-2+3 |
| 4    | **Use hardware MMA Atoms**  | Tensor Core throughput     | Tier-2+3 |
| 5    | **Pipeline stages**         | Hide memory latency        | Tier-3   |
| 6    | **Mixed precision**         | 2x compute throughput      | Tier-2   |
| 7    | **Cluster (Hopper+)**       | Inter-CTA cooperation      | Tier-3   |
| 8    | **TMEM (Blackwell)**        | Dedicated accumulator      | Tier-3   |

------

## Key Insight (核心洞察)

CuTe DSL's power comes from **three pillars**:

1.  **Layout Algebra (布局代数)** — `make_layout`, `composition`, `coalesce`, `Swizzle` — master these and everything else clicks
2.  **Copy Atoms (拷贝原子)** — `make_copy_atom` + `make_tiled_copy_tv` — control how data flows between memory levels
3.  **MMA Atoms (矩阵乘加原子)** — `make_tiled_mma` + `cute.gemm` — control how Tensor Cores execute

Everything else is plumbing to connect these three pillars.
