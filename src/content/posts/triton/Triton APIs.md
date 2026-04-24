---
title: "Triton APIs"
published: 2026-04-20
description: "Triton APIs"
image: ""
tags: ["triton","Triton APIs"]
category: triton
draft: false
lang: ""
---

# Triton APIs— Index (索引)

## Overview (概览)

==**Triton** is a GPU programming language (GPU编程语言) and compiler (编译器).== It is used to write custom GPU kernels (自定义GPU算子) using Python.

Triton's API is organized into 3 tiers. Master them in order — each tier builds on the previous.

$$ \text{Total APIs} \approx 20 + 12 + 8 = 40 \text{ core APIs} $$

------

## File Structure (文件结构)

| File                               | Tier                       | APIs | What You Can Build                            |
| ---------------------------------- | -------------------------- | ---- | --------------------------------------------- |
| `Triton_Tier1_Essential_APIs.md`   | Tier-1: Essential (必备)   | ~20  | Any kernel: ReLU, Softmax, basic GEMM         |
| `Triton_Tier2_Performance_APIs.md` | Tier-2: Performance (性能) | ~12  | Autotuned, pipelined, mixed-precision kernels |
| `Triton_Tier3_Advanced_APIs.md`    | Tier-3: Advanced (高级)    | ~8   | FlashAttention, custom fused ops, peak perf   |

------

## Learning Path (学习路径)

```
Week 1:  Tier-1 — Elementwise kernels (ReLU, GELU, SiLU)
                   Row reduction kernels (Softmax, LayerNorm)
                   Basic GEMM with manual offsets

Week 2:  Tier-2 — Add autotune to all kernels
                   Rewrite GEMM with block pointers
                   Mixed precision (FP16 load, FP32 acc)
                   Histogram with atomics

Week 3:  Tier-3 — Fused multi-op kernels (LayerNorm + GELU)
                   RoPE with sin/cos
                   Debugging workflow
                   Profile with Nsight Compute
```

------

## API Count Summary (API 统计)

### Tier-1: Essential (~20 APIs)

| Category                | APIs                                      |
| ----------------------- | ----------------------------------------- |
| Indexing (索引)         | `program_id`, `num_programs`, `arange`    |
| Memory (内存)           | `load`, `store`                           |
| Initialization (初始化) | `zeros`, `full`                           |
| Control Flow (控制流)   | `where`                                   |
| Math (数学)             | `exp`, `log`, `sqrt`, `abs`, `sin`, `cos` |
| Elementwise (逐元素)    | `minimum`, `maximum`                      |
| Reduction (归约)        | `sum`, `max`, `min`                       |
| Compute (计算)          | `dot`                                     |
| Utility (工具)          | `cdiv`, `constexpr`                       |

### Tier-2: Performance (~12 APIs)

| Category                | APIs                                     |
| ----------------------- | ---------------------------------------- |
| Tuning (调优)           | `autotune`, `num_warps`, `num_stages`    |
| Atomics (原子操作)      | `atomic_add`, `atomic_max`, `atomic_min` |
| Block Pointers (块指针) | `make_block_ptr`, `advance`              |
| Transform (变换)        | `trans`, `broadcast_to`                  |
| Type (类型)             | `.to(dtype)`                             |

### Tier-3: Advanced (~8 APIs)

| Category        | APIs                               |
| --------------- | ---------------------------------- |
| Shape (形状)    | `reshape`, `flip`, `join`, `split` |
| Debug (调试)    | `device_print`                     |
| Hardware (硬件) | `inline_asm_elementwise`           |

------

## Combination Cheat Sheet (组合速查)

| Pattern (模式)             | Tier-1 APIs                                | + Tier-2                          | + Tier-3                 |
| -------------------------- | ------------------------------------------ | --------------------------------- | ------------------------ |
| **Elementwise (逐元素)**   | `pid` + `arange` + `load` + math + `store` | `autotune`                        | —                        |
| **Row Reduction (行归约)** | + `max` + `sum` + `exp`                    | `autotune`                        | —                        |
| **GEMM (矩阵乘法)**        | + `zeros` + `dot`                          | `block_ptr` + `advance` + `.to()` | —                        |
| **Histogram (直方图)**     | + `minimum`/`maximum`                      | `atomic_add`                      | —                        |
| **Transpose (转置)**       | `load` + `store`                           | `trans`                           | —                        |
| **RoPE (位置编码)**        | `sin` + `cos` + `exp` + `log`              | —                                 | `reshape`                |
| **Fused LayerNorm+GELU**   | all Tier-1 reductions + math               | `autotune`                        | fusion pattern           |
| **FlashAttention**         | all Tier-1                                 | all Tier-2                        | `reshape` + `inline_asm` |

------

## Performance Optimization Priority (性能优化优先级)

| #    | Action                        | Impact                                          |
| ---- | ----------------------------- | ----------------------------------------------- |
| 1    | **Fuse ops** into one kernel  | 2-5x (eliminate GMEM round-trips)               |
| 2    | **`@triton.autotune`**        | 1.2-2x (find best config)                       |
| 3    | **Mixed precision** FP16+FP32 | 1.5-2x (Tensor Core throughput)                 |
| 4    | **Block pointers** for 2D     | 1.1-1.3x (better codegen)                       |
| 5    | **Minimize atomics**          | Varies (avoid serialization)                    |
| 6    | **Coalesced memory access**   | 1.5-10x (adjacent threads → adjacent addresses) |
