---
title: "Triton Architecture-Specific APIs"
published: 2026-04-20
description: "CuTe Architecture-Specific APIs"
image: ""
tags: ["CuTe","CuTe Architecture-Specific APIs"]
category: CuTe
draft: false
lang: ""
---

# I. CuTe DSL Tier-3: Architecture-Specific APIs (架构特定 API)

These ~15+ APIs per architecture unlock peak performance on specific GPU generations — Ampere (SM80), Hopper (SM90), and Blackwell (SM100).

------

# II. Ampere (SM80) — `cute.nvgpu.cpasync` (异步拷贝)

Ampere introduced `cp.async` for asynchronous GMEM→SMEM copies, enabling Software Pipelining (软件流水线) without thread involvement.

## 1. `CpAsyncOp` — Async Copy Operations (异步拷贝操作)

Hardware instructions for non-blocking GMEM→SMEM data transfers.

| Operation                    | Description (描述)               |
| ---------------------------- | -------------------------------- |
| `CpAsyncOp.LD_GLOBAL_ASYNC`  | Async load 16B from GMEM to SMEM |
| `CpAsyncBulkTensorTileG2SOp` | Bulk tensor tile copy GMEM→SMEM  |

```python
@cute.jit
def ampere_copy_setup():
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CpAsyncOp.LD_GLOBAL_ASYNC,
        cutlass.Float16
    )
    thr_layout = cute.make_layout((32, 4))
    val_layout = cute.make_layout((1, 8))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
```

------

## 2. `cp_async_commit_group()` / `cp_async_wait_group(n)` — Pipeline Control (流水线控制)

Commit submits pending copies as a group; wait blocks until ≤ `n` groups remain in-flight.

```python
@cute.kernel
def ampere_pipeline_kernel(tiled_copy, gA, sA_stages, num_stages):
    tidx, _, _ = cute.arch.thread_idx()
    thr_copy = tiled_copy.get_slice(tidx)

    # Prefill pipeline: issue num_stages copies
    for s in range(num_stages):
        cute.copy(tiled_copy, thr_copy.partition_S(gA[None, s]),
                  thr_copy.partition_D(sA_stages[None, s]))
        cute.arch.cp_async_commit_group()

    # Main loop: consume one, produce one
    cute.arch.cp_async_wait_group(num_stages - 1)
    cute.arch.syncthreads()
    # ... compute on sA_stages[read_stage] ...
```

------

## 3. Ampere MMA Atoms (Ampere 矩阵乘加原子)

| MmaAtomOp                       | Input (输入) | Output (输出) | Tile (瓦片) |
| ------------------------------- | ------------ | ------------- | ----------- |
| `MMA_16x8x8_F32_F16_F16_F32`    | FP16         | FP32          | 16×8×8      |
| `MMA_16x8x16_F32_F16_F16_F32`   | FP16         | FP32          | 16×8×16     |
| `MMA_16x8x16_F32_BF16_BF16_F32` | BF16         | FP32          | 16×8×16     |
| `MMA_16x8x8_F32_TF32_TF32_F32`  | TF32         | FP32          | 16×8×8      |

```python
@cute.jit
def ampere_mma_setup():
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaAtomOp.MMA_16x8x16_F32_F16_F16_F32
    )
    print(f"Ampere MMA shape: {tiled_mma.shape_mnk}")   # (16, 8, 16)
    print(f"Threads: {tiled_mma.size}")                   # 32 (1 warp)
```

------

# III. Hopper (SM90) — TMA + Warpgroup + Cluster (TMA + 线程束组 + 集群)

Hopper introduced TMA (Tensor Memory Accelerator, 张量内存加速器), Warpgroup MMA, and Thread Block Clusters for massive throughput gains.

## 4. TMA — `cute.nvgpu.cpasync.tma_partition()` — TMA Partitioning (TMA 分区)

Partitions Tensors for TMA bulk copies — hardware-managed data transfer with zero thread overhead.

```python
@cute.kernel
def hopper_tma_demo(tma_atom, sA, gA):
    # TMA: hardware copies entire tiles without thread involvement
    tBsA, tBgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom,
        0,                             # Multicast mask
        cute.make_layout(1),           # Thread participation
        cute.group_modes(sA, 0, 3),    # SMEM grouped
        cute.group_modes(gA, 0, 3),    # GMEM grouped
    )
    # Copy is done by hardware, not by threads
```

------

## 5. Warpgroup APIs — `cute.nvgpu.warpgroup` (线程束组)

Hopper introduces Warpgroup (线程束组, 4 warps = 128 threads) for larger MMA tiles.

| API                            | Purpose (用途)                    |
| ------------------------------ | --------------------------------- |
| `warpgroup.mma_arrive()`       | Signal MMA completion to pipeline |
| `warpgroup.mma_commit_group()` | Commit pending MMA operations     |
| `warpgroup.mma_wait_group(n)`  | Wait for ≤ n MMA groups           |

```python
@cute.kernel
def hopper_mma_demo(tiled_mma, rA, rB, acc):
    # Warpgroup-level MMA on Hopper
    cute.gemm(tiled_mma, rA, rB, acc)
    cute.nvgpu.warpgroup.mma_arrive()
    cute.nvgpu.warpgroup.mma_commit_group()
    cute.nvgpu.warpgroup.mma_wait_group(0)
```

------

## 6. `cute.arch.warpgroup_fence_operand(tensor)` — Warpgroup Fence (线程束组栅栏)

Ensures ordering of Tensor Core operations within a Warpgroup — required for correct Hopper MMA pipelining.

```python
@cute.kernel
def fence_demo(tiled_mma, rA, rB, acc):
    cute.arch.warpgroup_fence_operand(acc)   # Fence before MMA
    cute.gemm(tiled_mma, rA, rB, acc)
    cute.arch.warpgroup_fence_operand(acc)   # Fence after MMA
```

------

## 7. Cluster Launch — Multi-CTA Cooperation (集群启动 — 多 CTA 协作)

Hopper Clusters allow multiple CTAs (线程块) to cooperate via Distributed Shared Memory (分布式共享内存).

```python
@cute.jit
def hopper_launch(kernel_fn, grid, block):
    kernel_fn().launch(
        grid=grid,
        block=block,
        cluster=(2, 1, 1),    # 2 CTAs per cluster cooperate
        smem=shared_mem_size,
    )
```

------

## 8. Pipeline APIs — Producer-Consumer Pipeline (生产者-消费者流水线)

Manages multi-stage async pipelines between data loading (Producer) and computation (Consumer).

```python
@cute.kernel
def hopper_pipeline_demo(pipeline, tma_atom, smem_stages, gmem, tiled_mma, acc):
    # Producer: load tiles via TMA
    for stage in range(num_stages):
        pipeline.producer_acquire()
        # TMA copy gmem[stage] -> smem[stage]
        pipeline.producer_commit()

    # Consumer: compute on loaded tiles
    for k in range(k_tiles):
        pipeline.consumer_wait()
        cute.gemm(tiled_mma, smem_A[stage], smem_B[stage], acc)
        pipeline.consumer_release()
```

------

# IV. Blackwell (SM100) — `cute.nvgpu.tcgen05` (Tensor Core Gen 5)

Blackwell introduces 5th-generation Tensor Cores with Tensor Memory (TMEM) and wider MMA tiles.

## 9. `tcgen05.MmaOp` — Blackwell MMA Operations (Blackwell 矩阵乘加操作)

5th-gen Tensor Core instructions with support for FP16, FP8, FP4, and Block-Scaled formats.

```python
@cute.jit
def blackwell_mma_setup():
    # Blackwell MMA atom
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.tcgen05.MmaOp(...)  # Architecture-specific
    )
```

------

## 10. `tcgen05.CopyOp` — TMEM Copy Operations (TMEM 拷贝操作)

Copy operations between Shared Memory and Tensor Memory (张量内存, TMEM) — a new memory level on Blackwell.

| CopyOp | Direction   | Description                        |
| ------ | ----------- | ---------------------------------- |
| `S2T`  | SMEM → TMEM | Load from shared to tensor memory  |
| `T2S`  | TMEM → SMEM | Store from tensor to shared memory |

------

## 11. `tmem.allocate()` / `tmem.retrieve_ptr()` / `tmem.wait_for_alloc()` — TMEM Management (TMEM 管理)

Allocate, access, and synchronize Tensor Memory — Blackwell's dedicated accumulator memory.

```python
@cute.kernel
def blackwell_tmem_demo(tiled_mma, acc_dtype):
    # Allocate TMEM for accumulator
    tmem = cute.nvgpu.tcgen05.tmem_allocate(...)

    # Sync: only warp 0 allocates, others must wait
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)

    # Create accumulator tensor in TMEM
    acc_tensor = cute.make_tensor(tmem_ptr, acc_layout)
```

------

## 12. Block-Scaled MMA (块缩放矩阵乘加)

Blackwell supports Block-Scaled quantized formats (FP8/FP4 with per-block scale factors).

```python
@cute.jit
def blockscaled_setup():
    # Block-scaled MMA for quantized inference
    tiled_mma = cute.nvgpu.tcgen05.make_blockscaled_trivial_tiled_mma(
        mma_op=...,
        # Block-scaled configuration
    )
```

------

## 13. `tiled_mma.set/get(Field, value)` — Runtime MMA Control (运行时 MMA 控制)

Modify runtime state of MMA Atoms — e.g., toggle accumulation mode.

```python
@cute.kernel
def mma_control_demo(tiled_mma):
    # Toggle accumulation: True = C += A*B, False = C = A*B
    tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
    cute.gemm(tiled_mma, rA, rB, acc)

    # Check current setting
    is_accumulating = tiled_mma.get(cute.nvgpu.tcgen05.Field.ACCUMULATE)
```

------

# V. Advanced Control Flow & Loop APIs (高级控制流与循环 API)

## 14. `cutlass.range(start, stop, step)` — Advanced Loop Control (高级循环控制)

Unlike Python's `range`, `cutlass.range` generates GPU code with kernel-level Unrolling (展开) and Pipelining (流水线) control.

```python
@cute.kernel
def loop_demo(data: cute.Tensor, N: int):
    # Python range: always generates runtime loop
    for i in range(N):
        pass

    # cutlass.range: supports compiler hints
    for i in cutlass.range(0, N, 1):
        val = data[i].load()
```

------

## 15. `cutlass.const_expr` — Compile-time Conditional (编译期条件)

Marks a condition as Compile-time (编译期) — the compiler evaluates it during JIT and eliminates dead branches entirely.

```python
@cute.kernel
def constexpr_demo(use_smem: cutlass.Constexpr[bool]):
    if use_smem:  # Evaluated at compile time, dead branch eliminated
        # This code is only compiled if use_smem=True
        smem_ptr = cute.arch.get_dyn_smem_ptr(cutlass.Float16)
    else:
        # This branch compiled if use_smem=False
        pass
```

------

## 16. `cute.arch.get_dyn_smem_size()` — Query Dynamic SMEM Size (查询动态共享内存大小)

Returns the size of dynamically allocated Shared Memory at runtime — useful for adaptive algorithms.

------

# VI. Tier-3 Combination Patterns (Tier-3 组合模式)

## 1. Full Ampere GEMM Pattern (完整 Ampere GEMM 模式)

**APIs:** Tier-1 (Layout + Tensor + Tiling) + Tier-2 (Copy + MMA + Sync) + Tier-3 (cpasync pipeline)

```
1. Setup:
   - SMEM layouts with Swizzle (Tier-1)
   - TiledMma from Ampere MMA atom (Tier-3)
   - TiledCopy with cp.async atom (Tier-3)

2. Pipeline prefill:
   - for s in range(num_stages):
       cute.copy(tiled_copy, gA[k=s], sA[stage=s])
       cp_async_commit_group()

3. Main K-loop:
   - cp_async_wait_group(stages-1)
   - syncthreads()
   - cute.gemm(tiled_mma, sA[read], sB[read], acc)
   - cute.copy(tiled_copy, gA[k=next], sA[write])
   - cp_async_commit_group()
   - rotate read/write stages

4. Drain pipeline:
   - cp_async_wait_group(0)
   - syncthreads()
   - final gemm

5. Epilog: store acc -> GMEM
```

## 2. Full Hopper GEMM Pattern (完整 Hopper GEMM 模式)

**APIs:** All Ampere APIs + TMA + Warpgroup + Pipeline + Cluster

```
Key differences from Ampere:
- TMA replaces thread-based cp.async → zero thread overhead for copies
- Warpgroup MMA replaces warp-level MMA → 4x larger tiles
- Pipeline API replaces manual cp_async_commit/wait
- Cluster launch enables inter-CTA cooperation
- Warp Specialization: some warps produce (load), others consume (compute)
```

## 3. Full Blackwell GEMM Pattern (完整 Blackwell GEMM 模式)

**APIs:** All Hopper APIs + tcgen05 MMA + TMEM + Block-Scaled

```
Key differences from Hopper:
- TMEM replaces register accumulator → dedicated accumulator memory
- tcgen05 MMA atoms → larger tiles, FP4/FP8 support
- Block-Scaled MMA → quantized inference with per-block scales
- S2T/T2S copies → SMEM ↔ TMEM data movement
```

------

# VII. Tier-3 Quick Reference by Architecture (各架构速查表)

## Ampere (SM80)

| API                           | Purpose (用途)          |
| ----------------------------- | ----------------------- |
| `CpAsyncOp.LD_GLOBAL_ASYNC`   | Async GMEM→SMEM copy    |
| `cp_async_commit_group()`     | Commit pending copies   |
| `cp_async_wait_group(n)`      | Wait for copies         |
| `MMA_16x8x16_F32_F16_F16_F32` | FP16 Tensor Core MMA    |
| `Swizzle` (from Tier-1)       | Bank-conflict-free SMEM |

## Hopper (SM90)

| API                                | Purpose (用途)             |
| ---------------------------------- | -------------------------- |
| `tma_partition()`                  | TMA bulk copy setup        |
| `warpgroup.mma_arrive/commit/wait` | Warpgroup MMA control      |
| `warpgroup_fence_operand()`        | MMA ordering fence         |
| `Pipeline` APIs                    | Producer-consumer pipeline |
| `cluster_idx/dim`                  | Cluster indexing           |
| Cluster launch parameter           | Multi-CTA cooperation      |

## Blackwell (SM100)

| API                                  | Purpose (用途)             |
| ------------------------------------ | -------------------------- |
| `tcgen05.MmaOp`                      | 5th-gen Tensor Core MMA    |
| `tcgen05.CopyOp` (S2T, T2S)          | SMEM ↔ TMEM copies         |
| `tmem.allocate/retrieve_ptr`         | Tensor Memory management   |
| `make_blockscaled_trivial_tiled_mma` | Block-scaled quantized MMA |
| `tiled_mma.set/get(Field, val)`      | Runtime MMA configuration  |

------

# VIII. Memory Hierarchy by Architecture (各架构内存层级)

```
Ampere (SM80):
  GMEM ──cp.async──> SMEM ──copy──> RMEM ──MMA──> RMEM

Hopper (SM90):
  GMEM ──TMA──> SMEM ──copy──> RMEM ──wgMMA──> RMEM
                  ↕ (Distributed SMEM via Cluster)

Blackwell (SM100):
  GMEM ──TMA──> SMEM ──S2T──> TMEM ──tcgen05 MMA──> TMEM ──T2S──> SMEM
                  ↕ (Distributed SMEM via Cluster)
```

| Memory Level | Ampere           | Hopper           | Blackwell                |
| ------------ | ---------------- | ---------------- | ------------------------ |
| GMEM→SMEM    | `cp.async`       | TMA              | TMA                      |
| SMEM→Compute | Manual copy→RMEM | Manual copy→RMEM | **S2T→TMEM (new!)**      |
| Accumulator  | Registers (RMEM) | Registers (RMEM) | **Tensor Memory (TMEM)** |
| Inter-CTA    | None             | Cluster DSMEM    | Cluster DSMEM            |
