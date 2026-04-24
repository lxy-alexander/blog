---
title: "Triton Performance APIs"
published: 2026-04-20
description: "Triton Performance APIs"
image: ""
tags: ["CuTe","Triton Performance APIs"]
category: CuTe
draft: false
lang: ""
---

# I. CuTe DSL Tier-2: Performance APIs (性能关键 API)

These ~20 APIs take your Kernel from "works" to "peak performance" — Copy Engine (拷贝引擎), MMA Engine (矩阵乘加引擎), Synchronization (同步), Type System (类型系统), and Debugging (调试).

------

# II. Copy Engine — Moving Data Between Memory Levels (拷贝引擎 — 在内存层级间移动数据)

Data flows: **GMEM → SMEM → RMEM** (and back). CuTe provides hardware-specific Copy Atoms (拷贝原子) for each path.

$$ \text{GMEM (全局内存)} \xrightarrow{\text{TiledCopy}} \text{SMEM (共享内存)} \xrightarrow{\text{TiledCopy}} \text{RMEM (寄存器)} $$

## 1. `cute.copy(src, dst)` — Basic Copy (基本拷贝)

Copies data between Tensors in different memory spaces — the simplest copy, auto-vectorized by the compiler.

```python
@cute.kernel
def basic_copy_kernel(gmem_src: cute.Tensor, rmem_dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    # Simple copy: global memory -> registers
    cute.copy(gmem_src[tidx], rmem_dst[tidx])
```

------

## 2. `cute.copy(tiled_copy, src, dst, pred)` — Tiled Copy with Atom (使用原子的分块拷贝)

Copies data using a specific hardware Copy Atom (拷贝原子) — enables vectorized loads, TMA, `cp.async`, etc; `pred` is the Predicate Tensor (谓词张量) for boundary checks.

```python
@cute.kernel
def tiled_copy_kernel(tiled_copy_a, sA, rA):
    tidx, _, _ = cute.arch.thread_idx()
    # Get this thread's copy slice
    thr_copy = tiled_copy_a.get_slice(tidx)
    tAsA = thr_copy.partition_S(sA)    # Source partition
    tArA = thr_copy.partition_D(rA)    # Destination partition
    # Execute the copy
    cute.copy(tiled_copy_a, tAsA, tArA)
```

------

## 3. `cute.make_copy_atom(CopyOp, dtype)` — Create Copy Atom (创建拷贝原子)

Creates a hardware-specific Copy Atom (拷贝原子) — selects the hardware instruction for a given copy operation and data type.

```python
@cute.jit
def copy_atom_demo():
    # Ampere: cp.async for async GMEM->SMEM
    copy_atom_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CpAsyncOp.LD_GLOBAL_ASYNC,
        cutlass.Float16
    )
    print(f"Copy atom: {copy_atom_async}")
```

**Common Copy Operations by architecture:**

| Architecture      | CopyOp                       | Path            |
| ----------------- | ---------------------------- | --------------- |
| Ampere (SM80)     | `CpAsyncOp.LD_GLOBAL_ASYNC`  | GMEM→SMEM async |
| Hopper (SM90)     | `CpAsyncBulkTensorTileG2SOp` | TMA GMEM→SMEM   |
| Blackwell (SM100) | `tcgen05.CopyOp`             | SMEM↔TMEM       |

------

## 4. `cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)` — Create Tiled Copy (创建分块拷贝)

Tiles a Copy Atom across threads and values — defines how a group of threads cooperatively copies a tile of data.

```python
@cute.jit
def tiled_copy_demo():
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CpAsyncOp.LD_GLOBAL_ASYNC,
        cutlass.Float16
    )
    # 32 threads × 4 groups = 128 threads
    thr_layout = cute.make_layout((32, 4))
    # Each thread loads 1 row × 8 elements
    val_layout = cute.make_layout((1, 8))
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    print(f"Tiled copy: {tiled_copy}")
```

------

## 5. `tiled_copy.get_slice(thread_idx)` — Get Thread's Copy Slice (获取线程的拷贝切片)

Returns a `ThrCopy` object for a specific thread — provides `partition_S` and `partition_D` for partitioning source and destination.

------

## 6. `thr_copy.partition_S(src)` / `thr_copy.partition_D(dst)` — Partition Source/Destination (分区源/目标)

Partitions source and destination Tensors according to the copy pattern — ensures each thread copies the correct elements with the correct memory access pattern.

```python
@cute.kernel
def copy_partition_demo(tiled_copy, gmem_A, smem_A):
    tidx, _, _ = cute.arch.thread_idx()
    thr_copy = tiled_copy.get_slice(tidx)

    # Partition source (GMEM) and destination (SMEM)
    tAgA = thr_copy.partition_S(gmem_A)    # This thread's GMEM source
    tAsA = thr_copy.partition_D(smem_A)    # This thread's SMEM destination

    # Copy GMEM -> SMEM
    cute.copy(tiled_copy, tAgA, tAsA)
```

------

## 7. `cute.arch.cp_async_commit_group()` — Commit Async Copies (提交异步拷贝)

Submits all pending `cp.async` operations as a group — used on Ampere for asynchronous GMEM→SMEM pipelining.

------

## 8. `cute.arch.cp_async_wait_group(n)` — Wait for Async Copies (等待异步拷贝)

Blocks until at most `n` async copy groups remain in-flight — `wait_group(0)` waits for all copies to complete.

```python
@cute.kernel
def async_pipeline_demo(tiled_copy, src, dst):
    # ... partition setup ...
    cute.copy(tiled_copy, tAsA, tBsA)       # Issue async copies
    cute.arch.cp_async_commit_group()        # Submit as a group
    cute.arch.cp_async_wait_group(0)         # Wait for completion
    cute.arch.syncthreads()                  # Sync before using data
```

------

# III. MMA Engine — Tensor Core Operations (MMA 引擎 — 张量核心操作)

MMA (Matrix Multiply-Accumulate, 矩阵乘加) Atoms directly invoke Tensor Core instructions.

$$ C_{acc} \mathrel{+}= A \times B \quad \text{via Tensor Cores (张量核心)} $$

## 9. `cute.make_tiled_mma(mma_op)` — Create Tiled MMA (创建分块矩阵乘加)

Creates a `TiledMma` that tiles a hardware MMA Atom across threads — the central API for Tensor Core GEMM.

```python
@cute.jit
def mma_setup():
    # Ampere FP16 Tensor Core: 16×8×16 tile
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaAtomOp.MMA_16x8x16_F32_F16_F16_F32
    )
    print(f"MMA tile shape (M,N,K): {tiled_mma.shape_mnk}")
    print(f"Threads per MMA: {tiled_mma.size}")
```

**Common MMA operations:**

| Architecture      | MmaAtomOp                       | Types         | Tile         |
| ----------------- | ------------------------------- | ------------- | ------------ |
| Ampere (SM80)     | `MMA_16x8x16_F32_F16_F16_F32`   | FP16→FP32     | 16×8×16      |
| Ampere (SM80)     | `MMA_16x8x16_F32_BF16_BF16_F32` | BF16→FP32     | 16×8×16      |
| Hopper (SM90)     | `GMMA` variants                 | FP16/BF16/FP8 | 64×N×16      |
| Blackwell (SM100) | `tcgen05.MmaOp`                 | FP16/FP8/FP4  | Larger tiles |

------

## 10. `cute.gemm(tiled_mma, tCrA, tCrB, tCrC)` — Execute Tiled GEMM (执行分块矩阵乘法)

Performs the actual Matrix Multiply-Accumulate (矩阵乘加) using Tensor Cores — $C \mathrel{+}= A \times B$.

```python
@cute.kernel
def gemm_compute(tiled_mma, rA, rB, acc):
    # C += A * B using Tensor Cores
    cute.gemm(tiled_mma, rA, rB, acc)
```

------

## 11. `tiled_mma.get_slice(thread_idx)` — Get Thread's MMA Slice (获取线程的 MMA 切片)

Returns a `ThrMma` object for partitioning A, B, C Tensors according to the MMA's thread-data mapping.

------

## 12. `thr_mma.partition_A/B/C(tensor)` — MMA Partitioning (MMA 分区)

Partitions Tensors for MMA operands A, B, and accumulator C — ensures data is in the correct register layout for Tensor Cores.

```python
@cute.kernel
def mma_partition_demo(tiled_mma, sA, sB, smem_C):
    tidx, _, _ = cute.arch.thread_idx()
    thr_mma = tiled_mma.get_slice(tidx)

    # Partition A (M×K) for this thread
    tCsA = thr_mma.partition_A(sA)
    # Partition B (N×K) for this thread
    tCsB = thr_mma.partition_B(sB)
    # Partition C (M×N) accumulator for this thread
    tCrC = thr_mma.partition_C(smem_C)

    # Execute: C += A * B
    cute.gemm(tiled_mma, tCsA, tCsB, tCrC)
```

------

## 13. `tiled_mma.partition_shape_A/B/C(shape)` — Get Partitioned Shape (获取分区形状)

Returns the shape each thread needs for its MMA fragment — used to allocate register storage before computation.

```python
@cute.jit
def shape_demo():
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaAtomOp.MMA_16x8x16_F32_F16_F16_F32
    )
    # How much register space does each thread need?
    shape_A = tiled_mma.partition_shape_A((128, 64))  # M×K
    shape_B = tiled_mma.partition_shape_B((64, 128))   # N×K
    shape_C = tiled_mma.partition_shape_C((128, 128))  # M×N
    print(f"Per-thread A: {shape_A}, B: {shape_B}, C: {shape_C}")
```

------

# IV. Synchronization & Shared Memory (同步与共享内存)

## 14. `cute.arch.syncthreads()` — Block-level Barrier (块级同步屏障)

Synchronizes all threads in a Block (线程块) — equivalent to CUDA's `__syncthreads()`.

```python
@cute.kernel
def sync_demo(smem, gmem):
    tidx, _, _ = cute.arch.thread_idx()
    # Stage 1: All threads write to shared memory
    smem[tidx] = gmem[tidx].load()
    cute.arch.syncthreads()          # BARRIER: wait for all threads
    # Stage 2: Now safe to read other threads' data
    val = smem[tidx ^ 1].load()      # Read neighbor's data
```

------

## 15. `cute.arch.fence_view_async_shared()` — Async Fence (异步栅栏)

Ensures all previous async operations to Shared Memory (共享内存) are visible — used before switching from async copy to compute.

```python
@cute.kernel
def fence_demo():
    # After async copies complete
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_view_async_shared()   # Make SMEM writes visible
    cute.arch.syncthreads()               # Then sync threads
    # Now safe to read SMEM
```

------

## 16. `cute.arch.get_dyn_smem_ptr(dtype)` — Get Dynamic Shared Memory Pointer (获取动态共享内存指针)

Returns a pointer to the beginning of dynamically allocated Shared Memory — size set via `smem=` in `launch()`.

```python
@cute.kernel
def smem_kernel():
    # Get pointer to dynamic shared memory
    smem_ptr = cute.arch.get_dyn_smem_ptr(cutlass.Float16)
    # Wrap with layout to create usable tensor
    smem_layout = cute.make_layout((128, 64), stride=(64, 1))
    smem_tensor = cute.make_tensor(smem_ptr, smem_layout)
    # Use smem_tensor for staging data between GMEM and registers
```

------

# V. Type System & Arithmetic (类型系统与算术)

## 17. CuTe Numeric Types (数值类型)

| CuTe Type             | Description (描述)        | Bits | Use Case                   |
| --------------------- | ------------------------- | ---- | -------------------------- |
| `cutlass.Float16`     | Half Precision (半精度)   | 16   | Standard Tensor Core input |
| `cutlass.BFloat16`    | Brain Float (脑浮点)      | 16   | Training stability         |
| `cutlass.Float32`     | Single Precision (单精度) | 32   | Accumulation, output       |
| `cutlass.Float64`     | Double Precision (双精度) | 64   | Scientific computing       |
| `cutlass.Float8_e4m3` | FP8 E4M3                  | 8    | Hopper/Blackwell inference |
| `cutlass.Float8_e5m2` | FP8 E5M2                  | 8    | Hopper/Blackwell training  |
| `cutlass.Int32`       | 32-bit Integer (整数)     | 32   | Indices, counters          |

------

## 18. `tensor.to(dtype)` — Type Cast (类型转换)

Converts Tensor elements to a different numeric type — essential for Mixed Precision (混合精度) workflows.

```python
@cute.kernel
def mixed_precision_demo(gA_f16: cute.Tensor, gC_f16: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    # Load FP16
    val_f16 = gA_f16[tidx].load()
    # Cast to FP32 for computation
    val_f32 = val_f16.to(cutlass.Float32)
    result_f32 = val_f32 * val_f32 + val_f32
    # Cast back to FP16 for storage
    result_f16 = result_f32.to(cutlass.Float16)
    gC_f16[tidx].store(result_f16)
```

------

## 19. TensorSSA Arithmetic (张量 SSA 算术)

`TensorSSA` supports direct arithmetic operations — `+`, `-`, `*`, `/`, and scalar broadcasting.

```python
@cute.kernel
def arith_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    a = gA[tidx].load()     # TensorSSA
    b = gB[tidx].load()     # TensorSSA
    c = a + b                # Elementwise add
    c = c * 2.0              # Scalar multiply
    c = c - a                # Elementwise subtract
    gC[tidx].store(c)
```

------

# VI. Printing & Debugging (打印与调试)

## 20. `cute.printf(fmt, ...)` — Runtime GPU Print (运行时 GPU 打印)

Prints dynamic values from GPU threads at Runtime (运行时) — unlike Python `print` which only shows Compile-time (编译期) static values.

```python
@cute.kernel
def debug_kernel(tensor: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    val = tensor[tidx].load()
    cute.printf("Thread %d: value = %f\n", tidx, val)
```

------

## 21. `cute.print_tensor(tensor)` — Print Tensor Contents (打印张量内容)

Prints the full contents and layout of a Tensor — works for both compile-time layout info and runtime values.

------

## 22. `print(f"layout = {layout}")` — Compile-time Layout Print (编译期布局打印)

Python's `print` inside `@cute.jit` shows Layout shapes and strides at Compile Time (编译期) — dynamic values show as `?`.

```python
@cute.jit
def print_demo(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(f"a = {a}")   # Shows: a = ? (dynamic)
    print(f"b = {b}")   # Shows: b = 42 (static/constexpr)
```

**Debugging workflow:**

| Stage                | Tool                                 | When              |
| -------------------- | ------------------------------------ | ----------------- |
| 1. Layout inspection | `print(f"{layout}")` in `@cute.jit`  | Compile time      |
| 2. Value inspection  | `cute.printf(...)` in `@cute.kernel` | Runtime (GPU)     |
| 3. Tensor inspection | `cute.print_tensor(t)`               | Either            |
| 4. Environment vars  | `CUTE_DSL_LOG_TO_CONSOLE=1`          | Build diagnostics |
| 5. IR dump           | `CUTE_DSL_PRINT_IR=1`                | MLIR inspection   |

------

# VII. Tier-2 Combination Patterns (Tier-2 组合模式)

## 1. GMEM → SMEM → Compute Pattern (全局内存 → 共享内存 → 计算)

**APIs used:** `make_copy_atom` + `make_tiled_copy_tv` + `get_slice` + `partition_S/D` + `cute.copy` + `syncthreads`

```
1. Allocate SMEM (get_dyn_smem_ptr + make_tensor)
2. Create TiledCopy (make_copy_atom + make_tiled_copy_tv)
3. Partition source/dest (get_slice + partition_S/D)
4. Copy GMEM -> SMEM (cute.copy)
5. Barrier (syncthreads)
6. Compute from SMEM
```

## 2. GEMM Main Loop Pattern (GEMM 主循环)

**APIs used:** All Copy APIs + All MMA APIs + `Swizzle` + `syncthreads`

```
Setup:
  1. SMEM layouts with Swizzle for bank-conflict-free access
  2. TiledMma from hardware MMA atom
  3. TiledCopy for GMEM->SMEM

K-loop:
  4. cute.copy(tiled_copy, gmem_tile, smem_tile)
  5. syncthreads()
  6. cute.gemm(tiled_mma, smem_A, smem_B, acc)
  7. syncthreads()

Epilog:
  8. Store accumulator to GMEM
```

## 3. Async Pipelined GEMM — Ampere (异步流水线 GEMM — Ampere)

**APIs used:** Pattern 2 + `cp_async_commit_group` + `cp_async_wait_group` + multi-stage SMEM buffers

Overlaps loading tile K+1 with computing tile K — hides Memory Latency (内存延迟).

```python
# Conceptual pattern (simplified)
@cute.kernel
def pipelined_gemm(tiled_copy, tiled_mma, gmem_A, gmem_B, smem_A, smem_B, acc):
    # ... partition setup ...
    for k_tile in range(k_tile_count):
        # STAGE 1: Async copy next tile GMEM -> SMEM
        cute.copy(tiled_copy, gA_tile[k_tile], sA[write_stage])
        cute.copy(tiled_copy, gB_tile[k_tile], sB[write_stage])
        cute.arch.cp_async_commit_group()

        # STAGE 2: Wait for previous tile and compute
        cute.arch.cp_async_wait_group(1)   # Allow 1 group in-flight
        cute.arch.syncthreads()
        cute.gemm(tiled_mma, sA[read_stage], sB[read_stage], acc)

        # Rotate stages
        # read_stage, write_stage = write_stage, read_stage
```

------

# VIII. Tier-2 Quick Reference (速查表)

| #    | API                           | Purpose (用途)                              | Memory Path   |
| ---- | ----------------------------- | ------------------------------------------- | ------------- |
| 1    | `cute.copy(src, dst)`         | Basic copy (基本拷贝)                       | Any→Any       |
| 2    | `cute.copy(atom, src, dst)`   | Hardware-accelerated copy (硬件加速拷贝)    | Specific path |
| 3    | `cute.make_copy_atom`         | Create copy instruction (创建拷贝指令)      | —             |
| 4    | `cute.make_tiled_copy_tv`     | Tile copy across threads (在线程间分块拷贝) | —             |
| 5    | `tiled_copy.get_slice`        | Thread's copy slice (线程拷贝切片)          | —             |
| 6    | `partition_S` / `partition_D` | Source/dest partition (源/目标分区)         | —             |
| 7    | `cp_async_commit_group`       | Submit async copies (提交异步拷贝)          | GMEM→SMEM     |
| 8    | `cp_async_wait_group`         | Wait for async copies (等待异步拷贝)        | GMEM→SMEM     |
| 9    | `cute.make_tiled_mma`         | Create MMA (创建矩阵乘加)                   | Tensor Cores  |
| 10   | `cute.gemm`                   | Execute GEMM (执行矩阵乘法)                 | Registers     |
| 11   | `tiled_mma.get_slice`         | Thread's MMA slice (线程 MMA 切片)          | —             |
| 12   | `partition_A/B/C`             | MMA operand partition (MMA 操作数分区)      | —             |
| 13   | `partition_shape_A/B/C`       | Fragment shape query (片段形状查询)         | —             |
| 14   | `syncthreads`                 | Block barrier (块同步)                      | —             |
| 15   | `fence_view_async_shared`     | Async visibility fence (异步可见栅栏)       | SMEM          |
| 16   | `get_dyn_smem_ptr`            | Dynamic SMEM pointer (动态共享内存指针)     | SMEM          |
| 17   | Numeric types                 | Data types (数据类型)                       | —             |
| 18   | `.to(dtype)`                  | Type cast (类型转换)                        | Registers     |
| 19   | TensorSSA arithmetic          | `+`, `-`, `*`, `/`                          | Registers     |
| 20   | `cute.printf`                 | GPU debug print (GPU 调试打印)              | —             |
| 21   | `cute.print_tensor`           | Print tensor (打印张量)                     | —             |
| 22   | `print(layout)`               | Compile-time print (编译期打印)             | —             |
