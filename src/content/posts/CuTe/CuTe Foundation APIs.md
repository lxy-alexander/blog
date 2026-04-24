---
title: "Triton Foundation APIs"
published: 2026-04-20
description: "CuTe Foundation APIs"
image: ""
tags: ["CuTe","CuTe Foundation APIs"]
category: CuTe
draft: false
lang: ""
---

# I. CuTe DSL Tier-1: Foundation APIs (基础 API)

These ~25 APIs are required for writing **any** CuTe DSL Kernel (核函数) — decorators, thread indexing, Layout algebra, Tensor creation, and tiling.

$$ \text{Layout}: \text{Logical Coordinate (逻辑坐标)} \xrightarrow{(\text{Shape}, \text{Stride})} \text{Physical Index (物理索引)} $$

------

# II. Decorators & Launch (装饰器与启动)

## 1. `@cute.kernel` — Define a GPU Kernel (定义 GPU 核函数)

Marks a function to run on the GPU Device (设备端) — equivalent to CUDA's `__global__` keyword; every CuTe DSL program needs at least one.

```python
import cutlass
import cutlass.cute as cute

cutlass.cuda.initialize_cuda_context()

@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello from GPU thread 0!")
```

------

## 2. `@cute.jit` — Define a JIT Host Function (定义 JIT 主机函数)

Marks a function for JIT Compilation (即时编译) on the CPU Host (主机端) — this is where you set up data, create Tensors, configure grid/block, and launch Kernels.

```python
@cute.jit
def run_hello():
    hello_kernel().launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )

# Execute
run_hello()
```

------

## 3. `cute.compile(fn)` — Ahead-of-Time Compilation (预编译)

Pre-compiles a `@cute.jit` function so it can be reused without re-JITing — useful for CUDA Graphs (CUDA 图) and production deployment.

```python
# Compile once
run_hello_compiled = cute.compile(run_hello)

# Run many times without recompilation
run_hello_compiled()
run_hello_compiled()
```

------

## 4. `kernel().launch(grid, block, cluster, smem, stream)` — Launch a Kernel (启动核函数)

Launches the Kernel on the GPU with specified Grid (网格) and Block (线程块) dimensions; optional Cluster (集群, Hopper+), Shared Memory (共享内存) size, and CUDA Stream (流).

```python
@cute.jit
def launch_example(mA: cute.Tensor, mB: cute.Tensor):
    num_blocks = cute.size(mA) // 256
    my_kernel(mA, mB).launch(
        grid=(num_blocks, 1, 1),    # Grid dimensions
        block=(256, 1, 1),          # Threads per block
        # cluster=(2, 1, 1),        # Hopper cluster (optional)
        # smem=49152,               # Dynamic SMEM bytes (optional)
    )
```

------

# III. Thread & Block Indexing (线程与块索引)

## 5. `cute.arch.thread_idx()` — Get Thread Index (获取线程索引)

Returns `(tidx, tidy, tidz)` — the 3D Thread Index (线程索引) within the current Block, equivalent to CUDA's `threadIdx`.

```python
@cute.kernel
def thread_demo(out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    out[tidx] = tidx  # Each thread writes its own index
```

------

## 6. `cute.arch.block_idx()` — Get Block Index (获取块索引)

Returns `(bidx, bidy, bidz)` — the 3D Block Index (块索引) within the Grid, equivalent to CUDA's `blockIdx`.

```python
@cute.kernel
def block_demo(out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    global_idx = bidx * bdim + tidx
    out[global_idx] = bidx  # Store block ID
```

------

## 7. `cute.arch.block_dim()` — Get Block Dimensions (获取块维度)

Returns `(bdimx, bdimy, bdimz)` — the number of Threads (线程) per Block in each dimension, equivalent to CUDA's `blockDim`.

------

## 8. `cute.arch.grid_dim()` — Get Grid Dimensions (获取网格维度)

Returns `(gdimx, gdimy, gdimz)` — the number of Blocks per Grid in each dimension, equivalent to CUDA's `gridDim`.

------

## 9. `cute.arch.cluster_idx()` / `cute.arch.cluster_dim()` — Hopper Cluster Indexing (Hopper 集群索引)

For SM90+ architectures — returns Cluster Index (集群索引) and Cluster Dimensions (集群维度) for multi-CTA cooperation.

```python
@cute.kernel
def cluster_demo():
    cidx, _, _ = cute.arch.cluster_idx()   # Which cluster
    cdim, _, _ = cute.arch.cluster_dim()   # Cluster size
    # Used for inter-CTA communication on Hopper
```

------

# IV. Layout System — The Core of CuTe (布局系统 — CuTe 的核心)

Layout = `(Shape, Stride)` — maps logical coordinates to physical memory indices. This is the single most important concept in CuTe.

$$ \text{index}(\text{coord}) = \sum_{i} \text{coord}_i \times \text{stride}_i $$

## 10. `cute.make_layout(shape, stride)` — Create a Layout (创建布局)

Creates a Layout (布局) from a Shape (形状) and optional Stride (步长) — the fundamental building block.

```python
@cute.jit
def layout_demo():
    # Row-major 4×8: element (i,j) -> index i*8 + j
    layout_rm = cute.make_layout((4, 8), stride=(8, 1))
    print(f"Row-major: {layout_rm}")
    # (4,8):(8,1)

    # Column-major 4×8: element (i,j) -> index i + j*4
    layout_cm = cute.make_layout((4, 8), stride=(1, 4))
    print(f"Col-major: {layout_cm}")
    # (4,8):(1,4)

    # Default stride (compact, row-major)
    layout_default = cute.make_layout((4, 8))
    print(f"Default: {layout_default}")

    # Hierarchical / nested shapes
    layout_nested = cute.make_layout(((2, 4), 8))
    print(f"Nested: {layout_nested}")

layout_demo()
```

------

## 11. `cute.make_layout_tv(thr_layout, val_layout)` — Create Thread-Value Layout (创建线程-值布局)

Creates a combined Thread-Value Layout (线程-值布局) that maps `(thread_id, value_id)` to a logical tile — the key abstraction for partitioning work among threads.

```python
@cute.jit
def tv_layout_demo():
    # 4 rows × 32 cols of threads (128 threads total)
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    # Each thread handles 4 rows × 8 cols of values
    val_layout = cute.make_layout((4, 8), stride=(8, 1))

    # Combined: maps (tid, vid) -> (TileM=16, TileN=256)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler shape: {tiler_mn}")       # Tile each block handles
    print(f"TV Layout: {tv_layout}")        # Thread-Value mapping
    print(f"Threads: {cute.size(tv_layout, mode=[0])}")

tv_layout_demo()
```

------

## 12. `cute.size(layout_or_tensor, mode)` — Get Size of a Mode (获取模式大小)

Returns the number of elements along a specific Mode (模式/维度) — essential for computing Grid dimensions and loop bounds.

```python
@cute.jit
def size_demo():
    layout = cute.make_layout((4, 8, 16))
    total = cute.size(layout)               # 4 * 8 * 16 = 512
    dim0 = cute.size(layout, mode=[0])      # 4
    dim1 = cute.size(layout, mode=[1])      # 8
    dim2 = cute.size(layout, mode=[2])      # 16
    print(f"Total: {total}, Dims: {dim0}, {dim1}, {dim2}")

size_demo()
```

------

## 13. `cute.coalesce(layout)` — Flatten Compatible Modes (合并兼容模式)

Merges adjacent modes with compatible strides into a single mode — simplifies Layouts for more efficient memory access.

```python
@cute.jit
def coalesce_demo():
    # ((2,4), 8) with contiguous strides -> can be coalesced to (8, 8)
    layout = cute.make_layout(((2, 4), 8), stride=((1, 2), 8))
    coalesced = cute.coalesce(layout)
    print(f"Before: {layout}")
    print(f"After:  {coalesced}")

coalesce_demo()
```

------

## 14. `cute.composition(layout_a, layout_b)` — Compose Two Layouts (组合两个布局)

Composes Layout B through Layout A — the result maps B's coordinates through A's index function; enables complex Layout transformations.

```python
@cute.jit
def compose_demo():
    layout_a = cute.make_layout((4, 8), stride=(8, 1))
    layout_b = cute.make_layout((2, 2), stride=(1, 2))
    result = cute.composition(layout_a, layout_b)
    print(f"A: {layout_a}")
    print(f"B: {layout_b}")
    print(f"A ∘ B: {result}")

compose_demo()
```

------

## 15. `cute.complement(layout, size)` — Compute Layout Complement (计算布局补)

Produces the Layout that tiles the remaining space not covered by the input Layout — used internally for partitioning and tiling computations.

```python
@cute.jit
def complement_demo():
    layout = cute.make_layout((4,), stride=(2,))
    comp = cute.complement(layout, 16)
    print(f"Layout: {layout}")
    print(f"Complement: {comp}")

complement_demo()
```

------

## 16. `cute.logical_product(layout_a, layout_b)` — Layout Logical Product (布局逻辑乘积)

Replicates Layout A across Layout B — used to tile a small pattern across a larger space.

```python
@cute.jit
def product_demo():
    atom = cute.make_layout((2, 2), stride=(1, 2))
    tile = cute.make_layout((4, 4))
    result = cute.logical_product(atom, tile)
    print(f"Atom: {atom}")
    print(f"Tiled: {result}")

product_demo()
```

------

# V. Tensor APIs (张量 API)

Tensor = Pointer + Layout — represents data in a specific memory space with a specific access pattern.

## 17. `cute.make_tensor(ptr, layout)` — Create a Tensor (从指针和布局创建张量)

Wraps a memory Pointer (指针) with a Layout (布局) to create a Tensor — the primary way to represent GPU data.

```python
@cute.jit
def tensor_demo(ptr_a: cute.Pointer[cutlass.Float16]):
    layout = cute.make_layout((128, 64), stride=(64, 1))
    tensor_a = cute.make_tensor(ptr_a, layout)
    print(f"Tensor: {tensor_a}")
    print(f"Shape: {tensor_a.shape}")
```

------

## 18. `tensor.load()` / `tensor.store(val)` — Load and Store Data (加载和存储数据)

Reads data from or writes data to the Tensor's memory — the CuTe equivalent of Triton's `tl.load`/`tl.store`.

```python
@cute.kernel
def load_store_kernel(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    # Load from source
    val = src[tidx].load()
    # Compute
    val = val * 2.0
    # Store to destination
    dst[tidx].store(val)
```

------

## 19. `tensor[coord]` / `tensor[None, (mi, ni)]` — Tensor Slicing (张量切片)

Index into a Tensor using coordinates — `None` selects all elements in a mode, `(mi, ni)` selects a specific tile coordinate.

```python
@cute.kernel
def slice_kernel(tiled_tensor: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Select block's tile (mode-1) — all values within that tile (mode-0)
    block_tile = tiled_tensor[None, bidx]

    # Select this thread's element within the tile
    val = block_tile[tidx].load()
    # ... compute ...
```

------

## 20. `cute.make_fragment_like(tensor)` — Create Register Fragment (创建寄存器片段)

Allocates a register-resident copy matching a Tensor's shape — used to hold data in fast Registers (寄存器) during computation.

```python
@cute.kernel
def fragment_demo(smem_tensor: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    # Create register fragment matching shape
    frag = cute.make_fragment_like(smem_tensor[tidx])
    # Copy SMEM -> registers
    cute.copy(smem_tensor[tidx], frag)
    # Compute on registers (fast!)
```

------

## 21. `TensorSSA` — Value-Semantic Tensor (值语义张量)

An immutable Tensor (不可变张量) returned by `.load()` — lives in registers, supports direct arithmetic (`+`, `-`, `*`, `/`).

```python
@cute.kernel
def ssa_demo(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # .load() returns TensorSSA
    a_val = gA[(None, (bidx,))][tidx].load()   # TensorSSA
    b_val = gB[(None, (bidx,))][tidx].load()   # TensorSSA
    c_val = a_val + b_val                       # Direct arithmetic
    gC[(None, (bidx,))][tidx].store(c_val)
```

------

# VI. Tiling & Partitioning (分块与分区)

These APIs divide data among Blocks and Threads — the heart of parallel GPU programming in CuTe.

## 22. `cute.zipped_divide(tensor, tiler)` — Divide Tensor into Tiles (将张量分割为瓦片)

Splits a Tensor into tiles of the given shape, producing `((Tile), (NumTiles))` — mode-0 is one tile, mode-1 indexes across tiles.

```python
@cute.jit
def zipped_divide_demo(mA: cute.Tensor):
    block_shape = (128, 64)
    # Result shape: ((128, 64), (M/128, N/64))
    tiled_A = cute.zipped_divide(mA, block_shape)
    print(f"Tiled: {tiled_A}")
    # Mode-0: one tile (128×64)
    # Mode-1: tile index (grid of tiles)
```

------

## 23. `cute.local_tile(tensor, tiler, coord)` — Extract a Single Tile (提取单个瓦片)

Shortcut for `zipped_divide` + indexing — divides then immediately extracts the tile at the given Coordinate (坐标).

$$ \text{local_tile}(T, \text{tiler}, \text{coord}) = \text{zipped_divide}(T, \text{tiler})[\text{coord}] $$

```python
@cute.kernel
def local_tile_kernel(mA: cute.Tensor):
    bidx, bidy, _ = cute.arch.block_idx()
    # Directly extract this block's (128×64) tile
    block_tile = cute.local_tile(mA, (128, 64), (bidx, bidy))
    # block_tile has shape (128, 64)
```

------

## 24. `cute.local_partition(tensor, layout, thread_idx)` — Partition Among Threads (在线程间分区)

Divides a tile among threads according to a Thread Layout (线程布局) — each thread gets its unique slice.

```python
@cute.kernel
def partition_kernel(block_tile: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    # Each of 128 threads gets its portion of block_tile
    my_slice = cute.local_partition(block_tile, thr_layout, tidx)
```

------

## 25. `cute.group_modes(tensor, start, end)` — Group Tensor Modes (合并张量模式)

Flattens multiple consecutive Modes (模式) into a single mode — used to reshape multi-dimensional tiles for Copy/MMA operations.

```python
@cute.jit
def group_demo(tensor_3d: cute.Tensor):
    # If tensor_3d has shape (A, B, C), group modes 0..1:
    # Result shape: (A*B, C)
    grouped = cute.group_modes(tensor_3d, 0, 2)
    print(f"Grouped: {grouped}")
```

------

# VII. Swizzle — Bank Conflict Avoidance (Swizzle — 避免 Bank 冲突)

## 26. `cute.Swizzle(MBase, BBits, SShift)` — Memory Access Pattern Optimization (内存访问模式优化)

Creates a Swizzle (交错) pattern to avoid Shared Memory Bank Conflicts (共享内存 Bank 冲突) — critical for peak SMEM bandwidth.

```python
@cute.jit
def swizzle_demo():
    # MBase=3: keep 3 LSBs constant
    # BBits=2: 2-bit mask
    # SShift=3: shift mask by 3
    swz = cute.Swizzle(3, 2, 3)

    # Apply to SMEM layout for bank-conflict-free access
    smem_layout = cute.make_layout((32, 64), stride=(64, 1))
    swizzled = cute.composition(swz, smem_layout)
    print(f"Swizzled: {swizzled}")
```

**Why it matters:** Without Swizzle, 32 threads in a Warp accessing the same SMEM bank serialize to 32 sequential accesses. With Swizzle, each thread hits a different bank → 32x bandwidth.

------

# VIII. Tier-1 Combination Patterns (Tier-1 组合模式)

## 1. Elementwise Kernel (逐元素核函数)

**APIs used:** `@cute.kernel` + `@cute.jit` + `thread_idx` + `block_idx` + `block_dim` + `make_tensor` + `zipped_divide` + `.load()` + `.store()` + `launch`

```python
@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gid = bidx * bdim + tidx
    m, n = gA.shape[1]
    ni = gid % n
    mi = gid // n
    a = gA[(None, (mi, ni))].load()
    b = gB[(None, (mi, ni))].load()
    gC[(None, (mi, ni))] = a + b

@cute.jit
def run_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tiled_A = cute.zipped_divide(mA, (1, 4))
    tiled_B = cute.zipped_divide(mB, (1, 4))
    tiled_C = cute.zipped_divide(mC, (1, 4))
    grid = (cute.size(tiled_A, mode=[1]), 1, 1)
    add_kernel(tiled_A, tiled_B, tiled_C).launch(grid=grid, block=(256, 1, 1))
```

## 2. Vectorized Elementwise with TV Layout (向量化逐元素 + TV 布局)

**APIs used:** `make_layout_tv` + `zipped_divide` + Tensor slicing `[None, (mi, ni)]` + `.load()` + `.store()`

The idiomatic CuTe pattern — tile with `zipped_divide`, map threads with `make_layout_tv`, vectorize with slicing.

```python
@cute.jit
def vectorized_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    grid = (cute.size(gA, mode=[1]), 1, 1)
    block = (cute.size(tv_layout, mode=[0]), 1, 1)
    vectorized_add_kernel(gA, gB, gC).launch(grid=grid, block=block)
```

------

# IX. Tier-1 Quick Reference (速查表)

| #    | API                            | Purpose (用途)                           | CUDA/Triton Equivalent |
| ---- | ------------------------------ | ---------------------------------------- | ---------------------- |
| 1    | `@cute.kernel`                 | GPU kernel decorator (GPU 核函数装饰器)  | `@triton.jit`          |
| 2    | `@cute.jit`                    | Host JIT function (主机 JIT 函数)        | Python launch code     |
| 3    | `cute.compile`                 | Pre-compile (预编译)                     | —                      |
| 4    | `.launch(grid, block)`         | Launch kernel (启动核函数)               | `kernel[grid](...)`    |
| 5    | `cute.arch.thread_idx()`       | Thread index (线程索引)                  | `threadIdx` / implicit |
| 6    | `cute.arch.block_idx()`        | Block index (块索引)                     | `tl.program_id`        |
| 7    | `cute.arch.block_dim()`        | Block size (块大小)                      | —                      |
| 8    | `cute.arch.grid_dim()`         | Grid size (网格大小)                     | `tl.num_programs`      |
| 9    | `cute.arch.cluster_idx/dim`    | Cluster (集群, SM90+)                    | —                      |
| 10   | `cute.make_layout`             | Create layout (创建布局)                 | Manual offsets         |
| 11   | `cute.make_layout_tv`          | Thread-Value layout (线程-值布局)        | —                      |
| 12   | `cute.size`                    | Get mode size (获取模式大小)             | `.shape`               |
| 13   | `cute.coalesce`                | Flatten modes (合并模式)                 | —                      |
| 14   | `cute.composition`             | Compose layouts (组合布局)               | —                      |
| 15   | `cute.complement`              | Layout complement (布局补)               | —                      |
| 16   | `cute.logical_product`         | Layout product (布局乘积)                | —                      |
| 17   | `cute.make_tensor`             | Pointer+Layout→Tensor (创建张量)         | `tl.load` address      |
| 18   | `.load()` / `.store()`         | Memory access (内存访问)                 | `tl.load` / `tl.store` |
| 19   | `tensor[coord]` / `[None, ()]` | Slicing (切片)                           | Offset arithmetic      |
| 20   | `cute.make_fragment_like`      | Register fragment (寄存器片段)           | `tl.zeros`             |
| 21   | `TensorSSA`                    | Value tensor (值张量)                    | Triton block values    |
| 22   | `cute.zipped_divide`           | Tile tensor (分块张量)                   | Manual tiling          |
| 23   | `cute.local_tile`              | Extract one tile (提取单块)              | `pid * BLOCK + arange` |
| 24   | `cute.local_partition`         | Thread partition (线程分区)              | —                      |
| 25   | `cute.group_modes`             | Group modes (合并模式)                   | —                      |
| 26   | `cute.Swizzle`                 | Bank-conflict avoidance (避免 Bank 冲突) | Auto in Triton         |
