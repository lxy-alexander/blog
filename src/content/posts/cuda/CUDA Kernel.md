---
title: "CUDA Kernel"
published: 2026-04-20
description: "CUDA Kernel"
image: ""
tags: ["cuda","CUDA Kernel"]
category: cuda
draft: false
lang: ""
---

# I. CUDA Kernel Learning Roadmap (CUDA 核函数编写学习路线)

You already know the Runtime API (运行时 API) — `cudaMalloc`, `cudaMemcpy`, `<<<grid, block>>>` launch. Now you need to learn **what goes inside the kernel** and **how to make it fast**.

------

# II. Phase 1: Write Correct Kernels (写出正确的 Kernel) — Week 1-2

The goal is to think in threads: every thread computes one (or a few) output elements.

## 1. Thread Indexing (线程索引) — The First Thing Every Kernel Does

Global Thread ID (全局线程 ID) = `blockIdx * blockDim + threadIdx` — this is how each thread knows which data element to process.

```cpp
// 1D indexing: the most common pattern
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {           // Boundary check — NEVER forget this
        c[idx] = a[idx] + b[idx];
    }
}

// Launch: how many blocks do we need?
int n = 1000000;
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;  // Ceiling division (上取整)
vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
```

### 1) 2D Indexing for Matrices (二维索引用于矩阵)

```cpp
__global__ void matrix_add(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;   // Row-major (行主序) linearization
        C[idx] = A[idx] + B[idx];
    }
}

// Launch with 2D grid and 2D blocks
dim3 block(16, 16);        // 256 threads per block
dim3 grid((cols + 15) / 16, (rows + 15) / 16);
matrix_add<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
```

### 2) Grid-Stride Loop (网格跨步循环) — When Data > Threads

When input is larger than `gridDim * blockDim`, each thread processes multiple elements by striding across the grid — more flexible than launching one thread per element.

```cpp
__global__ void vector_add_stride(float* a, float* b, float* c, int n) {
    int stride = blockDim.x * gridDim.x;   // Total threads in grid
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        c[idx] = a[idx] + b[idx];
    }
}

// Can launch with fewer blocks — threads loop over data
vector_add_stride<<<128, 256>>>(d_a, d_b, d_c, n);
```

------

## 2. Memory Spaces (内存空间) — Where Your Data Lives

Understanding which memory to use is the **single most important** performance decision.

```
Speed:    Registers >> Shared Memory >> L1/L2 Cache >> Global Memory
Scope:    Per-thread    Per-block       Per-SM/GPU      All threads
Size:     ~256KB/SM     48-228KB/SM     varies          16-80GB
Latency:  ~1 cycle      ~20 cycles      ~200 cycles     ~400 cycles
```

### 1) Global Memory (全局内存) — Default, Slowest

All `cudaMalloc`'d memory is Global Memory — accessible by all threads, but high latency.

```cpp
__global__ void read_global(float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] * 2.0f;  // data is in global memory
    }
}
```

### 2) Shared Memory (共享内存) — Per-Block, Fast

Declared with `__shared__` — visible to all threads in the same block, ~20x faster than global.

```cpp
__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[256];    // Shared memory: one per block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();   // BARRIER: wait for all threads to finish loading

    // Reduce within shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // Sync after each reduction step
    }

    // Thread 0 writes result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### 3) Registers (寄存器) — Per-Thread, Fastest

Local variables in a kernel are stored in Registers (寄存器) — the fastest memory, but limited per thread.

```cpp
__global__ void register_demo(float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];      // 'val' lives in a register
        val = val * val + val;    // All computation in registers — fast!
        out[idx] = val;
    }
}
```

### 4) Constant Memory (常量内存) — Read-Only, Cached

64KB limit, cached aggressively — ideal for coefficients/lookup tables read by all threads.

```cpp
__constant__ float filter[9];   // Declared at file scope

// Host code: copy to constant memory
float h_filter[9] = {1,1,1, 1,-8,1, 1,1,1};
cudaMemcpyToSymbol(filter, h_filter, sizeof(float) * 9);

__global__ void conv_kernel(float* in, float* out) {
    // All threads read same 'filter' values — constant cache broadcast
    float sum = 0;
    for (int i = 0; i < 9; i++) {
        sum += in[...] * filter[i];  // Cached, fast
    }
}
```

------

## 3. Synchronization Basics (基本同步)

### 1) `__syncthreads()` — Block-Level Barrier (块级同步屏障)

All threads in a block wait here until everyone arrives — **required** after shared memory writes before reads. **让同一个 block 内的所有线程在这里“集合”，等大家都到齐了再一起往下执行。**

```cpp
__global__ void sync_demo(float* data) {
    __shared__ float smem[256];
    int tid = threadIdx.x;

    smem[tid] = data[blockIdx.x * 256 + tid];  // Write to SMEM
    __syncthreads();                             // WAIT for all threads

    // Now safe to read any position in smem
    float neighbor = smem[(tid + 1) % 256];
}
```

**Rules:**

-   ALL threads in the block must reach the same `__syncthreads()` — never put it inside divergent `if` branches
-   Must sync between shared memory write and read by other threads

------

## 4. Practice Exercises — Phase 1 (练习)

Write these kernels from scratch to build muscle memory:

| #    | Exercise                                        | Key Concepts                   |
| ---- | ----------------------------------------------- | ------------------------------ |
| 1    | Vector Add (向量加法)                           | 1D indexing, boundary check    |
| 2    | Matrix Transpose (矩阵转置)                     | 2D indexing, row-major layout  |
| 3    | Parallel Reduction Sum (并行归约求和)           | Shared memory, `__syncthreads` |
| 4    | 1D Stencil / Moving Average (一维模板/移动平均) | Shared memory halo cells       |
| 5    | Histogram (直方图)                              | `atomicAdd`                    |
| 6    | SAXPY: `y = a*x + y`                            | Grid-stride loop               |

------

# III. Phase 2: Write Fast Kernels (写出快速的 Kernel) — Week 3-4

Now you focus on **why** kernels are slow and **how** to fix it.

## 1. Memory Coalescing (内存合并访问) — The #1 Performance Rule

Adjacent threads must access adjacent memory addresses — if they don't, the GPU issues multiple memory transactions instead of one.

```cpp
// GOOD: Coalesced (合并) — thread i reads element i
// Threads 0-31 read addresses 0-31 in one 128-byte transaction
__global__ void coalesced(float* data, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = data[idx];  // Adjacent threads, adjacent addresses ✓
}

// BAD: Strided (跨步) — thread i reads element i*stride
// Threads 0-31 read addresses 0, 1024, 2048, ... — 32 separate transactions!
__global__ void strided(float* data, float* out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = data[idx * stride];  // Non-adjacent — 32x slower! ✗
}
```

**AoS vs SoA — the classic coalescing problem:**

```cpp
// BAD: Array of Structures (结构体数组) — non-coalesced
struct Particle { float x, y, z, w; };
// Thread 0 reads particle[0].x at byte 0
// Thread 1 reads particle[1].x at byte 16 — not adjacent!

// GOOD: Structure of Arrays (数组结构体) — coalesced
struct Particles { float* x; float* y; float* z; float* w; };
// Thread 0 reads x[0] at byte 0
// Thread 1 reads x[1] at byte 4 — adjacent! ✓
```

------

## 2. Shared Memory Bank Conflicts (共享内存 Bank 冲突)

Shared memory has 32 Banks (Bank) — if two threads in the same warp access the same bank (different addresses), accesses are serialized.

```cpp
// 32 banks, 4 bytes each, cycling: bank = (address / 4) % 32

// NO conflict: each thread hits different bank
sdata[threadIdx.x]          // Thread 0 -> bank 0, Thread 1 -> bank 1, ...

// 2-way conflict: stride-2 access
sdata[threadIdx.x * 2]      // Thread 0 -> bank 0, Thread 1 -> bank 2, ...
                             // But Thread 0 and Thread 16 both hit bank 0!

// 32-way conflict (worst case): all threads same bank
sdata[threadIdx.x * 32]     // All threads hit bank 0 — fully serialized!
```

**Fix: Padding (填充)**

```cpp
// Transpose with bank conflicts
__shared__ float tile[32][32];   // Column access = 32-way conflict

// Fix: add 1 padding element per row
__shared__ float tile[32][33];   // Column access now conflict-free!
//                          ^^ padding
```

------

## 3. Warp Divergence (Warp 分化)

A Warp (线程束) = 32 threads executing in lockstep. If threads in the same warp take different `if/else` branches, both branches execute sequentially — 2x slowdown.

```cpp
// BAD: Warp divergence — half the warp takes each branch
if (threadIdx.x % 2 == 0) {
    // Even threads do this
} else {
    // Odd threads do this — same warp, both execute serially!
}

// BETTER: Rearrange so entire warps take the same branch
if (threadIdx.x < 16) {
    // First 16 threads (might be in same warp)
} else {
    // Last 16 threads
}

// BEST: Predication — compiler replaces branch with conditional move
float result = (threadIdx.x % 2 == 0) ? a : b;  // No divergence
```

------

## 4. Occupancy (占用率) — Hiding Latency

Occupancy (占用率) = Active Warps / Max Warps per SM. Higher occupancy means more warps available to hide Memory Latency (内存延迟) when one warp is waiting for data.

$$ \text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Max Warps per SM}} $$

**What limits occupancy:**

| Resource                | Limit                       | How to check         |
| ----------------------- | --------------------------- | -------------------- |
| Registers per thread    | Too many → fewer warps fit  | `--ptxas-options=-v` |
| Shared memory per block | Too much → fewer blocks fit | `--ptxas-options=-v` |
| Block size              | Too small → underutilize SM | Use 128-256 threads  |

```bash
# Check register/SMEM usage at compile time
nvcc --ptxas-options=-v my_kernel.cu
# Output: "Used 32 registers, 4096 bytes smem, ..."
```

**Tool:** Use NVIDIA's Occupancy Calculator or `cudaOccupancyMaxActiveBlocksPerMultiprocessor()`.

------

## 5. Atomics (原子操作) — Thread-Safe but Slow

Atomics serialize concurrent writes to the same address — use sparingly and prefer local reduction first.

```cpp
// NAIVE: All threads atomic to one address — extremely slow
__global__ void bad_reduce(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        atomicAdd(result, data[idx]);   // Millions of threads fighting!
}

// BETTER: Block-level reduce first, then one atomic per block
__global__ void good_reduce(float* data, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Only ONE atomic per block (instead of 256)
    if (tid == 0)
        atomicAdd(result, sdata[0]);
}
```

**Available atomics:** `atomicAdd`, `atomicSub`, `atomicMax`, `atomicMin`, `atomicExch`, `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`.

------

## 6. Practice Exercises — Phase 2 (练习)

| #    | Exercise                            | Key Concepts                                       |
| ---- | ----------------------------------- | -------------------------------------------------- |
| 1    | Matrix Transpose (optimized)        | Shared memory + padding for bank conflicts         |
| 2    | Parallel Reduction (warp-optimized) | Warp divergence elimination, sequential addressing |
| 3    | GEMM v1 (naive)                     | 2D indexing, global memory only                    |
| 4    | GEMM v2 (tiled)                     | Shared memory tiling, `__syncthreads`              |
| 5    | Coalescing benchmark                | Measure strided vs coalesced bandwidth             |
| 6    | Image blur / convolution            | Shared memory halo, constant memory for filter     |

------

# IV. Phase 3: Write Expert Kernels (写出专家级 Kernel) — Week 5-8

## 1. Warp-Level Primitives (Warp 级原语)

Shuffle allows data exchange between threads in a warp **without shared memory** — faster and uses fewer resources.

### 1) `__shfl_sync` — Read From Any Lane (从任意 Lane 读取)

```cpp
__global__ void broadcast_demo(float* out) {
    int lane = threadIdx.x % 32;
    float val = (float)lane;
    // All threads in warp read lane 0's value
    float from_lane0 = __shfl_sync(0xffffffff, val, 0);
    out[threadIdx.x] = from_lane0;  // All get 0.0
}
```

### 2) `__shfl_down_sync` — Warp Reduction (Warp 归约)

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Lane 0 has the sum of all 32 lanes
}

__global__ void fast_reduce(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    // Step 1: Warp-level reduce (no shared memory needed!)
    val = warp_reduce_sum(val);

    // Step 2: First lane of each warp writes to shared memory
    __shared__ float warp_sums[8];  // Max 8 warps per block (256 threads)
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // Step 3: First warp reduces the warp sums
    if (warp_id == 0) {
        val = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(result, val);
    }
}
```

### 3) `__shfl_xor_sync` — Butterfly Reduction (蝶形归约)

```cpp
__device__ float warp_reduce_xor(float val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;  // ALL lanes have the sum (not just lane 0)
}
```

### 4) Warp Vote Functions (Warp 投票函数)

```cpp
__global__ void vote_demo(int* data, int* result) {
    int idx = threadIdx.x;
    bool is_positive = data[idx] > 0;

    // Do ALL threads in warp agree?
    int all_pos = __all_sync(0xffffffff, is_positive);

    // Does ANY thread in warp satisfy?
    int any_pos = __any_sync(0xffffffff, is_positive);

    // Get a bitmask of which lanes satisfy
    unsigned mask = __ballot_sync(0xffffffff, is_positive);
    int count = __popc(mask);  // Count set bits = count of positive values
}
```

------

## 2. Async Copy — `cp.async` (异步拷贝, Ampere+)

Hardware-accelerated GMEM→SMEM copy that doesn't occupy compute resources — enables pipelining.

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void async_copy_kernel(float* gmem, float* out) {
    __shared__ float smem[256];
    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline();
    int tid = threadIdx.x;

    // Issue async copy: GMEM -> SMEM (non-blocking)
    pipe.producer_acquire();
    cuda::memcpy_async(&smem[tid], &gmem[tid], sizeof(float), pipe);
    pipe.producer_commit();

    // Wait for copy to complete
    pipe.consumer_wait();
    block.sync();

    // Now SMEM data is ready
    out[tid] = smem[tid] * 2.0f;
    pipe.consumer_release();
}
```

------

## 3. Tensor Core via WMMA (张量核心 via WMMA API)

Direct access to Tensor Cores (张量核心) for matrix multiply — 8-16x faster than CUDA cores for GEMM.

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Each warp computes a 16x16 output tile
__global__ void wmma_gemm(half* A, half* B, float* C, int M, int N, int K) {
    // Declare fragments (寄存器片段)
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float>         c_frag;

    // Initialize accumulator to zero
    fill_fragment(c_frag, 0.0f);

    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y);
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        // Load 16x16 tiles from global memory
        load_matrix_sync(a_frag, A + warp_row * 16 * K + k, K);
        load_matrix_sync(b_frag, B + k * N + warp_col * 16, N);
        // Tensor Core: C += A * B
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, mem_row_major);
}
```

------

## 4. Cooperative Groups (协作组) — Modern Synchronization

Replace raw `__syncthreads()` with typed, composable group objects.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_reduce(float* data, float* result, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = block.group_index().x * block.group_dim().x + block.thread_rank();
    float val = (idx < n) ? data[idx] : 0.0f;

    // Warp-level reduce using cooperative groups
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }

    // Warp leader writes
    if (warp.thread_rank() == 0) {
        atomicAdd(result, val);
    }
}
```

------

## 5. CUDA Graphs (CUDA 图) — Reduce Launch Overhead

Record a sequence of kernel launches as a Graph (图) and replay it — eliminates per-launch CPU overhead.

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<grid, block, 0, stream>>>(args...);
kernel_B<<<grid, block, 0, stream>>>(args...);
kernel_C<<<grid, block, 0, stream>>>(args...);
cudaStreamEndCapture(stream, &graph);

// Instantiate once
cudaGraphInstantiate(&instance, graph, 0);

// Replay many times — near-zero CPU overhead
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}
```

------

## 6. Practice Exercises — Phase 3 (练习)

| #    | Exercise                       | Key Concepts                                     |
| ---- | ------------------------------ | ------------------------------------------------ |
| 1    | Warp Shuffle Reduce            | `__shfl_down_sync`, no shared memory             |
| 2    | Prefix Sum (Scan)              | Warp shuffle + shared memory hybrid              |
| 3    | Tiled GEMM + Tensor Core       | WMMA API, shared memory staging                  |
| 4    | Pipelined Kernel               | `cp.async`, double/triple buffering              |
| 5    | Softmax (fused)                | Online reduction + exp + normalize in one kernel |
| 6    | FlashAttention v1 (simplified) | Tiled attention, online softmax                  |

------

# V. Phase 4: Profiling & Production (性能分析与生产优化) — Ongoing

## 1. Profiling Tools (性能分析工具)

| Tool                     | What It Shows                                        | Command              |
| ------------------------ | ---------------------------------------------------- | -------------------- |
| `nsys` (Nsight Systems)  | Timeline: CPU/GPU overlap, kernel duration           | `nsys profile ./app` |
| `ncu` (Nsight Compute)   | Kernel details: occupancy, memory throughput, stalls | `ncu ./app`          |
| `nvcc -ptxas-options=-v` | Register/SMEM usage at compile time                  | Add to build         |

```bash
# Step 1: Big picture — where is time spent?
nsys profile --stats=true ./my_app

# Step 2: Deep dive into slowest kernel
ncu --set full -k "kernel_name" ./my_app

# Key metrics to look for:
# - Achieved Occupancy (目标: > 50%)
# - Memory Throughput (目标: close to peak bandwidth)
# - Compute Throughput (目标: close to peak FLOPS)
# - Warp Stall Reasons (找到瓶颈原因)
```

## 2. Optimization Decision Tree (优化决策树)

```
Is kernel memory-bound or compute-bound?
│
├── Memory-bound (内存瓶颈):
│   ├── Check coalescing → fix strided access patterns
│   ├── Check L2 hit rate → improve spatial locality
│   ├── Use shared memory → reduce global memory traffic
│   ├── Use `__ldg()` → read-only cache path
│   └── Fuse kernels → eliminate intermediate GMEM reads/writes
│
└── Compute-bound (计算瓶颈):
    ├── Use Tensor Cores → 8-16x for GEMM-like ops
    ├── Use fast math → `__expf`, `__sinf` intrinsics
    ├── Reduce register pressure → increase occupancy
    ├── Use FP16/BF16 → 2x throughput
    └── Vectorize → `float4` loads for 4x fewer instructions
```

------

# VI. Complete API Knowledge Map (完整 API 知识图谱)

## 1. What You Already Know (已掌握)

| Category                                     | Status |
| -------------------------------------------- | ------ |
| Runtime API (内存管理, kernel 启动, 流/事件) | ✅ Done |

## 2. Phase 1 APIs (基础 Kernel 编写)

| API                     | Purpose (用途)                             | Priority |
| ----------------------- | ------------------------------------------ | -------- |
| `threadIdx.x/y/z`       | Thread index within block (块内线程索引)   | ★★★      |
| `blockIdx.x/y/z`        | Block index within grid (网格内块索引)     | ★★★      |
| `blockDim.x/y/z`        | Block dimensions (块维度)                  | ★★★      |
| `gridDim.x/y/z`         | Grid dimensions (网格维度)                 | ★★       |
| `__shared__`            | Shared memory declaration (共享内存声明)   | ★★★      |
| `__constant__`          | Constant memory declaration (常量内存声明) | ★★       |
| `__syncthreads()`       | Block barrier (块同步)                     | ★★★      |
| `atomicAdd/Max/Min/CAS` | Atomic operations (原子操作)               | ★★★      |
| `if (idx < n)`          | Boundary check pattern (边界检查模式)      | ★★★      |

## 3. Phase 2 APIs (性能优化)

| API                                        | Purpose (用途)                                  | Priority |
| ------------------------------------------ | ----------------------------------------------- | -------- |
| Memory coalescing patterns                 | Aligned, contiguous access (合并访问模式)       | ★★★      |
| Shared memory padding                      | Bank conflict avoidance (避免 Bank 冲突)        | ★★★      |
| `__launch_bounds__(maxThreads, minBlocks)` | Hint compiler for register usage                | ★★       |
| `__ldg(ptr)`                               | Read-only cache load (只读缓存加载)             | ★★       |
| `float4` vectorized load/store             | 128-bit memory transactions                     | ★★       |
| Occupancy Calculator API                   | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | ★★       |

## 4. Phase 3 APIs (专家级)

| API                                   | Purpose (用途)                        | Priority |
| ------------------------------------- | ------------------------------------- | -------- |
| `__shfl_sync/down/up/xor`             | Warp shuffle (Warp 洗牌)              | ★★★      |
| `__all_sync/__any_sync/__ballot_sync` | Warp vote (Warp 投票)                 | ★★       |
| `__popc/__clz/__ffs/__brev`           | Bit manipulation intrinsics (位操作)  | ★        |
| `nvcuda::wmma::*`                     | Tensor Core WMMA API (张量核心)       | ★★★      |
| `cooperative_groups::*`               | Modern sync primitives (现代同步)     | ★★       |
| `cuda::memcpy_async`                  | Async GMEM→SMEM (异步拷贝)            | ★★       |
| `cuda::pipeline`                      | Multi-stage pipeline (多级流水线)     | ★★       |
| CUDA Graphs                           | Reduce launch overhead (减少启动开销) | ★★       |
| `asm("ptx...")`                       | Inline PTX assembly (内联 PTX 汇编)   | ★        |

## 5. Timeline Summary (时间线总结)

```
Week 1-2:  Phase 1 — Correct kernels
           Thread indexing, memory spaces, __syncthreads, atomics
           "My kernel gives the right answer"

Week 3-4:  Phase 2 — Fast kernels
           Coalescing, bank conflicts, warp divergence, occupancy
           "My kernel runs at >50% of peak bandwidth"

Week 5-8:  Phase 3 — Expert kernels
           Warp shuffle, Tensor Cores, async copy, cooperative groups
           "My kernel matches or beats cuBLAS/cuDNN"

Ongoing:   Phase 4 — Profile-driven optimization
           nsys/ncu, identify bottlenecks, iterate
           "I know exactly why my kernel is slow and how to fix it"
```

------

# VII. Recommended Resources (推荐资源)

| Resource                    | What It Covers                        | URL/Command                      |
| --------------------------- | ------------------------------------- | -------------------------------- |
| CUDA C++ Programming Guide  | Official reference for everything     | `docs.nvidia.com/cuda`           |
| CUDA Samples                | Working examples for every feature    | `github.com/NVIDIA/cuda-samples` |
| Nsight Compute              | Interactive kernel profiler           | `ncu --set full ./app`           |
| Mark Harris blog posts      | Classic optimization tutorials        | NVIDIA Developer Blog            |
| `cuda-mode` Discord/YouTube | Community lectures on GPU programming | Search "cuda-mode"               |
