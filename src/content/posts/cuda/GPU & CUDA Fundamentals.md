---
title: "GPU & CUDA Fundamentals"
published: 2026-04-24
description: "GPU & CUDA Fundamentals"
image: ""
tags: ["cuda","GPU & CUDA Fundamentals"]
category: cuda
draft: false
lang: ""
---

# I. GPU & CUDA Fundamentals (GPU与CUDA基础模型)

## 1. CPU vs GPU Design Philosophy

### 1) Core Architecture Difference

CPUs (中央处理器) optimize for **low-latency serial execution** with a few powerful cores, while GPUs (图形处理器) optimize for **high-throughput parallel execution** with thousands of simpler cores.

| Feature                   | CPU                              | GPU                      |
| ------------------------- | -------------------------------- | ------------------------ |
| Core count (核心数)       | 4 – 128                          | 1,000 – 16,000+          |
| Clock speed (时钟频率)    | 3 – 5 GHz                        | 1 – 2.5 GHz              |
| Cache per core (每核缓存) | Large (MB)                       | Small (KB)               |
| Control logic (控制逻辑)  | Complex (branch prediction, OOO) | Simple                   |
| Best for (适用场景)       | Serial, branchy logic            | Data-parallel arithmetic |

### 2) When to Use a GPU

Use a GPU when the **same operation** must be applied to **millions of independent data elements** simultaneously — e.g., matrix multiply (矩阵乘法), image processing (图像处理), deep learning (深度学习).

------

## 2. SIMT Execution Model

### 1) Definition

**SIMT** (Single Instruction, Multiple Threads — 单指令多线程) means all threads (线程) in a Warp execute the **same instruction** each cycle, each on its own data — like SIMD (单指令多数据) but with per-thread program counters.

### 2) Warp Divergence

**Warp divergence (Warp分歧)** occurs when threads in a warp take different `if/else` branches — the GPU serializes both paths and **masks** inactive threads, halving throughput.

```cuda
// BAD: divergence — odd/even threads take different paths
__global__ void divergent(int *a) {
    int tid = threadIdx.x;
    if (tid % 2 == 0)   // half the warp goes here
        a[tid] = tid * 2;
    else                // other half goes here → serialized!
        a[tid] = tid + 1;
}
```

------

## 3. Thread Hierarchy (线程层次结构)

### 1) Four Levels

CUDA organizes threads into a **three-level software hierarchy** that maps onto GPU hardware:

```
Grid (网格)
 └── Block (线程块)  ×  many
      └── Warp (线程束)  ×  (blockDim / 32)   ← hardware unit
           └── Thread (线程)  ×  32
```

### 2) Built-in Index Variables

Every thread knows its position via read-only built-in variables:

| Variable               | Type   | Meaning                              |
| ---------------------- | ------ | ------------------------------------ |
| `threadIdx` (线程索引) | `dim3` | Thread position **within** its block |
| `blockIdx` (块索引)    | `dim3` | Block position **within** the grid   |
| `blockDim` (块维度)    | `dim3` | Number of threads per block          |
| `gridDim` (格维度)     | `dim3` | Number of blocks in the grid         |

### 3) Computing a Global Thread ID

The most common pattern — map each thread to one array element:

$$ \text{globalId} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x} $$

```cuda
// Each thread processes one element of array a[]
__global__ void addOne(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread ID
    if (idx < n)          // boundary guard (越界保护)
        a[idx] += 1;
}
```

------

## 4. Warp (线程束)

### 1) Definition

A **warp** (线程束) is the hardware scheduling unit — always **32 consecutive threads** from the same block executing in lockstep (锁步) on one SM.

### 2) Why 32?

32 balances two constraints: wide enough SIMD datapaths to hide memory latency (内存延迟), but small enough that the warp scheduler (Warp调度器) can context-switch between hundreds of warps with **zero overhead** (零开销).

### 3) Key Numbers

| Quantity                                       | Typical Value  |
| ---------------------------------------------- | -------------- |
| Threads per warp (每个Warp的线程数)            | **32** (fixed) |
| Max warps resident per SM (每SM最大驻留Warp数) | 32 – 64        |
| Max threads per SM (每SM最大线程数)            | 1024 – 2048    |

------

## 5. Block (线程块)

### 1) Definition

A **block** (线程块) is a group of up to **1024 threads** that run on a **single SM** (流式多处理器), share **shared memory** (共享内存), and can synchronize with `__syncthreads()`.

### 2) Block ↔ SM Mapping Rule

One block is **always assigned to exactly one SM** — but one SM can hold **multiple blocks** concurrently (subject to resource limits like shared memory and registers).

```cuda
// Threads within a block cooperate via shared memory + barrier
__global__ void blockCooperate(int *out, int *in, int n) {
    __shared__ int tile[256];          // shared memory (共享内存) — visible to all threads in block
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;

    tile[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();                   // barrier (屏障同步) — all threads reach here before continuing

    // Example: each thread reads its neighbour's value
    if (tid > 0 && gid < n)
        out[gid] = tile[tid] + tile[tid - 1];
}
```

### 3) Choosing Block Size

Always choose `blockDim` as a **multiple of 32** to avoid partial warps; **128 or 256** is the common starting point.

------

## 6. Grid (网格)

### 1) Definition

A **grid** (网格) is the full collection of blocks launched by a single kernel call — it covers the entire problem domain and maps across **all SMs** on the device.

### 2) 1-D Grid Launch Pattern

```cuda
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *c, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;            // 1M elements
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;   // ceiling division (向上取整)

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Launch: GRID blocks, each with BLOCK threads
    vectorAdd<<<GRID, BLOCK>>>(d_c, d_a, d_b, N);
    cudaDeviceSynchronize();          // wait for GPU (等待GPU完成)

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    printf("Done. Grid=%d  Block=%d  Total threads=%d\n", GRID, BLOCK, GRID*BLOCK);
    return 0;
}
```

### 3) 2-D Grid for Images

Use `dim3` when the data is naturally 2-D (e.g., image pixels):

```cuda
dim3 block(16, 16);                            // 16×16 = 256 threads per block
dim3 grid((W + 15) / 16, (H + 15) / 16);      // enough blocks to cover W×H image
kernel<<<grid, block>>>(...);
```

------

## 7. SM — Streaming Multiprocessor (流式多处理器)

### 1) Definition

An **SM** (流式多处理器) is the fundamental compute unit of the GPU — each SM contains CUDA cores (CUDA核心), a warp scheduler (Warp调度器), register file (寄存器文件), shared memory (共享内存), and L1 cache (L1缓存).

### 2) SM Internal Components

| Component                    | Role                                                   |
| ---------------------------- | ------------------------------------------------------ |
| CUDA Cores (CUDA核心)        | Execute integer / FP32 arithmetic                      |
| Tensor Cores (张量核心)      | Accelerate matrix ops (Ampere+)                        |
| Warp Schedulers (Warp调度器) | Issue instructions to ready warps every cycle          |
| Register File (寄存器文件)   | 65536 × 32-bit registers shared among resident threads |
| Shared Memory (共享内存)     | Fast on-chip SRAM, 16–100 KB, programmer-managed       |
| L1 / Texture Cache (L1缓存)  | Physically unified with shared memory                  |

### 3) Latency Hiding via Warp Switching

When a warp stalls on a memory load (内存加载), the warp scheduler **instantly switches** to another ready warp — zero switching overhead — this is how GPUs tolerate high memory latency (高内存延迟).

```
Cycle:  1   2   3   4   5   6   7   8
Warp A: LOAD ···stall···       COMPUTE
Warp B:           LOAD ···stall···    COMPUTE
Warp C:                    COMPUTE
          ↑ scheduler fills stall slots with other warps
```

------

## 8. Kernel Launch Model (Kernel启动模型)

### 1) Syntax

```cuda
kernelName<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| Parameter        | Default | Meaning                                       |
| ---------------- | ------- | --------------------------------------------- |
| `gridDim`        | —       | Number of blocks (可以是`int`或`dim3`)        |
| `blockDim`       | —       | Threads per block (可以是`int`或`dim3`)       |
| `sharedMemBytes` | `0`     | Dynamic shared memory (动态共享内存) in bytes |
| `stream`         | `0`     | CUDA stream (CUDA流); `0` = default stream    |

### 2) Execution Flow

```
Host (CPU主机端)                    Device (GPU设备端)
─────────────────                   ──────────────────
cudaMalloc()          →  allocate GPU memory (分配显存)
cudaMemcpy(H→D)       →  copy input data
kernel<<<G,B>>>()     →  launch (异步返回, GPU开始执行)
cudaDeviceSynchronize()→  block CPU until GPU finishes
cudaMemcpy(D→H)       →  copy results back
cudaFree()            →  release GPU memory
```

### 3) Complete Minimal Example

```cuda
// Compile: nvcc -o hello hello.cu && ./hello
#include <stdio.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();           // 2 blocks × 4 threads = 8 threads total
    cudaDeviceSynchronize();    // flush GPU printf buffer (刷新GPU打印缓冲)
    return 0;
}
/* Expected output (order may vary — parallel!):
   Hello from block 0, thread 0
   Hello from block 0, thread 1
   Hello from block 0, thread 2
   Hello from block 0, thread 3
   Hello from block 1, thread 0
   ...
*/
```

------

## 9. Mental Model Summary (心智模型总结)

### 1) Hardware ↔ Software Mapping

```
Software (CUDA编程模型)        Hardware (GPU硬件)
──────────────────────        ──────────────────
Grid    (网格)           →    Entire GPU device
Block   (线程块)         →    One SM  (流式多处理器)
Warp    (线程束, 32)     →    One set of SIMD lanes in SM
Thread  (线程)           →    One CUDA core execution lane
```

### 2) One-Sentence Rules to Remember

-   **Warp** = 32 threads, always runs together — design to avoid divergence (分歧).
-   **Block** = fits one SM — use shared memory to communicate within a block.
-   **Grid** = one kernel launch — scale to millions of threads across all SMs.
-   **SM** = hides latency by switching warps — keep it busy with enough resident warps (驻留Warp).
