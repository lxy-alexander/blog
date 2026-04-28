---
title: "CUDA Memory Hierarchy"
published: 2026-04-27
description: "CUDA Memory Hierarchy"
image: ""
tags: ["cuda","CUDA Memory Hierarchy"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-28T01:59:36.550.773177004Z"
---

# CUDA Memory Hierarchy (CUDA 内存体系)

Goal: understand why CUDA performance is often bottlenecked by memory access (内存访问) rather than computation.

## 1. Memory Hierarchy Architecture (内存层级架构)

The GPU memory system is organized as a pyramid: the closer to the compute units (计算单元), the faster but smaller the memory.Now let me build the diagram:<br>

![image-20260427220011100](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260427220011100)

## 2. Concepts to Learn (with Use Cases)

### 1) Global Memory (全局内存)

Global Memory (全局内存) is the GPU's main DRAM (显存), large (GBs) but slow (~400-800 cycles), used for **input/output buffers** (输入输出缓冲区) of every kernel.

**Use case**: storing the full input vectors/matrices that the host (主机) copies into the device (设备).

### 2) Shared Memory (共享内存)

Shared Memory (共享内存) is fast on-chip (片上) memory shared by all threads (线程) in a block (线程块), used for **data reuse** (数据复用) across threads.

**Use case**: tiling (分块) in matrix multiplication or transpose — load a tile once from global memory, reuse it many times.

### 3) Register (寄存器)

Registers (寄存器) are the fastest memory, private to each thread (线程), used for **local scalar variables** (局部标量变量) inside a kernel.

**Use case**: loop counters, intermediate `float` values like `sum`, `idx`.

### 4) Local Memory (本地内存)

Local Memory (本地内存) is per-thread (每线程) but actually lives in DRAM (显存), used only when **registers spill** (寄存器溢出) — e.g., large local arrays.

**Use case**: usually unintentional — when a kernel uses too many registers, the compiler spills to local memory.

### 5) Constant Memory (常量内存)

Constant Memory (常量内存) is a 64KB read-only (只读) region with a dedicated cache (专用缓存), used when **all threads read the same value** (所有线程读相同值).

**Use case**: convolution kernel weights, lookup tables (查找表), broadcast constants.

### 6) L1 Cache (一级缓存)

L1 Cache (一级缓存) is per-SM (流式多处理器) on-chip cache that **physically shares storage with shared memory** (与共享内存共用片上存储) on most GPUs.

**Use case**: automatic — caches recent global memory loads to hide latency (隐藏延迟).

### 7) L2 Cache (二级缓存)

L2 Cache (二级缓存) is a unified cache (统一缓存) shared by **all SMs**, sitting between L1 and global memory.

**Use case**: automatic — lets multiple blocks (线程块) reuse data without going back to DRAM.

### 8) Memory Coalescing (内存合并访问)

Memory Coalescing (内存合并访问) means a warp (线程束) of 32 threads accesses **32 contiguous addresses** (连续地址), merged by hardware into one transaction.

**Use case**: `a[idx]` where `idx = blockIdx.x * blockDim.x + threadIdx.x` — the gold standard access pattern.

### 9) Bank Conflict (Bank 冲突)

Bank Conflict (Bank 冲突) happens when multiple threads in a warp (线程束) access **different addresses in the same shared-memory bank** (共享内存的同一 bank), forcing serialization (串行化).

**Use case**: matrix transpose using `tile[ty][tx]` — column access hits the same bank, fixed by **padding** (填充) `tile[TILE][TILE+1]`.

### 10) Memory Bandwidth (内存带宽)

Memory Bandwidth (内存带宽) is the maximum bytes/second the GPU can move from DRAM (显存), and most kernels are **bandwidth-bound** (带宽受限) rather than compute-bound (计算受限).

**Use case**: profiling — if a kernel's achieved bandwidth (实际带宽) is near peak (峰值), it's already optimal.

<br>

## 3. Code to Write

### 1) copy_kernel.cu (内存拷贝基准)

A pure copy kernel measures **peak bandwidth** (峰值带宽) — every other kernel is compared against it.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];  // Coalesced (合并访问)
}

int main() {
    const int n = 8;
    size_t bytes = n * sizeof(float);
    float h_in[n] = {1,2,3,4,5,6,7,8}, h_out[n];

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    copy_kernel<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_out[i]);
    // Output: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 2) strided_access.cu (步长访问)

Strided access (步长访问) breaks coalescing (合并访问), so each warp (线程束) issues many transactions instead of one — much slower.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void strided_copy(const float* in, float* out, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) out[idx] = in[idx];  // Stride breaks coalescing (破坏合并访问)
}

int main() {
    const int n = 16;
    size_t bytes = n * sizeof(float);
    float h_in[n], h_out[n] = {0};
    for (int i = 0; i < n; i++) h_in[i] = i + 1;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);

    strided_copy<<<1, 8>>>(d_in, d_out, n, 2);  // stride=2
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.0f ", h_out[i]);
    // Output: 1 0 3 0 5 0 7 0 9 0 11 0 13 0 15 0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 3) matrix_transpose_naive.cu (朴素矩阵转置)

Naive transpose (朴素转置) writes to `out[x * N + y]` — the **write is strided** (写入步长大), causing uncoalesced (非合并) global memory access.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void transpose_naive(const float* in, float* out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) out[x * N + y] = in[y * N + x];  // Write is strided (写入非合并)
}

int main() {
    const int N = 4;
    size_t bytes = N * N * sizeof(float);
    float h_in[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}, h_out[16];

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(2, 2), grid(2, 2);
    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) { printf("%2.0f ", h_out[i]); if (i%4==3) printf("\n"); }
    // Output: 1  5  9 13
    //         2  6 10 14
    //         3  7 11 15
    //         4  8 12 16
    cudaFree(d_in); cudaFree(d_out);
}
```

### 4) matrix_transpose_shared.cu (共享内存矩阵转置)

Using shared memory (共享内存) as a staging tile (暂存块) makes **both global reads and writes coalesced** (读写都合并); padding (填充) avoids bank conflicts (Bank 冲突).

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#define TILE 16

__global__ void transpose_shared(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding avoids bank conflict (避免 bank 冲突)
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();  // Sync block (同步线程块)
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int N = 16;
    size_t bytes = N * N * sizeof(float);
    float *h_in = new float[N * N], *h_out = new float[N * N];
    for (int i = 0; i < N * N; i++) h_in[i] = i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE), grid(1, 1);
    transpose_shared<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("h_out[0..3]=%g %g %g %g\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    // Output: h_out[0..3]=0 16 32 48
    delete[] h_in; delete[] h_out;
    cudaFree(d_in); cudaFree(d_out);
}
```

### 5) shared_memory_demo.cu (共享内存演示)

This kernel computes a per-block sum (块内求和) using shared memory (共享内存) as scratch space (暂存区) — classic data reuse (数据复用) pattern.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    // Tree reduction (树形规约) within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 256;
    size_t bytes = n * sizeof(float);
    float *h_in = new float[n], h_out = 0;
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;  // Sum = 256

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    block_sum<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %.1f\n", h_out);
    // Output: sum = 256.0
    delete[] h_in; cudaFree(d_in); cudaFree(d_out);
}
```

### 6) bank_conflict_demo.cu (Bank 冲突演示)

Shared memory (共享内存) has 32 banks (32 个 bank); column-major access (列访问) on `tile[TILE][TILE]` causes a **32-way bank conflict** (32 路冲突), fixed by padding `[TILE+1]`.

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#define TILE 32

// Bad: 32-way bank conflict on column access
__global__ void bank_conflict_bad(float* out) {
    __shared__ float tile[TILE][TILE];  // No padding (无填充)
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = ty * TILE + tx;
    __syncthreads();
    out[ty * TILE + tx] = tile[tx][ty];  // Column access -> conflict (列访问 -> 冲突)
}

// Good: padding eliminates conflict
__global__ void bank_conflict_good(float* out) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding (填充)
    int tx = threadIdx.x, ty = threadIdx.y;
    tile[ty][tx] = ty * TILE + tx;
    __syncthreads();
    out[ty * TILE + tx] = tile[tx][ty];  // No conflict (无冲突)
}

int main() {
    size_t bytes = TILE * TILE * sizeof(float);
    float *d_out, h_out[TILE * TILE];
    cudaMalloc(&d_out, bytes);

    dim3 block(TILE, TILE);
    bank_conflict_good<<<1, block>>>(d_out);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("h_out[0..3] = %.0f %.0f %.0f %.0f\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    // Output: h_out[0..3] = 0 32 64 96
    cudaFree(d_out);
}
```

<br>

## 4. Must-Know Questions

### 1) What is coalesced memory access (合并内存访问)?

Coalesced access (合并访问) means a warp's 32 threads (线程) access 32 contiguous, aligned addresses (连续对齐地址), letting the hardware service them in **a single 128-byte transaction** (单次 128 字节事务) instead of 32 separate ones.

### 2) Why is strided access (步长访问) slow?

Because each thread (线程) hits a different cache line (缓存行), the hardware must issue **multiple transactions per warp** (每束多次事务), wasting bandwidth (带宽) — most fetched bytes are unused.

### 3) Why is Shared Memory (共享内存) fast?

Because it sits **on-chip** (片上) next to the SM (流式多处理器), has ~100x lower latency (延迟) than DRAM (显存), and is split into 32 banks for parallel access (并行访问).

### 4) What is a Bank Conflict (Bank 冲突)?

A bank conflict (Bank 冲突) occurs when multiple threads in a warp (线程束) hit **different addresses in the same bank**, forcing the hardware to serialize accesses (串行化访问), turning one cycle into N cycles.

### 5) Why does register spilling (寄存器溢出) hurt performance?

Because spilled variables move to local memory (本地内存), which physically lives in **DRAM** (显存) — what looks like a register access becomes a slow global memory access (全局内存访问).

<br>

## 5. Stage Outcome (阶段成果)

You should now be able to explain this sentence:

>   A CUDA kernel is often slow not because computation (计算) is slow, but because data movement (数据搬运) is slow.

The key insight (核心洞察) is that modern GPUs are **memory-bound** (内存受限), not compute-bound (计算受限) — optimizing memory access patterns (内存访问模式) usually yields more speedup than optimizing arithmetic (算术运算).

<br>
