---
title: "Classic Parallel Patterns"
published: 2026-04-27
description: "Classic Parallel Patterns"
image: ""
tags: ["cuda","Classic Parallel Patterns"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-28T03:22:18.713.491513479Z"
---

# Classic Parallel Patterns (经典并行模式)

Goal: master the common algorithm templates (算法模板) for CUDA kernels (内核函数).

## 1. Patterns to Learn

### 1) Elementwise (逐元素计算)

Elementwise (逐元素) operations apply the same function to each element independently — perfectly parallel (完全并行) and bandwidth-bound (带宽受限).

### 2) Reduction (归约)

Reduction (归约) collapses an array into a single value (e.g., sum, max) using a tree-shaped (树形) parallel pattern with `O(log n)` steps.

### 3) Scan / Prefix Sum (前缀和)

Scan (前缀和) computes the running cumulative result (累积结果) where `out[i] = in[0] + in[1] + ... + in[i]`, using a two-phase up-sweep/down-sweep (向上扫描/向下扫描) algorithm.

### 4) Transpose (矩阵转置)

Transpose (矩阵转置) swaps rows and columns; the naive version (朴素版本) suffers uncoalesced writes (非合并写入), so shared memory tiling (共享内存分块) is required.

### 5) Histogram (直方图)

Histogram (直方图) counts how many input elements fall into each bin (区间), requiring atomic operations (原子操作) to handle write conflicts (写冲突).

### 6) Gather / Scatter (聚集 / 分散)

Gather (聚集) is `out[i] = in[idx[i]]` (random reads, 随机读), and Scatter (分散) is `out[idx[i]] = in[i]` (random writes, 随机写) — both have unpredictable memory patterns (不规则内存模式).

### 7) Stencil (模板计算)

Stencil (模板计算) computes each output from a fixed-shape (固定形状) neighborhood of inputs (e.g., 3x3 convolution, 3x3 卷积), benefiting from shared-memory reuse (共享内存复用).

### 8) Matrix Multiplication (矩阵乘法)

Matrix Multiplication (矩阵乘法) is the canonical compute-bound (计算受限) workload, made fast by tiling (分块) into shared memory (共享内存) for data reuse (数据复用).

<br>

## 2. Code to Write

### 1) reduce_naive.cu (朴素归约)

The naive reduction (朴素归约) does a tree reduction (树形归约) in shared memory (共享内存) but uses divergent branches (分支发散) inside warps (线程束).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void reduce_naive(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    // Divergent loop (分支发散): warp 内一半线程空转
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 256;
    float h_in[256], h_out;
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;  // Sum = 256
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_naive<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("naive sum = %.1f\n", h_out);
    // Output: naive sum = 256.0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 2) reduce_shared.cu (共享内存归约)

The improved version uses **sequential addressing** (顺序寻址), keeping warps (线程束) coherent and avoiding bank conflicts (Bank 冲突).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void reduce_shared(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();
    // Sequential addressing (顺序寻址): 上半 warp 整体退出,无发散
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 256;
    float h_in[256], h_out;
    for (int i = 0; i < n; i++) h_in[i] = (float)(i + 1);  // Sum = 32896
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_shared<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("shared sum = %.1f\n", h_out);
    // Output: shared sum = 32896.0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 3) reduce_warp_shuffle.cu (Warp Shuffle 归约)

The warp-shuffle reduction (Warp Shuffle 归约) replaces shared memory (共享内存) with register-to-register (寄存器到寄存器) shuffles, removing one `__syncthreads()` and saving shared memory.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__device__ float warp_reduce(float val) {
    for (int off = 16; off > 0; off /= 2)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;  // Lane 0 holds the warp sum (lane 0 持有 warp 和)
}

__global__ void reduce_shuffle(const float* in, float* out, int n) {
    __shared__ float sdata[32];                  // One slot per warp (每 warp 一格)
    int tid = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    float v = (tid < n) ? in[tid] : 0.0f;
    v = warp_reduce(v);                           // Step 1: warp-level
    if (lane == 0) sdata[wid] = v;
    __syncthreads();
    v = (tid < blockDim.x / 32) ? sdata[lane] : 0.0f;
    if (wid == 0) v = warp_reduce(v);             // Step 2: reduce warp results
    if (tid == 0) *out = v;
}

int main() {
    const int n = 256;
    float h_in[256], h_out;
    for (int i = 0; i < n; i++) h_in[i] = 2.0f;  // Sum = 512
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_shuffle<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("shuffle sum = %.1f\n", h_out);
    // Output: shuffle sum = 512.0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 4) prefix_sum.cu (前缀和)

This is an inclusive scan (包含式扫描) using the Hillis-Steele algorithm (Hillis-Steele 算法) with `O(n log n)` work — simple but correct for one block (单块).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void prefix_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    // Hillis-Steele scan (Hillis-Steele 扫描)
    for (int offset = 1; offset < n; offset *= 2) {
        float v = (tid >= offset) ? sdata[tid - offset] : 0.0f;
        __syncthreads();
        sdata[tid] += v;
        __syncthreads();
    }
    if (tid < n) out[tid] = sdata[tid];
}

int main() {
    const int n = 8;
    float h_in[8] = {1,2,3,4,5,6,7,8}, h_out[8];
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    prefix_sum<<<1, n>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%.0f ", h_out[i]);
    // Output: 1 3 6 10 15 21 28 36
    cudaFree(d_in); cudaFree(d_out);
}
```

### 5) histogram_atomic.cu (原子直方图)

Histogram (直方图) needs `atomicAdd` (原子加) because many threads (线程) can hit the same bin (区间) simultaneously, causing write conflicts (写冲突).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void histogram(const int* in, int* hist, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = in[idx] % bins;
        atomicAdd(&hist[b], 1);  // Atomic add (原子加),防止写冲突
    }
}

int main() {
    const int n = 16, bins = 4;
    int h_in[16] = {0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3}, h_hist[4] = {0};
    int *d_in, *d_hist;
    cudaMalloc(&d_in, n * sizeof(int)); cudaMalloc(&d_hist, bins * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bins * sizeof(int));
    histogram<<<1, n>>>(d_in, d_hist, n, bins);
    cudaMemcpy(h_hist, d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < bins; i++) printf("bin[%d]=%d ", i, h_hist[i]);
    // Output: bin[0]=4 bin[1]=4 bin[2]=4 bin[3]=4
    cudaFree(d_in); cudaFree(d_hist);
}
```

### 6) transpose_optimized.cu (优化矩阵转置)

The optimized transpose (优化转置) uses a shared-memory tile (共享内存分块) with `+1` padding (填充) to make both reads and writes coalesced (合并访问) and avoid bank conflicts (Bank 冲突).

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#define TILE 32

__global__ void transpose_optimized(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];   // +1 padding (填充避免 bank 冲突)
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // Coalesced read (合并读)
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // Coalesced write (合并写)
}

int main() {
    const int N = 32;
    size_t bytes = N * N * sizeof(float);
    float *h_in = new float[N * N], *h_out = new float[N * N];
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE), grid(1, 1);
    transpose_optimized<<<grid, block>>>(d_in, d_out, N);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("h_out[0..3]=%.0f %.0f %.0f %.0f\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    // Output: h_out[0..3]=0 32 64 96
    delete[] h_in; delete[] h_out;
    cudaFree(d_in); cudaFree(d_out);
}
```

### 7) matmul_naive.cu (朴素矩阵乘法)

Each thread (线程) computes one output element by reading a full row of `A` and a full column of `B` from global memory (全局内存) — high arithmetic but no data reuse (无数据复用).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];   // Re-read global mem (重复读全局内存)
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 4;
    size_t bytes = N * N * sizeof(float);
    float h_A[16], h_B[16], h_C[16];
    for (int i = 0; i < 16; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }  // C = N (all 4)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 block(N, N), grid(1, 1);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) { printf("%.0f ", h_C[i]); if (i % 4 == 3) printf("\n"); }
    // Output: 4 4 4 4
    //         4 4 4 4
    //         4 4 4 4
    //         4 4 4 4
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

### 8) matmul_tiled.cu (分块矩阵乘法)

Tiled GEMM (分块矩阵乘法) loads `TILE x TILE` blocks (块) of `A` and `B` into shared memory (共享内存) once and reuses them `TILE` times — the canonical compute-bound (计算受限) optimization.

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Load tile (加载分块) into shared memory
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        // Compute on shared memory (在共享内存上计算,数据复用 TILE 次)
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

int main() {
    const int N = 16;
    size_t bytes = N * N * sizeof(float);
    float *h_A = new float[N * N], *h_B = new float[N * N], *h_C = new float[N * N];
    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }  // C = N (all 16)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("C[0]=%.0f  C[N*N-1]=%.0f\n", h_C[0], h_C[N * N - 1]);
    // Output: C[0]=16  C[N*N-1]=16
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

<br>

## 3. Priority Order (重点排序)

The top three patterns (Reduction, Transpose, GEMM) cover every core CUDA optimization technique: Shared Memory (共享内存), Warp-level Primitives (Warp 级原语), Memory Coalescing (内存合并访问), Bank Conflict (Bank 冲突), Tiling (分块), and Synchronization (同步).

<br>

## 4. Stage Outcome (阶段成果)

You should now be able to answer:

### 1) Why is naive transpose (朴素转置) slow?

Because writes go to `out[x * N + y]` where consecutive threads (相邻线程) hit addresses `N` apart, breaking memory coalescing (内存合并访问) and turning one transaction (事务) into 32.

### 2) Why is tiled matmul (分块矩阵乘法) faster than naive matmul (朴素矩阵乘法)?

Because each tile (分块) of `A` and `B` is loaded **once** from global memory (全局内存) into shared memory (共享内存) and reused `TILE` times, raising the compute-to-memory ratio (计算访存比) from `O(1)` to `O(TILE)`.

### 3) Why is reduction (归约) split into block-level (块级) and warp-level (Warp 级)?

Because `__syncthreads()` only synchronizes within a block (线程块) — block-level reduction (块级归约) handles cross-warp (跨 warp) accumulation via shared memory (共享内存), and warp-level reduction (Warp 级归约) handles intra-warp (warp 内) accumulation via `__shfl_down_sync` for maximum speed.

<br>
