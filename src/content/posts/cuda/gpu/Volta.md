---
title: "Volta"
published: 2026-04-27
description: "Volta"
image: ""
tags: ["cuda","gpu","Volta"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:54:45.626.738074508Z"
---
# Volta Architecture (伏特架构)

The Volta Architecture (伏特架构, 2017) introduced first-generation Tensor Cores (张量核心), Independent Thread Scheduling (独立线程调度), and Cooperative Groups (协作组) — turning the GPU from a parallel SIMT (单指令多线程) processor into a true deep-learning accelerator (深度学习加速器).

**Representative GPU Models**:

- Tesla V100 / V100S (GV100) — 80 SM, 5120 CUDA Cores + 640 Tensor Cores, HBM2 (data center)
- Quadro GV100
- TITAN V (GV100, 80 SM)
- Jetson Xavier (embedded, 嵌入式)

**Architecture Diagram** — GV100 Example (80 SM):

```
┌────────────────────────────────────────────────────────────────────────┐
│                          GV100 GPU (Volta)                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │     Host Interface (PCIe 3.0) + 6× NVLink 2.0 (300 GB/s)         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       GigaThread Engine                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│   GPC 0          GPC 1         ...                       GPC 5         │
│  ┌─────────┐    ┌─────────┐                            ┌─────────┐     │
│  │ ~14 SM  │    │ ~14 SM  │                            │ ~14 SM  │     │
│  │         │    │         │                            │         │     │
│  │ Each SM:│    │ Each SM:│           ...              │ Each SM:│     │
│  │ 64 SP   │    │ 64 SP   │                            │ 64 SP   │     │
│  │ 32 DP   │    │ 32 DP   │                            │ 32 DP   │     │
│  │ 8 Tensor│    │ 8 Tensor│                            │ 8 Tensor│     │
│  │ 128 KB  │    │ 128 KB  │                            │ 128 KB  │     │
│  │ L1+Shmem│    │ L1+Shmem│                            │ L1+Shmem│     │
│  └─────────┘    └─────────┘                            └─────────┘     │
│       │              │                                       │         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                  Unified L2 Cache (6 MB)                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │       HBM2 Memory (16/32 GB, 4096-bit, 900 GB/s, ECC)            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘

Total: 80 SM × 64 SP = 5120 CUDA Cores + 80 SM × 8 = 640 Tensor Cores
Each SM: unified 128 KB L1 + Shared Memory (configurable up to 96 KB shared)
```

**Solved**: Pascal lacked dedicated matrix-multiply hardware and had lockstep warp execution (锁步执行) that deadlocked on fine-grained synchronization (细粒度同步) — Volta added Tensor Cores for 4×4 matrix FMA, gave each thread its own program counter, and introduced explicit `_sync` warp primitives plus Cooperative Groups.

**Best Suited For**: Deep learning training (深度学习训练) with mixed precision (混合精度), large language model (大语言模型) workloads, and producer-consumer kernels (生产者-消费者核函数) needing fine-grained warp synchronization.

<br>

## 1. Tensor Cores (张量核心)

Tensor Cores (张量核心) are dedicated hardware units that execute a 4×4 matrix-multiply-accumulate (矩阵乘累加) `D = A * B + C` in a single instruction — orders of magnitude faster than CUDA Cores for deep-learning matrix math.

**Solved**: CUDA Cores compute one FMA (融合乘加) per thread per cycle — far too slow for the matrix multiplications dominating neural-network training.

### 1) Mixed Precision (混合精度)

Mixed Precision (混合精度) on Volta uses FP16 (半精度) for matrix inputs A and B, accumulating into FP32 (单精度) — preserving accuracy while doubling throughput vs. pure FP32.

**Old**: Pure FP32 matrix multiply using CUDA Cores — slow.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void matmul16x16_fp32(const float* A, const float* B, float* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    float sum = 0.f;
    for (int k = 0; k < 16; ++k) sum += A[row*16 + k] * B[k*16 + col];
    C[row*16 + col] = sum;
}

int main() {
    const int M = 16;
    float hA[M*M], hB[M*M], hC[M*M];
    for (int i = 0; i < M*M; ++i) { hA[i] = 1.0f; hB[i] = 1.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*M*sizeof(float));
    cudaMalloc(&dB, M*M*sizeof(float));
    cudaMalloc(&dC, M*M*sizeof(float));
    cudaMemcpy(dA, hA, M*M*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, M*M*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    matmul16x16_fp32<<<1, block>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*M*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0][0]=%g, C[15][15]=%g (expect 16)\n", hC[0], hC[M*M-1]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
C[0][0]=16, C[15][15]=16 (expect 16)
*/
```

**New**: Use the WMMA (Warp Matrix Multiply-Accumulate) API to invoke Tensor Cores with FP16 inputs and FP32 accumulator.

```cpp
// Compile: nvcc -arch=sm_70 file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void matmul16x16_tc(const half* A, const half* B, float* C) {
    // 16x16x16 fragment shape — one Tensor Core MMA per warp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // Tensor Core MMA
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main() {
    const int M = 16;
    half hA[M*M], hB[M*M];
    float hC[M*M];
    for (int i = 0; i < M*M; ++i) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, M*M*sizeof(half));
    cudaMalloc(&dB, M*M*sizeof(half));
    cudaMalloc(&dC, M*M*sizeof(float));
    cudaMemcpy(dA, hA, M*M*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, M*M*sizeof(half), cudaMemcpyHostToDevice);

    matmul16x16_tc<<<1, 32>>>(dA, dB, dC);   // one warp = 32 threads
    cudaMemcpy(hC, dC, M*M*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0][0]=%g, C[15][15]=%g (expect 16)\n", hC[0], hC[M*M-1]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
C[0][0]=16, C[15][15]=16 (expect 16)
*/
```

<br>

## 2. Independent Thread Scheduling (独立线程调度)

Independent Thread Scheduling (独立线程调度) gives every thread its own Program Counter (程序计数器) and call stack, so divergent branches (分支发散) inside a warp can make forward progress independently.

**Solved**: Pre-Volta warps executed in lockstep — divergent threads were masked off, which deadlocked any pattern where one thread waited on another in the same warp (e.g., a fine-grained spin-lock).

**Old**: A spin-lock between threads in the same warp deadlocks on Pascal because the holder is masked off.

```cpp
// On Pascal/older this deadlocks; on Volta+ it completes.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void buggyLock(int* lock, int* counter) {
    int tid = threadIdx.x;
    // Each thread tries to take the lock, increment, release
    while (atomicCAS(lock, 0, 1) != 0) { /* spin */ }
    int v = *counter;
    *counter = v + 1;
    atomicExch(lock, 0);
    if (tid == 0) printf("Done by thread 0\n");
}

int main() {
    int *dLock, *dCount;
    int zero = 0;
    cudaMalloc(&dLock, sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(dLock, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // 32 threads = one warp — would deadlock on Pascal lockstep semantics
    buggyLock<<<1, 32>>>(dLock, dCount);
    cudaDeviceSynchronize();

    int hCount;
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Final counter = %d (Pascal: hangs; Volta: works)\n", hCount);

    cudaFree(dLock); cudaFree(dCount);
    return 0;
}

/* Expected Output (Volta+):
Done by thread 0
Final counter = 32 (Pascal: hangs; Volta: works)
*/
```

**New**: Volta's independent scheduling lets the lock holder make progress while other threads spin — the same code runs correctly.

```cpp
// Compile: nvcc -arch=sm_70 file.cu
// Same code as above runs correctly on Volta+ thanks to independent thread scheduling.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpLock(int* lock, int* counter) {
    while (atomicCAS(lock, 0, 1) != 0) { /* spin */ }
    int v = *counter;
    *counter = v + 1;
    __threadfence();
    atomicExch(lock, 0);
}

int main() {
    int *dLock, *dCount;
    int zero = 0;
    cudaMalloc(&dLock, sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(dLock, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

    warpLock<<<1, 32>>>(dLock, dCount);
    cudaDeviceSynchronize();

    int hCount;
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Final counter = %d (Volta independent scheduling)\n", hCount);

    cudaFree(dLock); cudaFree(dCount);
    return 0;
}

/* Expected Output:
Final counter = 32 (Volta independent scheduling)
*/
```

<br>

## 3. Synchronized Warp Primitives (同步线程束原语)

Synchronized Warp Primitives (同步线程束原语) are `_sync` versions of warp intrinsics that take an explicit thread mask (线程掩码) — required from Volta onward because threads in a warp may execute independently.

**Solved**: With Independent Thread Scheduling, the implicit "all threads in the warp arrive together" assumption broke — `_sync` primitives restore correctness by explicitly synchronizing the specified threads.

### 1) `__shfl_sync` (同步洗牌指令)

`__shfl_sync` is the Volta+ replacement for `__shfl`, taking a thread mask as the first argument so the hardware knows exactly which lanes participate.

**Old**: `__shfl_down(value, offset)` — implicit warp synchronization, deprecated and undefined on Volta+.

```cpp
// Pre-Volta style — deprecated on sm_70+, may give wrong results.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpSumOld(const int* in, int* out) {
    int v = in[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);   // no mask — implicit
    }
    if (threadIdx.x == 0) *out = v;
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpSumOld<<<1, 32>>>(d, o);

    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Old shfl sum: %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Old shfl sum: 528
*/
```

**New**: `__shfl_down_sync(0xFFFFFFFF, v, offset)` — explicit full-warp mask, correct on Volta+.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpSumNew(const int* in, int* out) {
    int v = in[threadIdx.x];
    unsigned mask = 0xFFFFFFFF;   // all 32 lanes participate
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    if (threadIdx.x == 0) *out = v;
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpSumNew<<<1, 32>>>(d, o);

    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sync shfl sum: %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Sync shfl sum: 528
*/
```

<br>

## 4. Cooperative Groups (协作组)

Cooperative Groups (协作组) are a CUDA C++ API for explicitly grouping threads — a warp, a tile within a warp, a block, or even an entire grid — and synchronizing or communicating within that group via uniform `.sync()` and `.shfl()` methods.

**Solved**: Pre-Volta CUDA had only block-level `__syncthreads()` and ad-hoc warp intrinsics — there was no clean abstraction for sub-warp tiles or grid-wide synchronization.

**Old**: Hand-rolled warp reduction with raw `__shfl_down_sync` — no abstraction.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpReduceRaw(const int* in, int* out) {
    int v = in[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    if (threadIdx.x == 0) *out = v;
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpReduceRaw<<<1, 32>>>(d, o);
    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Raw warp sum: %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Raw warp sum: 528
*/
```

**New**: Cooperative Groups expose a typed `thread_block_tile<32>` with clean `.shfl_down()` and `.sync()` methods.

```cpp
// Compile: nvcc -arch=sm_70 file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void warpReduceCG(const int* in, int* out) {
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);

    int v = in[threadIdx.x];
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
        v += warp.shfl_down(v, offset);

    if (warp.thread_rank() == 0) *out = v;
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpReduceCG<<<1, 32>>>(d, o);
    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Cooperative-Groups warp sum: %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Cooperative-Groups warp sum: 528
*/
```

<br>

## 5. Warp-level Synchronization (线程束级同步)

Warp-level Synchronization (线程束级同步) uses `__syncwarp(mask)` to explicitly synchronize the specified lanes within a warp — required on Volta+ when threads converge after divergence.

**Solved**: Pre-Volta code relied on implicit warp-lockstep convergence after `if`/`else` branches; on Volta this no longer holds, so any read-after-write on shared data within a warp must explicitly synchronize.

**Old**: Implicit reconvergence — bug-prone on Volta+.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void implicitConverge(int* out) {
    __shared__ int s[32];
    int tid = threadIdx.x;

    if (tid < 16) {
        s[tid] = tid;
    } else {
        s[tid] = 100 + tid;
    }
    // No barrier — Pascal happened to converge here, Volta may not
    out[tid] = s[31 - tid];   // read across the divergence boundary
}

int main() {
    int h[32];
    int* d;
    cudaMalloc(&d, 32*sizeof(int));

    implicitConverge<<<1, 32>>>(d);
    cudaMemcpy(h, d, 32*sizeof(int), cudaMemcpyDeviceToHost);

    printf("h[0]=%d h[31]=%d (expect 131 and 0; may be undefined on Volta)\n", h[0], h[31]);

    cudaFree(d);
    return 0;
}

/* Expected Output (Volta: undefined, may show stale values):
h[0]=131 h[31]=0 (expect 131 and 0; may be undefined on Volta)
*/
```

**New**: `__syncwarp()` explicitly reconverges the warp before the cross-thread read.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void explicitConverge(int* out) {
    __shared__ int s[32];
    int tid = threadIdx.x;

    if (tid < 16) {
        s[tid] = tid;
    } else {
        s[tid] = 100 + tid;
    }
    __syncwarp();             // explicit reconvergence — required on Volta+
    out[tid] = s[31 - tid];
}

int main() {
    int h[32];
    int* d;
    cudaMalloc(&d, 32*sizeof(int));

    explicitConverge<<<1, 32>>>(d);
    cudaMemcpy(h, d, 32*sizeof(int), cudaMemcpyDeviceToHost);

    printf("h[0]=%d h[31]=%d\n", h[0], h[31]);

    cudaFree(d);
    return 0;
}

/* Expected Output:
h[0]=131 h[31]=0
*/
```

<br>
<br>