---
title: "Fermi"
published: 2026-04-27
description: "Fermi"
image: ""
tags: ["cuda","gpu","Fermi"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:47:21.575.901990647Z"
---
# Fermi Architecture (费米架构)

The Fermi Architecture (费米架构, 2010) is NVIDIA's first GPU with a true cache hierarchy (缓存层次结构) — adding configurable L1 cache (一级缓存), unified L2 cache (统一二级缓存), ECC memory (ECC 显存), and full IEEE-754 double precision (双精度浮点) — making GPUs viable for HPC (高性能计算).

**Representative GPU Models**:

- GeForce GTX 480 / GTX 580 (GF100 / GF110) — 15–16 SMs, 480–512 CUDA Cores
- GeForce GTX 460 / GTX 560 Ti (GF104 / GF114)
- Tesla C2050 / C2070 / M2090 (data center)
- Quadro 6000

**Architecture Diagram** — GF100 Example (16 SMs):

```
┌──────────────────────────────────────────────────────────────────┐
│                       GF100 GPU (Fermi)                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Host Interface (PCIe 2.0)                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  GigaThread Engine (Scheduler)             │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│   GPC 0           GPC 1           GPC 2           GPC 3          │
│  ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐        │
│  │SM SM │        │SM SM │        │SM SM │        │SM SM │        │
│  │SM SM │        │SM SM │        │SM SM │        │SM SM │        │
│  │      │        │      │        │      │        │      │        │
│  │ each │        │ each │        │ each │        │ each │        │
│  │ SM:  │        │ SM:  │        │ SM:  │        │ SM:  │        │
│  │ 32 SP│        │ 32 SP│        │ 32 SP│        │ 32 SP│        │
│  │ 64KB │        │ 64KB │        │ 64KB │        │ 64KB │        │
│  │L1+Sh │        │L1+Sh │        │L1+Sh │        │L1+Sh │        │
│  └──────┘        └──────┘        └──────┘        └──────┘        │
│       │              │              │              │             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │             Unified L2 Cache (768 KB)                      │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │           Global Memory (GDDR5, 384-bit, ECC)              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

Total: 16 SMs × 32 SP = 512 CUDA Cores
Each SM: 64 KB on-chip, configurable as 16/48 or 48/16 (L1 / Shared)
```

**Solved**: Tesla had no cache hierarchy and weak double precision — Fermi added L1/L2 caches, ECC, full IEEE-754 FP64, shared-memory atomics, and `atomicAdd` for `float`.

**Best Suited For**: HPC workloads (高性能计算负载), scientific simulation (科学计算), CUDA-aware libraries (cuBLAS, cuFFT), and irregular memory-access patterns (不规则访存模式).

<br>

## 1. L1 Cache (一级缓存)

The L1 Cache (L1 缓存) on Fermi is a per-SM (每个 SM) on-chip cache that automatically caches Global Memory (全局内存) loads, sharing 64 KB with Shared Memory in a configurable 16/48 or 48/16 split.

**Solved**: On Tesla, every global load went to DRAM (DRAM 显存) — Fermi caches them, dramatically reducing latency for repeated accesses (重复访存).

**Old**: Manual staging through Shared Memory for every reuse — programmer must explicitly load and `__syncthreads`.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void readSumManual(const float* x, float* out, int N) {
    __shared__ float tile[128];
    int tid = threadIdx.x;
    if (tid < 128) tile[tid] = x[tid];   // manual stage into shared
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < 32; ++k) s += tile[i % 128];  // reuse from shared
    out[i] = s;
}

int main() {
    const int N = 1024;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *o;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&o, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);

    readSumManual<<<(N+127)/128, 128>>>(d, o, N);

    float r[4];
    cudaMemcpy(r, o, 4*sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0..3]: %g %g %g %g\n", r[0], r[1], r[2], r[3]);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
out[0..3]: 32 32 32 32
*/
```

**New**: Configure cache preference and let hardware cache automatically — code stays simple.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void readSum(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < 32; ++k) s += x[i % 128];   // L1 cache catches reuse
    out[i] = s;
}

int main() {
    const int N = 1024;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *o;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&o, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(readSum, cudaFuncCachePreferL1); // 48 KB L1
    readSum<<<(N+127)/128, 128>>>(d, o, N);

    float r[4];
    cudaMemcpy(r, o, 4*sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0..3]: %g %g %g %g\n", r[0], r[1], r[2], r[3]);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
out[0..3]: 32 32 32 32
*/
```

<br>

## 2. L2 Cache (二级缓存)

The L2 Cache (L2 缓存) on Fermi is a unified (统一的), GPU-wide (全 GPU 共享) cache (768 KB on GF100) sitting between all SMs and Global Memory, serving as a coherence point (一致性节点) for inter-SM data sharing.

**Solved**: Threads in different SMs touching the same data previously had to re-fetch from DRAM — L2 captures this reuse automatically.

**Old**: Tesla had no L2; programmers had to coalesce (合并访存) and avoid re-reads at all costs by doing one-shot loads into per-block shared memory.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void crossSmManual(const float* lookup, float* out, int N) {
    __shared__ float local[64];
    int tid = threadIdx.x;
    if (tid < 64) local[tid] = lookup[tid];   // every block re-loads from DRAM
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < 64; ++k) s += local[k];
    out[i] = s;
}

int main() {
    const int N = 4096;
    float *d, *o;
    cudaMalloc(&d, 64*sizeof(float));
    cudaMalloc(&o, N*sizeof(float));

    float h[64];
    for (int i = 0; i < 64; ++i) h[i] = 1.0f;
    cudaMemcpy(d, h, 64*sizeof(float), cudaMemcpyHostToDevice);

    crossSmManual<<<(N+127)/128, 128>>>(d, o, N);

    float r[2];
    cudaMemcpy(r, o, 2*sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0]=%g, out[1]=%g\n", r[0], r[1]);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
out[0]=64, out[1]=64
*/
```

**New**: Read directly from global memory; the unified L2 cache catches cross-SM reuse automatically.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void crossSmReuse(const float* lookup, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int k = 0; k < 64; ++k) s += lookup[k];   // L2 hits across SMs
    out[i] = s;
}

int main() {
    const int N = 4096;
    float *d, *o;
    cudaMalloc(&d, 64*sizeof(float));
    cudaMalloc(&o, N*sizeof(float));

    float h[64];
    for (int i = 0; i < 64; ++i) h[i] = 1.0f;
    cudaMemcpy(d, h, 64*sizeof(float), cudaMemcpyHostToDevice);

    crossSmReuse<<<(N+127)/128, 128>>>(d, o, N);

    float r[2];
    cudaMemcpy(r, o, 2*sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0]=%g, out[1]=%g\n", r[0], r[1]);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
out[0]=64, out[1]=64
*/
```

<br>

## 3. Shared Memory (共享内存)

Shared Memory (共享内存) is per-block (每个块独享), low-latency on-chip memory enabling thread cooperation (线程协作) within a block; on Fermi it shares 64 KB with L1 and supports atomics (原子操作).

**Solved**: Avoids redundant Global Memory loads (冗余全局内存读取) and enables fast inter-thread communication (线程间通信).

**Old**: Tesla provided only 16 KB of shared memory per SM and no shared-memory atomics — large reductions had to be split into multiple smaller blocks.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// Tesla-style: small block (limited by 16 KB shared memory) + manual reduction
__global__ void blockReduceSmall(const float* in, float* out, int N) {
    __shared__ float sdata[64];   // small to fit Tesla's 16 KB
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]); // global atomic to merge blocks
}

int main() {
    const int N = 256;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *o;
    float zero = 0.f;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&o, sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(o, &zero, sizeof(float), cudaMemcpyHostToDevice);

    blockReduceSmall<<<4, 64>>>(d, o, N);  // 4 small blocks

    float r;
    cudaMemcpy(&r, o, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Total sum (small blocks) = %g\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Total sum (small blocks) = 256
*/
```

**New**: Fermi's 48 KB shared memory enables a single large block to reduce in one pass.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void blockReduce(const float* in, float* out, int N) {
    __shared__ float sdata[256];   // larger block possible on Fermi
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 256;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *o;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&o, sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);

    blockReduce<<<1, 256>>>(d, o, N);

    float r;
    cudaMemcpy(&r, o, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Block sum = %g\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Block sum = 256
*/
```

<br>

## 4. Global Memory Atomics (全局内存原子操作)

Fermi greatly expanded Global Memory Atomics (全局内存原子操作) — adding 64-bit integer atomics, `atomicAdd` on `float`, and accelerating them via the L2 cache (经由 L2 缓存加速).

**Solved**: Tesla atomics were slow (DRAM round-trip, DRAM 往返) and limited to 32-bit integers — Fermi made atomics fast and broadly typed.

**Old**: No `float` atomics — must emulate via `atomicCAS` (compare-and-swap, 比较并交换) loop.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__device__ float atomicAddFloat_CAS(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        float updated = __int_as_float(assumed) + val;
        old = atomicCAS(addr_i, assumed, __float_as_int(updated));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void floatSumCAS(const float* in, float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAddFloat_CAS(sum, in[i]);
}

int main() {
    const int N = 100;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 0.5f;

    float *d, *dSum;
    float hSum = 0.f;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&dSum, sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSum, &hSum, sizeof(float), cudaMemcpyHostToDevice);

    floatSumCAS<<<(N+31)/32, 32>>>(d, dSum, N);
    cudaMemcpy(&hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Total (CAS loop) = %g\n", hSum);

    cudaFree(d); cudaFree(dSum);
    return 0;
}

/* Expected Output:
Total (CAS loop) = 50
*/
```

**New**: Native `atomicAdd(float*)` on global memory — one line.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void floatAtomicSum(const float* in, float* sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(sum, in[i]);   // Fermi feature
}

int main() {
    const int N = 100;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 0.5f;

    float *d, *dSum;
    float hSum = 0.f;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&dSum, sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dSum, &hSum, sizeof(float), cudaMemcpyHostToDevice);

    floatAtomicSum<<<(N+31)/32, 32>>>(d, dSum, N);
    cudaMemcpy(&hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Total = %g\n", hSum);

    cudaFree(d); cudaFree(dSum);
    return 0;
}

/* Expected Output:
Total = 50
*/
```

<br>

## 5. `atomicAdd` (原子加)

`atomicAdd` performs an indivisible (不可分割的) read-modify-write addition on a memory location, guaranteeing correctness when many threads update the same address concurrently.

**Solved**: Race condition (竞争条件) on `*p += v;` — without atomicity, lost updates (丢失更新) corrupt results.

**Old**: Plain `*p += v;` from many threads — non-deterministic, undercounts.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void wrongIncrement(int* counter, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) (*counter)++;       // race condition!
}

int main() {
    const int N = 10000;
    int *dWrong;
    int wrong = 0;

    cudaMalloc(&dWrong, sizeof(int));
    cudaMemcpy(dWrong, &wrong, sizeof(int), cudaMemcpyHostToDevice);

    wrongIncrement<<<(N+255)/256, 256>>>(dWrong, N);
    cudaMemcpy(&wrong, dWrong, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Without atomicAdd: %d (should be %d)\n", wrong, N);

    cudaFree(dWrong);
    return 0;
}

/* Expected Output (wrong value varies per run):
Without atomicAdd: 312 (should be 10000)
*/
```

**New**: `atomicAdd(p, v);` — correct under any contention.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void correctIncrement(int* counter, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(counter, 1);
}

int main() {
    const int N = 10000;
    int *dRight;
    int right = 0;

    cudaMalloc(&dRight, sizeof(int));
    cudaMemcpy(dRight, &right, sizeof(int), cudaMemcpyHostToDevice);

    correctIncrement<<<(N+255)/256, 256>>>(dRight, N);
    cudaMemcpy(&right, dRight, sizeof(int), cudaMemcpyDeviceToHost);

    printf("With atomicAdd: %d\n", right);

    cudaFree(dRight);
    return 0;
}

/* Expected Output:
With atomicAdd: 10000
*/
```

<br>
<br>