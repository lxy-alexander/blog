---
title: "Tesla|Early CUDA"
published: 2026-04-27
description: "Tesla|Early CUDA"
image: ""
tags: ["cuda","gpu","Tesla|Early CUDA"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:45:16.878.289847808Z"
---
# Tesla / Early CUDA Architecture

The Tesla Architecture (特斯拉架构) is NVIDIA's first unified shader GPU architecture (2006), which introduced CUDA (统一计算设备架构) and made general-purpose GPU computing (通用 GPU 计算) practical for the first time.

**Representative GPU Models**:

- GeForce 8800 GTX (G80, 2006) — 16 SMs, 128 CUDA Cores
- GeForce GTX 280 (GT200, 2008) — 30 SMs, 240 CUDA Cores
- Tesla C870 / C1060 (data center cards)
- Quadro FX 5600

**Architecture Diagram** — G80 Example (16 SMs):

```
┌────────────────────────────────────────────────────────────────┐
│                        G80 GPU (Tesla)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Host Interface (PCIe)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Thread Execution Manager (Grid Scheduler)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│   TPC 0          TPC 1          ...          TPC 7             │
│  ┌──────┐      ┌──────┐                    ┌──────┐            │
│  │SM0|SM1│     │SM2|SM3│                   │SM14|SM15│         │
│  │ ───── │     │ ───── │                   │ ─────  │          │
│  │ 8 SP  │     │ 8 SP  │                   │ 8 SP   │          │
│  │ each  │     │ each  │     ...           │ each   │          │
│  │ 16KB  │     │ 16KB  │                   │ 16KB   │          │
│  │ Shmem │     │ Shmem │                   │ Shmem  │          │
│  └──────┘      └──────┘                    └──────┘            │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Global Memory (GDDR3, no L2 cache)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Total: 16 SMs × 8 SP = 128 CUDA Cores
```

**Solved**: Replaced the fixed-function graphics pipeline (固定功能图形管线) with unified programmable cores, enabling general-purpose computation on GPUs.

**Best Suited For**: Embarrassingly parallel tasks (高度并行任务) such as matrix multiplication (矩阵乘法), image processing (图像处理), and scientific simulation (科学计算).

<br>

## 1. Kernel Launch (核函数启动)

A Kernel (核函数) is a `__global__` function executed in parallel by many GPU threads (线程), launched from the host using the triple-angle-bracket syntax `<<<grid, block>>>`.

**Solved**: Before CUDA, GPGPU required mapping computation to graphics shaders (图形着色器) — Tesla introduced direct C-like kernel launch as the first available way, so there is no "old vs new" comparison here.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 4>>>();   // 1 block, 4 threads
    cudaDeviceSynchronize();
    return 0;
}

/* Expected Output (order may vary):
Hello from thread 0
Hello from thread 1
Hello from thread 2
Hello from thread 3
*/
```

<br>

## 2. Grid / Block / Thread (网格 / 块 / 线程)

CUDA organizes parallel work as a Grid (网格) of Blocks (块), where each Block contains many Threads (线程) that share Shared Memory (共享内存) and can synchronize.

**Solved**: Provides a scalable hierarchy (可扩展的层次结构) so the same kernel runs on GPUs with different SM counts without rewriting.

**Old**: CPU serial loop processes elements one at a time.

```cpp
#include <cstdio>

void addVectorsCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];   // serial loop
}

int main() {
    const int N = 8;
    float A[N] = {1,2,3,4,5,6,7,8};
    float B[N] = {10,20,30,40,50,60,70,80};
    float C[N] = {0};
    addVectorsCPU(A, B, C, N);
    for (int i = 0; i < N; ++i) printf("%g ", C[i]);
    printf("\n");
    return 0;
}

/* Expected Output:
11 22 33 44 55 66 77 88
*/
```

**New**: Launch a 2-level grid; each thread handles one element in parallel.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void addVectors(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index (全局线程索引)
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 8;
    float hA[N] = {1,2,3,4,5,6,7,8};
    float hB[N] = {10,20,30,40,50,60,70,80};
    float hC[N] = {0};

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N*sizeof(float));
    cudaMalloc(&dB, N*sizeof(float));
    cudaMalloc(&dC, N*sizeof(float));
    cudaMemcpy(dA, hA, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N*sizeof(float), cudaMemcpyHostToDevice);

    int block = 4;
    int grid  = (N + block - 1) / block;
    addVectors<<<grid, block>>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) printf("%g ", hC[i]);
    printf("\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
11 22 33 44 55 66 77 88
*/
```

<br>

## 3. Streams (流)

A CUDA Stream (流) is a queue of GPU operations that execute in issue order, while operations in different streams can run concurrently (并发执行).

**Solved**: Overlap (重叠) of host-device data transfer (主机-设备数据传输) and kernel execution to hide latency (隐藏延迟).

**Old**: All operations on the default stream (默认流) — strictly serial, copy-then-kernel-then-copy.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= k;
}

int main() {
    const int N = 4;
    float h[N] = {1,2,3,4};
    float *d1, *d2;
    cudaMalloc(&d1, N*sizeof(float));
    cudaMalloc(&d2, N*sizeof(float));

    // All on default stream — serial
    cudaMemcpy(d1, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scale<<<1, N>>>(d1, 2.0f, N);
    cudaMemcpy(d2, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scale<<<1, N>>>(d2, 3.0f, N);

    float r1[N], r2[N];
    cudaMemcpy(r1, d1, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(r2, d2, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("s1: ");
    for (int i = 0; i < N; ++i) printf("%g ", r1[i]);
    printf("\ns2: ");
    for (int i = 0; i < N; ++i) printf("%g ", r2[i]);
    printf("\n");

    cudaFree(d1); cudaFree(d2);
    return 0;
}

/* Expected Output:
s1: 2 4 6 8
s2: 3 6 9 12
*/
```

**New**: Multiple non-default streams + `cudaMemcpyAsync` enable concurrent execution.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= k;
}

int main() {
    const int N = 4;
    float* h;  cudaMallocHost(&h, N*sizeof(float)); // pinned for async copy
    h[0]=1; h[1]=2; h[2]=3; h[3]=4;
    float *d1, *d2;
    cudaMalloc(&d1, N*sizeof(float));
    cudaMalloc(&d2, N*sizeof(float));

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaMemcpyAsync(d1, h, N*sizeof(float), cudaMemcpyHostToDevice, s1);
    cudaMemcpyAsync(d2, h, N*sizeof(float), cudaMemcpyHostToDevice, s2);
    scale<<<1, N, 0, s1>>>(d1, 2.0f, N);
    scale<<<1, N, 0, s2>>>(d2, 3.0f, N);

    float r1[N], r2[N];
    cudaMemcpyAsync(r1, d1, N*sizeof(float), cudaMemcpyDeviceToHost, s1);
    cudaMemcpyAsync(r2, d2, N*sizeof(float), cudaMemcpyDeviceToHost, s2);
    cudaDeviceSynchronize();

    printf("s1: ");
    for (int i = 0; i < N; ++i) printf("%g ", r1[i]);
    printf("\ns2: ");
    for (int i = 0; i < N; ++i) printf("%g ", r2[i]);
    printf("\n");

    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFreeHost(h); cudaFree(d1); cudaFree(d2);
    return 0;
}

/* Expected Output:
s1: 2 4 6 8
s2: 3 6 9 12
*/
```

<br>

## 4. Events (事件)

A CUDA Event (事件) is a marker inserted into a stream used for precise GPU-side timing (GPU 端计时) and inter-stream synchronization (流间同步).

**Solved**: CPU-side `clock()` cannot accurately time asynchronous GPU work — events solve this.

**Old**: `clock()` around `cudaDeviceSynchronize()` — coarse, includes host overhead.

```cpp
#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>

__global__ void busy(float* x, int N, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        for (int k = 0; k < iters; ++k) v = v * 1.0001f + 0.001f;
        x[i] = v;
    }
}

int main() {
    const int N = 1 << 16;
    float* d;
    cudaMalloc(&d, N*sizeof(float));
    cudaMemset(d, 0, N*sizeof(float));

    clock_t t0 = clock();
    busy<<<(N+255)/256, 256>>>(d, N, 1000);
    cudaDeviceSynchronize();
    clock_t t1 = clock();

    double ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC;
    printf("CPU-timed kernel: %.2f ms (coarse)\n", ms);

    cudaFree(d);
    return 0;
}

/* Expected Output (value depends on GPU and CPU clock resolution):
CPU-timed kernel: 1.00 ms (coarse)
*/
```

**New**: `cudaEventRecord` + `cudaEventElapsedTime` measure pure GPU time.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void busy(float* x, int N, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = x[i];
        for (int k = 0; k < iters; ++k) v = v * 1.0001f + 0.001f;
        x[i] = v;
    }
}

int main() {
    const int N = 1 << 16;
    float* d;
    cudaMalloc(&d, N*sizeof(float));
    cudaMemset(d, 0, N*sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    busy<<<(N+255)/256, 256>>>(d, N, 1000);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU-timed kernel: %.2f ms\n", ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d);
    return 0;
}

/* Expected Output (value depends on GPU):
GPU-timed kernel: 1.23 ms
*/
```

<br>

## 5. Pinned Memory (锁页内存)

Pinned Memory (锁页内存 / 页锁定内存) is host memory locked from being paged out by the OS, enabling DMA (直接内存访问) and asynchronous transfers (异步传输).

**Solved**: Pageable memory (可分页内存) cannot be used for true async copy and provides lower bandwidth (带宽).

**Old**: `malloc` allocates pageable memory — slower, blocks `cudaMemcpyAsync`.

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* pageable = (float*)malloc(bytes);
    float* d;
    cudaMalloc(&d, bytes);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    cudaMemcpy(d, pageable, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);

    float t; cudaEventElapsedTime(&t, s, e);
    printf("Pageable H2D: %.2f ms\n", t);

    free(pageable); cudaFree(d);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}

/* Expected Output:
Pageable H2D: 0.85 ms
*/
```

**New**: `cudaMallocHost` allocates pinned memory — faster, async-capable.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* pinned;
    cudaMallocHost(&pinned, bytes);
    float* d;
    cudaMalloc(&d, bytes);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaEventRecord(s);
    cudaMemcpy(d, pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);

    float t; cudaEventElapsedTime(&t, s, e);
    printf("Pinned H2D: %.2f ms (typically 1.5x-3x faster)\n", t);

    cudaFreeHost(pinned); cudaFree(d);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}

/* Expected Output:
Pinned H2D: 0.32 ms (typically 1.5x-3x faster)
*/
```

<br>

## 6. Early Atomic Operations (早期原子操作)

Tesla introduced limited Atomic Operations (原子操作) on Global Memory (全局内存) for integers only — guaranteeing read-modify-write (读-改-写) without race conditions (竞争条件).

**Solved**: Allowed multiple threads to safely update shared counters (共享计数器) or histogram bins (直方图桶).

**Old**: Without atomics, plain `(*counter)++` from many threads loses updates.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void countEvensRacy(const int* data, int N, int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (data[i] % 2 == 0)) {
        (*counter)++;   // race condition — undercounts
    }
}

int main() {
    const int N = 10000;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i; // 5000 even values

    int *d, *dCount;
    int hCount = 0;
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &hCount, sizeof(int), cudaMemcpyHostToDevice);

    countEvensRacy<<<(N+255)/256, 256>>>(d, N, dCount);
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Even count without atomics: %d (should be 5000)\n", hCount);

    free(h); cudaFree(d); cudaFree(dCount);
    return 0;
}

/* Expected Output (wrong, value varies per run):
Even count without atomics: 312 (should be 5000)
*/
```

**New**: Native `atomicAdd` on integer global memory — correct under contention.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void countEvens(const int* data, int N, int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (data[i] % 2 == 0)) {
        atomicAdd(counter, 1);
    }
}

int main() {
    const int N = 10000;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i; // 5000 even values

    int *d, *dCount;
    int hCount = 0;
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &hCount, sizeof(int), cudaMemcpyHostToDevice);

    countEvens<<<(N+255)/256, 256>>>(d, N, dCount);
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Even count: %d\n", hCount);

    free(h); cudaFree(d); cudaFree(dCount);
    return 0;
}

/* Expected Output:
Even count: 5000
*/
```

<br>
<br>
