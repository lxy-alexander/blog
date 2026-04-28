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

# Tesla / Early CUDA (基础 CUDA 模型)

This generation established the basic CUDA execution model (执行模型) including Kernel Launch (内核启动), Grid (网格), Block (线程块), Thread (线程), Streams (流), Events (事件), and Pinned Memory (页锁定内存).

## 1. Architecture Diagram (架构图)

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428001139702" alt="image-20260428001139702" style="zoom:50%;" /> 

-   This file illustrates the architecture of NVIDIA Tesla（英伟达 Tesla）in the early CUDA（统一计算设备架构）era around 2008.
-   It uses the Tesla C1060 as a representative model.
-   It shows the PCIe Host Interface（PCIe 主机接口）, Streaming Multiprocessors / SMs（流式多处理器）, shared L2 cache / ROP（共享二级缓存 / 光栅操作单元）, memory controllers（显存控制器）, and GDDR3 graphics memory（GDDR3 显存）.
-   It highlights the 512-bit memory bus（512 位显存总线）used for high-bandwidth memory access.
-   It summarizes key specifications, including 240 CUDA cores（CUDA 核心）, 55 nm process technology（55 纳米制程）, 4 GB GDDR3 memory（4GB GDDR3 显存）, 102 GB/s memory bandwidth（显存带宽）, and 933 GFLOPS single-precision performance（单精度浮点性能）.

<br>

## 2. Streams (流)

A Stream (流) is a sequence of operations that execute in order on the GPU, while different streams (流) can overlap (重叠) for concurrency (并发).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the serial execution bottleneck (串行执行瓶颈) where all kernels (内核) and memcpys (内存拷贝) run one after another, leaving GPU resources idle (空闲).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for pipelined workloads (流水线工作负载) like multi-stage data processing (多阶段数据处理) where compute (计算) and memcpy (内存拷贝) can overlap.

### 3) What goes wrong without it? (不用它时有什么问题?)

Without streams (流), every operation uses the default stream (默认流) and runs sequentially, so memcpy and kernel never overlap and GPU utilization (GPU 利用率) stays low.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_kernel(float* x, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += val;
}

int main() {
    const int n = 1024;
    size_t bytes = n * sizeof(float);
    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes);
    cudaMemset(d_a, 0, bytes); cudaMemset(d_b, 0, bytes);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1); cudaStreamCreate(&s2);

    add_kernel<<<(n + 255) / 256, 256, 0, s1>>>(d_a, 1.0f, n);
    add_kernel<<<(n + 255) / 256, 256, 0, s2>>>(d_b, 2.0f, n);

    cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);

    float h_a, h_b;
    cudaMemcpy(&h_a, d_a, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    printf("a[0]=%.1f b[0]=%.1f\n", h_a, h_b);
    // Output: a[0]=1.0 b[0]=2.0

    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(d_a); cudaFree(d_b);
}
```

<br>

## 3. Events (事件)

Events (事件) are GPU-side timestamps (时间戳) used for precise kernel timing (精确计时) and inter-stream synchronization (流间同步).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the timing accuracy bottleneck (计时精度瓶颈) where CPU-side timers (CPU 计时器) include host-device sync overhead (主机-设备同步开销) and miss the real GPU duration (GPU 实际耗时).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for any kernel (任何内核) being benchmarked (基准测试) or profiled (性能分析), and for cross-stream dependencies (跨流依赖) like "wait until kernel A finishes."

### 3) What goes wrong without it? (不用它时有什么问题?)

Without events (事件), you measure with `cudaDeviceSynchronize` plus `clock()`, which inflates timing (放大计时) due to sync overhead and gives misleading benchmark results (错误的基准结果).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void busy_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        for (int i = 0; i < 100; i++) v = v * 1.001f + 0.001f;
        x[idx] = v;
    }
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *d_x;
    cudaMalloc(&d_x, bytes);
    cudaMemset(d_x, 0, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    busy_kernel<<<(n + 255) / 256, 256>>>(d_x, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("kernel time = %.3f ms\n", ms);
    // Output (example): kernel time = 0.150 ms

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_x);
}
```

<br>

## 4. Pinned Memory (页锁定内存)

Pinned Memory (页锁定内存) is host memory (主机内存) that is non-pageable (不可换页), enabling faster `cudaMemcpyAsync` (异步拷贝) and overlap with kernels (内核重叠).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the host-to-device transfer bottleneck (主机到设备传输瓶颈) caused by the OS paging memory (操作系统换页) — pinned pages let DMA (直接内存访问) move data directly.

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for kernels (内核) that need overlapped memcpy and compute (重叠传输与计算), and for high-frequency host-device exchange (高频主机-设备交换) like deep learning training (深度学习训练).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without pinned memory (页锁定内存), `cudaMemcpyAsync` quietly falls back to synchronous copy (悄悄退化为同步拷贝), so streams (流) cannot overlap and bandwidth (带宽) is roughly halved.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    float* h_pageable = (float*)malloc(bytes);
    float* h_pinned;
    cudaMallocHost(&h_pinned, bytes);

    for (int i = 0; i < n; i++) { h_pageable[i] = 1.0f; h_pinned[i] = 1.0f; }

    float *d_x;
    cudaMalloc(&d_x, bytes);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float t1, t2;

    cudaEventRecord(s);
    cudaMemcpy(d_x, h_pageable, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&t1, s, e);

    cudaEventRecord(s);
    cudaMemcpy(d_x, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&t2, s, e);

    printf("pageable=%.3f ms, pinned=%.3f ms\n", t1, t2);
    // Output (example): pageable=0.800 ms, pinned=0.350 ms

    free(h_pageable); cudaFreeHost(h_pinned); cudaFree(d_x);
    cudaEventDestroy(s); cudaEventDestroy(e);
}
```

<br>

## 5. Early Atomic Operations (早期原子操作)

Early CUDA (早期 CUDA) supported integer atomic operations (整数原子操作) on global memory (全局内存); float atomic (浮点原子操作) and double atomic (双精度原子操作) came in later compute capabilities (计算能力).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the race condition bottleneck (竞态条件瓶颈) where multiple threads (线程) try to update the same memory location, causing lost updates (丢失更新).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for counters (计数器), histograms (直方图), and lock-free data structures (无锁数据结构) where multiple threads (线程) write to the same address (地址).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without atomics (原子操作), shared writes (共享写) produce undefined results (未定义结果) — final values depend on the unpredictable interleaving (交错) of threads.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void count_kernel(int* counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(counter, 1);
}

int main() {
    const int n = 1024;
    int *d_counter, h_counter = 0;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    count_kernel<<<(n + 255) / 256, 256>>>(d_counter, n);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    printf("counter = %d\n", h_counter);
    // Output: counter = 1024

    cudaFree(d_counter);
}
```

<br>
