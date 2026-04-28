---
title: "Maxwell"
published: 2026-04-27
description: "Maxwell"
image: ""
tags: ["cuda","gpu","Maxwell"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:54:24.259.903477228Z"
---
# Maxwell (共享内存原子操作变快)

Maxwell (麦克斯韦) implemented native shared memory atomic operations (原生共享内存原子操作), especially for 32-bit integer atomics (32 位整数原子操作), making atomic-heavy kernels (原子操作密集型内核) like histogram (直方图) much faster.

## 1. Architecture Diagram (架构图)

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428002137876" alt="image-20260428002137876" style="zoom:50%;" /> 

-   This file illustrates the architecture of NVIDIA Maxwell（英伟达 Maxwell）around 2014.
-   It uses the GeForce GTX 980 as a representative GPU model.
-   It shows the PCIe Host Interface（PCIe 主机接口）, GPC Cluster（图形处理集群）, SMM units（Maxwell 流式多处理器）, L2 Cache（二级缓存）, memory controllers（显存控制器）, and GDDR5 graphics memory（GDDR5 显存）.
-   It highlights key Maxwell-era improvements, including high power efficiency（高能效）and memory compression（显存压缩）.
-   It summarizes key specifications, including 2048 CUDA cores（CUDA 核心）, 28 nm process technology（28 纳米制程）, 1126 / 1216 MHz base / boost clock（基础 / 加速频率）, 4 GB GDDR5 memory（4GB GDDR5 显存）, 256-bit memory bus（256 位显存总线）, 224 GB/s memory bandwidth（显存带宽）, and 4.61 TFLOPS single-precision performance（单精度浮点性能）.



Uses of the **Maxwell architecture（Maxwell 架构）** include:

-   **GeForce GTX 980**: The representative model shown in the diagram.
-   **GeForce GTX 970**: A popular high-end Maxwell GPU.
-   **GeForce GTX 960**: A mid-range Maxwell GPU.
-   **GeForce GTX 750 Ti**: One of the first Maxwell-based consumer GPUs.
-   **GeForce GTX Titan X**: High-end Maxwell GPU based on the GM200 core（GM200 核心）.
-   **GeForce GTX 980 Ti**: High-performance Maxwell GPU, also based on GM200.
-   **Tesla M40**: Maxwell-based data-center GPU for deep learning（深度学习）and HPC（高性能计算）.
-   **Tesla M60**: Maxwell-based data-center GPU mainly used for virtualization（虚拟化）and cloud graphics（云图形）.
-   **Quadro M2000 / M4000 / M5000 / M6000**: Professional workstation GPUs（专业工作站 GPU）based on Maxwell.

The most typical model for this architecture is **GeForce GTX 980**, while Maxwell was also used in **GeForce GTX 700/900 series, Tesla M series, and Quadro M series**. 

<br>

## 2. atomicAdd in Shared Memory (共享内存原子加)

`atomicAdd` (原子加) on shared memory (共享内存) became hardware-native (硬件原生) in Maxwell (麦克斯韦), making per-block accumulation (块内累加) extremely fast.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the slow-shared-atomics bottleneck (慢共享原子瓶颈) of pre-Maxwell GPUs where shared atomics (共享原子操作) were emulated with locks (锁模拟), making them barely faster than global atomics (全局原子操作).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for privatized histograms (私有化直方图), per-block counters (块内计数器), and any kernel (内核) that aggregates many threads' contributions (多线程贡献) into a small shared structure (小型共享结构).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without fast shared atomics (快速共享原子), all atomic updates (原子更新) must hit global memory (全局内存), creating contention (竞争) at hot spots (热点) and serializing throughput (串行化吞吐).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void shared_counter(int* global_count, int n) {
    __shared__ int local_count;
    if (threadIdx.x == 0) local_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&local_count, 1);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(global_count, local_count);
}

int main() {
    const int n = 1024;
    int *d_count, h_count = 0;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    shared_counter<<<4, 256>>>(d_count, n);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("count = %d\n", h_count);
    // Output: count = 1024

    cudaFree(d_count);
}
```

<br>

## 3. Shared Memory Atomics Performance (共享内存原子操作性能)

Shared memory atomics (共享内存原子操作) became ~100x faster (约 100 倍) than global memory atomics (全局原子操作), enabling efficient privatization (私有化) patterns.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the global-atomic-contention bottleneck (全局原子竞争瓶颈) where many warps (线程束) hammering the same global address (全局地址) serialize all updates (所有更新串行化).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for two-stage reduction patterns (两阶段归约模式): first accumulate locally in shared memory (先共享累加), then merge globally with one atomic per block (再每块一次全局原子).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without fast shared atomics (快速共享原子), the privatization optimization (私有化优化) is not worth the bookkeeping cost (簿记开销), and developers default to slow global atomics (慢全局原子).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void parallel_voting(const int* in, int* yes_count, int n) {
    __shared__ int s_yes;
    if (threadIdx.x == 0) s_yes = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && in[idx] > 0) atomicAdd(&s_yes, 1);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(yes_count, s_yes);
}

int main() {
    const int n = 1024;
    int h_in[1024];
    for (int i = 0; i < n; i++) h_in[i] = (i % 2 == 0) ? 1 : -1;

    int *d_in, *d_count, h_count = 0;
    cudaMalloc(&d_in, n * sizeof(int)); cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    parallel_voting<<<4, 256>>>(d_in, d_count, n);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("yes_count = %d\n", h_count);
    // Output: yes_count = 512

    cudaFree(d_in); cudaFree(d_count);
}
```

<br>

## 4. Privatized Histogram (私有化直方图)

Privatized histogram (私有化直方图) keeps a per-block (块内) histogram in shared memory (共享内存), then merges to global — the canonical Maxwell-era (麦克斯韦时代) optimization.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the histogram-hotspot bottleneck (直方图热点瓶颈) where many threads (线程) hit the same bin (区间) on global memory (全局内存), causing severe atomic contention (原子竞争).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for histogram (直方图), bucket sort (桶排序), counting sort (计数排序), and any kernel where each thread (线程) updates one of a small set of bins (少量桶).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without privatization (私有化), histogram throughput drops by 10-100x (吞吐下降 10-100 倍) under heavy contention (重度竞争), even with native shared atomics (原生共享原子) available.

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#define BINS 8

__global__ void privatized_hist(const int* in, int* hist, int n) {
    __shared__ int s_hist[BINS];
    if (threadIdx.x < BINS) s_hist[threadIdx.x] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&s_hist[in[idx] % BINS], 1);
    __syncthreads();

    if (threadIdx.x < BINS) atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
}

int main() {
    const int n = 64;
    int h_in[64], h_hist[BINS] = {0};
    for (int i = 0; i < n; i++) h_in[i] = i;

    int *d_in, *d_hist;
    cudaMalloc(&d_in, n * sizeof(int)); cudaMalloc(&d_hist, BINS * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BINS * sizeof(int));

    privatized_hist<<<2, 32>>>(d_in, d_hist, n);
    cudaMemcpy(h_hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BINS; i++) printf("bin[%d]=%d ", i, h_hist[i]);
    // Output: bin[0]=8 bin[1]=8 bin[2]=8 bin[3]=8 bin[4]=8 bin[5]=8 bin[6]=8 bin[7]=8

    cudaFree(d_in); cudaFree(d_hist);
}
```

<br>