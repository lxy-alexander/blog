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

# Fermi (缓存与原子操作变强)

Fermi (费米) made CUDA more like a modern GPU computing platform by introducing L1/L2 Cache (一级/二级缓存), stronger Shared Memory (共享内存), and more practical Atomic Operations (原子操作).



## 1. Architecture Diagram (架构图)

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428001506266" alt="image-20260428001506266" style="zoom:50%;" /> 

-   This file illustrates the architecture of NVIDIA Fermi（英伟达 Fermi）around 2010.
-   It uses the Tesla C2050 as a representative GPU model.
-   It shows the PCIe Host Interface（PCIe 主机接口）, Graphics Processing Clusters / GPCs（图形处理集群）, Streaming Multiprocessors / SMs（流式多处理器）, L2 Cache（二级缓存）, memory controllers（显存控制器）, and GDDR5 graphics memory（GDDR5 显存）.
-   It highlights major Fermi-era improvements, including ECC（错误检查与纠正）support and enhanced double-precision performance（增强双精度性能）.
-   It summarizes key specifications, including 448 CUDA cores（CUDA 核心）, 40 nm process technology（40 纳米制程）, 1.15 GHz core frequency（核心频率）, 3 GB GDDR5 memory（3GB GDDR5 显存）, 384-bit memory bus（384 位显存总线）, 144 GB/s memory bandwidth（显存带宽）, and 1.03 TFLOPS single-precision performance（单精度浮点性能）.



GPU models that used the **Fermi architecture（Fermi 架构）** include:

-   **Tesla C2050**: The representative model shown in the diagram, mainly used for HPC（高性能计算）and CUDA computing（CUDA 计算）.
-   **Tesla C2070**: A Fermi-based Tesla GPU with larger memory capacity.
-   **Tesla M2050 / M2070**: Server and data-center GPUs based on the Fermi architecture.
-   **GeForce GTX 480**: A high-end consumer GPU based on the GF100 core（GF100 核心）.
-   **GeForce GTX 470**: Another high-end GeForce model using the Fermi architecture.
-   **GeForce GTX 580**: An improved Fermi GPU based on the GF110 core（GF110 核心）.
-   **GeForce GTX 570**: Also based on the GF110 core.
-   **Quadro 4000 / 5000 / 6000**: Professional workstation GPUs using the Fermi architecture.

The most typical model for this architecture is **Tesla C2050**, while Fermi was also widely used in **Tesla C/M series, GeForce GTX 400/500 series, and Quadro professional GPUs（专业图形 GPU）**.

<br>

## 2. atomicAdd Becomes Practical (原子加变实用)

`atomicAdd` (原子加) became fast enough for practical use in Fermi (费米), enabling patterns like histogram (直方图) and counter (计数器) without major performance loss.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the slow-atomics bottleneck (慢原子操作瓶颈) of pre-Fermi GPUs where atomics (原子操作) were so slow they were avoided in performance-critical kernels (性能关键内核).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for histogram (直方图), reduction-with-conflict (有冲突的归约), and any kernel (内核) that gathers global statistics (全局统计) from many threads (线程).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without practical atomics (原子操作), histograms must use multi-pass algorithms (多次扫描算法) or per-thread privatization plus reduction (每线程私有化加归约), adding significant code complexity (代码复杂度).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void histogram_atomic(const int* in, int* hist, int n, int bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = in[idx] % bins;
        atomicAdd(&hist[b], 1);
    }
}

int main() {
    const int n = 32, bins = 4;
    int h_in[32], h_hist[4] = {0};
    for (int i = 0; i < n; i++) h_in[i] = i;

    int *d_in, *d_hist;
    cudaMalloc(&d_in, n * sizeof(int)); cudaMalloc(&d_hist, bins * sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, bins * sizeof(int));

    histogram_atomic<<<1, n>>>(d_in, d_hist, n, bins);
    cudaMemcpy(h_hist, d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < bins; i++) printf("bin[%d]=%d ", i, h_hist[i]);
    // Output: bin[0]=8 bin[1]=8 bin[2]=8 bin[3]=8

    cudaFree(d_in); cudaFree(d_hist);
}
```

<br>

## 3. Global Memory Atomics (全局内存原子操作)

Fermi (费米) matured global memory atomics (全局内存原子操作), supporting `atomicAdd`, `atomicMax`, `atomicMin`, `atomicCAS` (比较交换), and `atomicExch` (交换) on integers.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the inter-block coordination bottleneck (块间协调瓶颈) where blocks (线程块) cannot use `__syncthreads()` to coordinate, but still need to update shared global state (共享全局状态).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for global reductions (全局归约), atomic min/max searches (原子最值搜索), and lock-free queues (无锁队列) using `atomicCAS` (比较交换).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without global atomics (全局原子操作), inter-block coordination requires a kernel relaunch (重启内核) or atomic locks (原子锁) built from `atomicCAS`, both adding latency (延迟).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void find_max(const int* in, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicMax(result, in[idx]);
}

__global__ void find_min(const int* in, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicMin(result, in[idx]);
}

int main() {
    const int n = 8;
    int h_in[8] = {3, 1, 4, 1, 5, 9, 2, 6};
    int *d_in, *d_max, *d_min;
    int h_max = -1, h_min = 1000000;

    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_max, sizeof(int)); cudaMalloc(&d_min, sizeof(int));
    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice);

    find_max<<<1, n>>>(d_in, d_max, n);
    find_min<<<1, n>>>(d_in, d_min, n);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    printf("max=%d min=%d\n", h_max, h_min);
    // Output: max=9 min=1

    cudaFree(d_in); cudaFree(d_max); cudaFree(d_min);
}
```

<br>

## 4. Shared Memory and Cache (共享内存与缓存)

Fermi (费米) introduced configurable on-chip memory (片上内存) where Shared Memory (共享内存) and L1 Cache (一级缓存) physically share storage (共用存储), tunable via `cudaFuncSetCacheConfig`.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the global memory latency bottleneck (全局内存延迟瓶颈) by giving each kernel (内核) a fast on-chip scratchpad (片上暂存) plus a hardware cache (硬件缓存) for repeated access.

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for tiled algorithms (分块算法) like matrix multiplication (矩阵乘法), stencils (模板计算), and reductions (归约) that reuse data within a block (块内复用数据).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without shared memory and cache (共享内存与缓存), every data access hits global memory (全局内存) at ~400-cycle latency (延迟), making most kernels memory-bound (内存受限) and slow.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 256;
    float h_in[256], h_out;
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;

    cudaFuncSetCacheConfig(block_sum, cudaFuncCachePreferShared);

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    block_sum<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum = %.1f\n", h_out);
    // Output: sum = 256.0

    cudaFree(d_in); cudaFree(d_out);
}
```

<br>
