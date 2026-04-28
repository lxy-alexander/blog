---
title: "Pascal"
published: 2026-04-27
description: "Pascal"
image: ""
tags: ["cuda","gpu","Pascal"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:54:34.191.530739267Z"
---
# Pascal (统一内存更实用)

Pascal (帕斯卡) advanced Unified Memory (统一内存) from "convenient programming" (方便编程) to "actually usable" (真正可用) by enhancing page migration (页迁移) and finer-grained data migration (细粒度数据迁移).

## 1. Architecture Diagram (架构图)

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428002354936" alt="image-20260428002354936" style="zoom:50%;" /> 

 

<br>

## 2. Unified Memory (统一内存)

Unified Memory (统一内存) provides a single memory space (单一内存空间) accessible by both CPU and GPU, with the driver (驱动) automatically migrating data on demand.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_one(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += 1.0f;
}

int main() {
    const int n = 1024;
    float* x;

    // Single allocation accessible from both CPU and GPU (CPU/GPU 共享访问)
    cudaMallocManaged(&x, n * sizeof(float));

    // CPU writes (CPU 写入)
    for (int i = 0; i < n; i++) x[i] = (float)i;

    // GPU reads/writes — driver migrates pages automatically (驱动自动迁移页)
    add_one<<<(n + 255) / 256, 256>>>(x, n);
    cudaDeviceSynchronize();

    // CPU reads back without explicit copy (无需显式拷贝)
    printf("x[0]=%.1f x[1023]=%.1f\n", x[0], x[1023]);
    // Output: x[0]=1.0 x[1023]=1024.0

    cudaFree(x);
}
```

<br>

## 3. Managed Memory with Prefetch (托管内存与预取)

Pascal (帕斯卡) added `cudaMemPrefetchAsync` (异步预取) to give the programmer (程序员) hints about where data should reside (数据驻留位置).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(float* x, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= factor;
}

int main() {
    const int n = 1 << 20;
    float* x;
    cudaMallocManaged(&x, n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = 1.0f;

    int device = 0;
    cudaGetDevice(&device);

    // Prefetch to GPU before kernel — avoids page-fault stalls (避免缺页停顿)
    cudaMemPrefetchAsync(x, n * sizeof(float), device);

    scale_kernel<<<(n + 255) / 256, 256>>>(x, n, 3.0f);
    cudaDeviceSynchronize();

    // Prefetch back to CPU before host reads (回 CPU 前再预取)
    cudaMemPrefetchAsync(x, n * sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();

    printf("x[0]=%.1f x[%d]=%.1f\n", x[0], n - 1, x[n - 1]);
    // Output: x[0]=3.0 x[1048575]=3.0

    cudaFree(x);
}
```

<br>

## 4. Page Migration with Hints (页迁移提示)

Page Migration (页迁移) lets the GPU fault on access (按需缺页), then the driver moves the page; `cudaMemAdvise` (内存建议) optimizes the migration policy (迁移策略).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void read_only_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] * 2.0f;
}

int main() {
    const int n = 1024;
    float *x, *out;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    for (int i = 0; i < n; i++) x[i] = (float)i;

    int device = 0;
    cudaGetDevice(&device);

    // Tell driver: x is mostly read (告诉驱动: x 主要被读)
    cudaMemAdvise(x, n * sizeof(float), cudaMemAdviseSetReadMostly, device);
    // Tell driver: prefer GPU residency (建议数据在 GPU 上)
    cudaMemAdvise(out, n * sizeof(float), cudaMemAdviseSetPreferredLocation, device);

    cudaMemPrefetchAsync(x, n * sizeof(float), device);

    read_only_kernel<<<(n + 255) / 256, 256>>>(x, out, n);
    cudaDeviceSynchronize();

    printf("out[0]=%.1f out[1023]=%.1f\n", out[0], out[1023]);
    // Output: out[0]=0.0 out[1023]=2046.0

    cudaFree(x); cudaFree(out);
}
```

<br>