---
title: "Kepler"
published: 2026-04-27
description: "Kepler"
image: ""
tags: ["cuda","gpu","Kepler"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:52:30.335.569021499Z"
---

# Kepler (Warp 内通信出现)

Kepler (开普勒) is the generation where Warp-level Primitives (Warp 级原语) became important — it introduced Warp Shuffle Functions (Warp 洗牌函数), requiring Compute Capability (计算能力) 3.0 or higher.

## 1. Architecture Diagram

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428001909777" alt="image-20260428001909777" style="zoom:50%;" /> 

-   This file illustrates the architecture of NVIDIA Kepler（英伟达 Kepler）around 2012.
-   It uses the Tesla K20X as a representative GPU model.
-   It shows the PCIe Host Interface（PCIe 主机接口）, GPC Cluster（图形处理集群）, SMX units（流式多处理器扩展单元）, L2 Cache（二级缓存）, memory controllers（显存控制器）, and GDDR5 graphics memory（GDDR5 显存）.
-   It highlights key Kepler-era features, including Hyper-Q（多队列并行调度技术）and Dynamic Parallelism（动态并行）.
-   It summarizes key specifications, including 2688 CUDA cores（CUDA 核心）, 28 nm process technology（28 纳米制程）, 732 MHz core frequency（核心频率）, 6 GB GDDR5 memory（6GB GDDR5 显存）, 384-bit memory bus（384 位显存总线）, 250 GB/s memory bandwidth（显存带宽）, and 3.95 TFLOPS single-precision performance（单精度浮点性能）.

Uses of the **Kepler architecture（Kepler 架构）** include:

-   **Tesla K20X**: The representative model shown in the diagram, mainly used for HPC（高性能计算）and CUDA computing（CUDA 计算）.
-   **Tesla K20 / K20Xm**: Kepler-based GPUs for servers and supercomputing.
-   **Tesla K40 / K40c / K40m**: Later Kepler Tesla GPUs with larger memory capacity and higher performance.
-   **Tesla K80**: Dual-GPU Kepler accelerator widely used in data centers.
-   **GeForce GTX 680**: High-end consumer GPU based on Kepler.
-   **GeForce GTX 670 / GTX 660 Ti**: Consumer gaming GPUs using the Kepler architecture.
-   **GeForce GTX 780 / GTX 780 Ti / GTX Titan**: Higher-end Kepler GPUs based on GK110（GK110 核心）.
-   **Quadro K2000 / K4000 / K5000 / K6000**: Professional workstation GPUs using the Kepler architecture.

The most typical model for this architecture is **Tesla K20X**, while Kepler was also widely used in **Tesla K series, GeForce GTX 600/700 series, Titan, and Quadro K series professional GPUs（专业图形 GPU）**.

<br>

## 2. Warp-level Primitives (Warp 级原语)

Warp-level primitives (Warp 级原语) let threads (线程) in the same warp (线程束) exchange register values (寄存器值) directly, bypassing shared memory (共享内存).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void warp_broadcast(int* out) {
    int tid = threadIdx.x;
    int val = tid * 10;
    // Broadcast lane 0's value to all lanes (将 lane 0 的值广播到所有 lane)
    int broadcast = __shfl_sync(0xffffffff, val, 0);
    out[tid] = broadcast;
}

int main() {
    int *d_out, h_out[32];
    cudaMalloc(&d_out, 32 * sizeof(int));
    warp_broadcast<<<1, 32>>>(d_out);
    cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) printf("%d ", h_out[i]);
    // Output: 0 0 0 0  (all lanes get lane 0's value, which is 0)

    cudaFree(d_out);
}
```

<br>

## 3. Early `__shfl` (Warp 内数据交换)

The original `__shfl` (Warp 洗牌) on Kepler (开普勒) had no explicit mask (掩码) — modern code should use `__shfl_sync` from Volta (沃尔塔) onward, but the principle is the same.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void warp_shift(int* out) {
    int tid = threadIdx.x;
    int val = tid;
    // Shift values down by 1 lane (向下平移 1 个 lane)
    int shifted = __shfl_up_sync(0xffffffff, val, 1);
    out[tid] = shifted;
}

int main() {
    int *d_out, h_out[8];
    cudaMalloc(&d_out, 8 * sizeof(int));
    warp_shift<<<1, 8>>>(d_out);
    cudaMemcpy(h_out, d_out, 8 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; i++) printf("%d ", h_out[i]);
    // Output: 0 0 1 2 3 4 5 6  (lane 0 keeps its own value)

    cudaFree(d_out);
}
```

<br>

## 4. Dynamic Parallelism (动态并行)

Dynamic Parallelism (动态并行) lets a kernel (内核) launch other kernels (内核) directly from the device (设备) without going back to the host (主机).

```cpp
// Compile with: nvcc -rdc=true dynamic_parallelism.cu -o app
#include <cuda_runtime.h>
#include <cstdio>

__global__ void child_kernel(float* x, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += val;
}

__global__ void parent_kernel(float* x, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Launch child kernel from device (从设备端启动子内核)
        child_kernel<<<(n + 31) / 32, 32>>>(x, n, 5.0f);
        cudaDeviceSynchronize();
    }
}

int main() {
    const int n = 64;
    float h_x[64], *d_x;
    for (int i = 0; i < n; i++) h_x[i] = 1.0f;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    parent_kernel<<<1, 1>>>(d_x, n);
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("x[0]=%.1f x[63]=%.1f\n", h_x[0], h_x[63]);
    // Output: x[0]=6.0 x[63]=6.0  (1.0 + 5.0)

    cudaFree(d_x);
}
```

<br>

## 5. Hyper-Q (多硬件工作队列)

Hyper-Q (多硬件工作队列) provides 32 hardware work queues (硬件队列), allowing multiple CPU streams (CPU 流) or processes to feed the GPU concurrently (并发) without serialization (串行化).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void compute_kernel(float* x, int n, float v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += v;
}

int main() {
    const int n = 4096, NSTREAMS = 4;
    size_t bytes = n * sizeof(float);
    float* d_x[NSTREAMS];
    cudaStream_t streams[NSTREAMS];

    for (int i = 0; i < NSTREAMS; i++) {
        cudaMalloc(&d_x[i], bytes);
        cudaMemset(d_x[i], 0, bytes);
        cudaStreamCreate(&streams[i]);  // Each stream uses its own queue (每流独立队列)
    }

    // All 4 streams can run concurrently thanks to Hyper-Q (Hyper-Q 让 4 流并发)
    for (int i = 0; i < NSTREAMS; i++)
        compute_kernel<<<(n + 255) / 256, 256, 0, streams[i]>>>(d_x[i], n, (float)(i + 1));

    cudaDeviceSynchronize();

    float h_v;
    for (int i = 0; i < NSTREAMS; i++) {
        cudaMemcpy(&h_v, d_x[i], sizeof(float), cudaMemcpyDeviceToHost);
        printf("stream %d: x[0]=%.1f\n", i, h_v);
    }
    // Output: stream 0: x[0]=1.0
    //         stream 1: x[0]=2.0
    //         stream 2: x[0]=3.0
    //         stream 3: x[0]=4.0

    for (int i = 0; i < NSTREAMS; i++) {
        cudaFree(d_x[i]); cudaStreamDestroy(streams[i]);
    }
}
```

<br>
