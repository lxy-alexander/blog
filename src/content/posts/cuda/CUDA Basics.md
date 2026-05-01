---
title: "CUDA Basics"
published: 2026-04-27
description: "CUDA Basics"
image: ""
tags: ["cuda","CUDA Basics"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-28T01:59:14.954.063180661Z"
---

# CUDA Basics (CUDA 基础入门)

Goal: independently write simple CUDA kernels (内核函数) and understand how threads (线程) map to data.

## 1. Concepts to Learn

### 1) Host (主机) and Device (设备)

The Host (主机) is the CPU side that runs serial code, while the Device (设备) is the GPU side that runs parallel kernels (内核函数).

### 2) Kernel (内核函数)

A Kernel (内核函数) is a function marked with `__global__` that runs on the GPU and is launched by the CPU.

### 3) Grid (网格)

A Grid (网格) is the collection of all thread blocks (线程块) launched by a single kernel call.

### 4) Block (线程块)

A Block (线程块) is a group of threads (线程) that can cooperate via shared memory (共享内存) and synchronize with each other.

### 5) Thread (线程)

A Thread (线程) is the smallest execution unit (执行单元) on the GPU, and each thread runs the same kernel code on different data.

### 6) Warp (线程束)

A Warp (线程束) is a group of 32 threads (线程) that execute the same instruction in lockstep (SIMT, 单指令多线程).

### 7) Built-in Variables (内建变量)

`threadIdx` is the thread's index within its block (线程块), `blockIdx` is the block's index within the grid (网格), `blockDim` is the block size (线程块大小), and `gridDim` is the grid size (网格大小).

### 8) Global Memory (全局内存)

Global Memory (全局内存) is the GPU's main DRAM (显存), accessible by all threads (线程) but with high latency (高延迟).

### 9) Kernel Launch (内核启动)

Kernel Launch (内核启动) uses the `<<<gridDim, blockDim>>>` syntax to specify how many blocks (线程块) and threads (线程) to launch.

### 10) Boundary Check (边界检查)

Boundary Check (边界检查) is the `if (idx < n)` guard to prevent extra threads (线程) from accessing out-of-range memory (越界内存).

<br>

## 2. Code to Write

### 1) vector_add.cu (向量加法)

Vector addition (向量加法) is the "Hello World" of CUDA, where each thread (线程) computes one output element.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];  // Boundary check (边界检查)
}

int main() {
    const int n = 8;
    size_t bytes = n * sizeof(float);
    float h_a[n] = {1,2,3,4,5,6,7,8}, h_b[n] = {8,7,6,5,4,3,2,1}, h_c[n];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vector_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_c[i]);
    // Output: 9.0 9.0 9.0 9.0 9.0 9.0 9.0 9.0
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

### 2) saxpy.cu (单精度 a*x+y)

SAXPY (Single-precision A·X Plus Y) computes `y = a * x + y` and is the canonical BLAS Level 1 (一级 BLAS) benchmark.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a * x[idx] + y[idx];
}

int main() {
    const int n = 5;
    size_t bytes = n * sizeof(float);
    float h_x[n] = {1,2,3,4,5}, h_y[n] = {10,10,10,10,10};

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

    saxpy<<<(n + 255) / 256, 256>>>(2.0f, d_x, d_y, n);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_y[i]);
    // Output: 12.0 14.0 16.0 18.0 20.0
    cudaFree(d_x); cudaFree(d_y);
}
```

### 3) relu.cu (ReLU 激活函数)

ReLU (修正线性单元, Rectified Linear Unit) is `max(0, x)`, a fully element-wise (逐元素) operation perfect for GPU parallelism (并行).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
}

int main() {
    const int n = 6;
    size_t bytes = n * sizeof(float);
    float h_x[n] = {-2, -1, 0, 1, 2, 3}, h_y[n];

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    relu<<<(n + 255) / 256, 256>>>(d_x, d_y, n);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_y[i]);
    // Output: 0.0 0.0 0.0 1.0 2.0 3.0
    cudaFree(d_x); cudaFree(d_y);
}
```

### 4) sigmoid.cu (Sigmoid 激活函数)

Sigmoid (S 型函数) maps any real number into (0, 1) via the formula:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$  (Sigma of x equals one over, one plus e to the negative x.)

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void sigmoid(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

int main() {
    const int n = 5;
    size_t bytes = n * sizeof(float);
    float h_x[n] = {-2, -1, 0, 1, 2}, h_y[n];

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);

    sigmoid<<<(n + 255) / 256, 256>>>(d_x, d_y, n);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.3f ", h_y[i]);
    // Output: 0.119 0.269 0.500 0.731 0.881
    cudaFree(d_x); cudaFree(d_y);
}
```

### 5) elementwise_mul.cu (逐元素乘法)

Element-wise multiplication (逐元素乘法), also called the Hadamard product (哈达玛积), multiplies two arrays position by position.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void elementwise_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

int main() {
    const int n = 5;
    size_t bytes = n * sizeof(float);
    float h_a[n] = {1,2,3,4,5}, h_b[n] = {2,2,2,2,2}, h_c[n];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    elementwise_mul<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_c[i]);
    // Output: 2.0 4.0 6.0 8.0 10.0
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 3. Must-Know Questions

### 1) Why does the kernel need an `idx < n` check?

Because the launched thread count (线程数) is rounded up to a multiple of `blockDim.x`, extra threads (线程) must be masked out to avoid out-of-bounds memory access (越界访问).

### 2) Why are block sizes (线程块大小) commonly 128, 256, or 512?

Block sizes like 128, 256, or 512 are multiples of 32 (warp size), so they map efficiently to warps and are easy for the GPU to schedule. Larger blocks may use more registers and shared memory, which can reduce the number of blocks that fit on one SM. So 128, 256, and 512 are common choices because they usually provide a good balance between parallelism, scheduling efficiency, and resource usage.

### 3) Why is the GPU good at large-scale data parallelism (数据并行)?

Because the GPU has thousands of lightweight cores (轻量级核心) and uses SIMT (单指令多线程) to hide memory latency (内存延迟) by switching warps (线程束).

### 4) Why does kernel launch (内核启动) have overhead (开销)?

Because each launch involves driver API calls (驱动 API 调用), parameter passing (参数传递), and host-device synchronization (主机-设备同步), which costs several microseconds.

### 5) Why can't the first run be used as a benchmark (基准测试)?

Because the first run includes CUDA context initialization (上下文初始化), JIT compilation (即时编译), and cold cache (冷缓存) effects, which inflate the measured time.

<br>
