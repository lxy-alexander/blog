---
title: "CUDA Core APIs"
published: 2026-04-27
description: "CUDA Core APIs"
image: ""
tags: ["cuda","CUDA Core APIs"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-27T14:37:51.469.801451207Z"
---

# CUDA Core APIs

Concise interview-style summary of CUDA Core APIs (核心 API).

## 1. Execution Model (执行模型)

The Execution Model (执行模型) defines how threads (线程) are organized into grid (网格), block (线程块), and warp (线程束) for parallel execution.

### 1) Key Concepts

-   **grid (网格)**: A grid (网格) is the top-level execution structure (顶层执行结构) that organizes multiple blocks (线程块) to fully utilize GPU parallelism (GPU 并行能力).
-   **block (线程块)**: A block (线程块) is a group of threads (线程集合) that can share data through shared memory (共享内存) and synchronize efficiently within the block (块内同步).
-   **warp (线程束)**: A warp (线程束) is the smallest execution unit (最小执行单位) of 32 threads that run in lockstep (同步执行) under SIMD (单指令多数据).

### 2) Example

```cpp
#include <stdio.h>

__global__ void helloKernel() {
    printf("Thread (%d, %d)\n", threadIdx.x, blockIdx.x);
}

int main() {
    helloKernel<<<2, 4>>>(); // 2 blocks, each with 4 threads
    cudaDeviceSynchronize();
    return 0;
}

/*
Output (order not guaranteed):
Thread (0, 0)
Thread (1, 0)
Thread (2, 0)
Thread (3, 0)
Thread (0, 1)
Thread (1, 1)
Thread (2, 1)
Thread (3, 1)
*/
```

<br>

## 2. Kernel Launch & Configuration (启动与配置)

Kernel Launch (核函数启动) uses <<<grid, block>>> to define parallelism and resource usage.

### 1) Formula

$$
Total\ Threads = gridDim \times blockDim
$$

### 2) Example

```cpp
#include <stdio.h>

__global__ void indexKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global index: %d\n", idx);
}

int main() {
    indexKernel<<<2, 4>>>(); // total 8 threads
    cudaDeviceSynchronize();
    return 0;
}

/*
Output (order not guaranteed):
Global index: 0
Global index: 1
...
Global index: 7
*/
```

<br>

## 3. Memory Management (内存管理)

Memory Management (内存管理) handles allocation (分配), transfer (传输), and access between Host (主机) and Device (设备).

### 1) Types

-   Global Memory (全局内存)
-   Shared Memory (共享内存)
-   Unified Memory (统一内存)

### 2) Example

```cpp
#include <stdio.h>

__global__ void add(int *a) {
    a[0] += 10;
}

int main() {
    int *d_a;
    cudaMalloc(&d_a, sizeof(int));

    int h_a = 5;
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_a);

    cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    printf("Result: %d\n", h_a);
    return 0;
}

/*
Output:
Result: 15
*/
```

<br>

## 4. Streams & Events (流与事件)

Streams (流) enable asynchronous execution (异步执行), while Events (事件) are used for timing and synchronization.

### 1) Example

```cpp
#include <stdio.h>

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("Stream created\n");

    cudaStreamDestroy(stream);
    return 0;
}

/*
Output:
Stream created
*/
```

<br>

## 5. Synchronization (同步机制)

Synchronization (同步机制) ensures correct execution order and prevents race conditions (竞态条件).

### 1) Example

```cpp
#include <stdio.h>

__global__ void syncKernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = idx;

    __syncthreads(); // block-level sync (块级同步)

    if (idx == 0) {
        printf("All threads updated\n");
    }
}

int main() {
    int *d;
    cudaMalloc(&d, 4 * sizeof(int));

    syncKernel<<<1,4>>>(d);
    cudaDeviceSynchronize();

    cudaFree(d);
    return 0;
}

/*
Output:
All threads updated
*/
```

<br>

## 6. Device Management (设备管理)

Device Management (设备管理) controls GPU selection (设备选择) and queries device properties (设备属性).

### 1) Example

```cpp
#include <stdio.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);

    printf("Device count: %d\n", count);

    cudaSetDevice(0);
    return 0;
}

/*
Output:
Device count: 1  // depends on system
*/
```

<br>

## 7. Error Handling (错误处理)

Error Handling (错误处理) detects runtime failures using cudaError_t (错误类型) and related APIs.

### 1) Example

```cpp
#include <stdio.h>

int main() {
    cudaError_t err = cudaMalloc(NULL, 100);

    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    return 0;
}

/*
Output:
Error: invalid argument
*/
```

<br>

## 8. Performance Analysis & Optimization (性能分析与优化)

Performance Optimization (性能优化) improves efficiency using occupancy (占用率), memory coalescing (内存合并), and timing tools.

### 1) Example (Event Timing)

```cpp
#include <stdio.h>

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // dummy workload
    for (int i = 0; i < 1000000; i++);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Time: %f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

/*
Output:
Time: ~0.x ms
*/
```
