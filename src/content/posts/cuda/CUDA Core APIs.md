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

-   **grid (网格)**: A grid (网格) is the top-level execution structure (顶层执行结构) that organizes a collection of blocks (线程块) to fully utilize GPU parallelism (GPU 并行能力).
-   **block (线程块)**: A block (线程块) is a group of threads (线程集合) that can share data through shared memory (共享内存) and synchronize efficiently within the block (块内同步).
-   **warp (线程束)**: A warp (线程束) is the smallest hardware execution unit (最小执行单位) . it has a group of 32 threads scheduled together by an SM.

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

Kernel Launch (核函数启动) uses <<<grid, block>>> (triple angle brackets, launch with grid and block dimensions )to define parallelism and resource usage.

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

-   **Global Memory**: large, high-latency device memory accessible by all threads in the grid; used as the main data storage on the GPU.
-   **Shared Memory**: small, low-latency on-chip memory shared by threads within the same block; used for fast data reuse and cooperation.
-   **Unified Memory**: memory model that provides a unified address space shared by CPU and GPU; the system automatically manages data movement between host and device.

### 2) Example

```cpp
#include <stdio.h>

__global__ void add(int *a) {
    __shared__ int sharedValue;   // Shared Memory: visible within one block

    if (threadIdx.x == 0) {
        sharedValue = a[0];       // Read from Global/Unified Memory into Shared Memory
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        sharedValue += 10;
        a[0] = sharedValue;       // Write back to Global/Unified Memory
    }
}

int main() {
    int *a;

    // Unified Memory: accessible by both CPU and GPU
    cudaMallocManaged(&a, sizeof(int));

    a[0] = 5;   // CPU writes directly

    add<<<1, 4>>>(a);

    // Wait for GPU to finish before CPU reads
    cudaDeviceSynchronize();

    printf("Result: %d\n", a[0]);

    cudaFree(a);
    return 0;
}

/*
Compile:
nvcc main.cu -o main

Output:
Result: 15
*/
```

<br>

## 4. Streams & Events (流与事件)

Streams (流) enable asynchronous execution (异步执行), allowing data transfer and kernel execution to overlap instead of running sequentially.

### 1) Example

```cpp
#include <stdio.h>

__global__ void add(int *a) {
    a[0] += 10;
}

int main() {
    int h_a = 5;
    int *d_a;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_a, sizeof(int));

    // Async copy H -> D (in stream)
    cudaMemcpyAsync(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice, stream);

    // Launch kernel in the same stream
    add<<<1, 1, 0, stream>>>(d_a);

    // Async copy D -> H
    cudaMemcpyAsync(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Wait for stream to finish
    cudaStreamSynchronize(stream);

    printf("Result: %d\n", h_a);

    cudaFree(d_a);
    cudaStreamDestroy(stream);
    return 0;
}

/*
Compile:
nvcc main.cu -o main

Output:
Result: 15
*/
```

<br>

## 5. Synchronization (同步机制)

Synchronization (同步机制) ensures that threads or operations execute in the correct order and prevents race conditions.

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

    printf("Device count: %d\n\n", count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d:\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("\n");
    }

    // Select device 0 (if exists)
    if (count > 0) {
        cudaSetDevice(0);
        printf("Using device 0\n");
    }

    return 0;
}

/*
Example Output:
Device count: 1

Device 0:
  Name: NVIDIA RTX XXXX
  Compute Capability: 8.6
  Global Memory: 8.00 GB
  Multiprocessors: 48
  Max Threads per Block: 1024

Using device 0
*/
```

<br>

## 7. Error Handling (错误处理)

Error Handling (错误处理) detects runtime failures using cudaError_t (错误类型) and related APIs.

### 1) Example

```cpp
#include <stdio.h>

#define CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

int main() {
    int *d_ptr;

    CHECK(cudaMalloc((void**)&d_ptr, 100));

    // error
    CHECK(cudaMalloc(NULL, 100));

    cudaFree(d_ptr);
    return 0;
}
```

<br>

## 8. Performance Analysis & Optimization (性能分析与优化)

Performance Optimization (性能优化) improves efficiency using occupancy (占用率), memory coalescing (内存合并), and timing tools, mesuring **floating-point compute capability**.

### 1) Example (Event Timing)

```cpp
#include <stdio.h>
#include <stdlib.h>

// 通用计时 + TFLOPS
template <typename Func>
double measureTFLOPS(Func func, double flops) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    func();   // 执行 workload（lambda）

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double seconds = ms / 1000.0;
    return flops / seconds / 1e12;   // TFLOPS
}

int main() {
    const int M = 1024;
    const int K = 2048;
    const int N = 2048;

    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // FLOPs = 2 * M * N * K
    double flops = 2.0 * M * N * K;

    // 用 lambda 定义 workload
    auto tflops = measureTFLOPS([&]() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }, flops);

    printf("TFLOPS: %e\n", tflops);

    free(A);
    free(B);
    free(C);

    return 0;
}
```
