---
title: "CUDA Fundamentals"
published: 2026-04-27
description: "CUDA Fundamentals"
image: ""
tags: ["cuda","CUDA Fundamentals"]
category: cuda
draft: false
lang: ""
---

# CUDA Fundamentals 

CUDA (统一计算设备架构) is a parallel computing platform that lets the CPU (中央处理器) control the GPU (图形处理器) for high-throughput computation.


## 1. Heterogeneous Computing Model

Heterogeneous Computing (异构计算) means using different processors, usually CPU (中央处理器) and GPU (图形处理器), together for different types of work.

### 1）Host vs Device

The Host (主机端) is the CPU (中央处理器) side that controls execution, while the Device (设备端) is the GPU (图形处理器) side that runs parallel kernels.

```cpp
#include <iostream>

int main() {
    std::cout << "Host code runs on CPU." << std::endl;
    return 0;
}

// Output:
// Host code runs on CPU.
```

### 2）Function Qualifiers

CUDA Function Qualifiers (函数限定符) define where a function runs and where it can be called.

| Qualifier (限定符) | Runs On (运行位置) | Called From (调用位置) | Meaning (含义)      |
| ------------------ | ------------------ | ---------------------- | ------------------- |
| `__host__`         | Host (主机端)      | Host (主机端)          | CPU function        |
| `__device__`       | Device (设备端)    | Device (设备端)        | GPU helper function |
| `__global__`       | Device (设备端)    | Host (主机端)          | GPU kernel function |

```cpp
#include <iostream>

__host__ void hostFunction() {
    std::cout << "This is a host function." << std::endl;
}

__global__ void kernelFunction() {
    printf("This is a global kernel running on GPU.\n");
}

int main() {
    hostFunction();

    kernelFunction<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// This is a host function.
// This is a global kernel running on GPU.
```

### 3）Data Flow Between Host and Device

Data Transfer (数据传输) is required because Host Memory (主机内存) and Device Memory (设备内存) are physically separate in most CUDA programs.

```cpp
#include <iostream>

int main() {
    int h_value = 10;
    int h_result = 0;
    int* d_value = nullptr;

    cudaMalloc((void**)&d_value, sizeof(int));
    cudaMemcpy(d_value, &h_value, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&h_result, d_value, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Copied value: " << h_result << std::endl;

    cudaFree(d_value);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// Copied value: 10
```

### 4）Why GPU Is Needed

Parallelism (并行性) makes GPU (图形处理器) effective because many simple operations can run at the same time instead of one by one.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> data = {1, 2, 3, 4};

    for (int i = 0; i < data.size(); i++) {
        data[i] *= 2;
    }

    for (int x : data) {
        std::cout << x << " ";
    }

    return 0;
}

// Output:
// 2 4 6 8
```

Serial Execution (串行执行) handles tasks one after another, while Parallel Execution (并行执行) handles many tasks together.

## 2. GPU Hardware Architecture

GPU Hardware Architecture (GPU硬件架构) is designed around many lightweight cores that execute massive numbers of threads.

### 1）SM

An SM (流式多处理器) is the main execution unit inside a GPU (图形处理器), responsible for scheduling and running groups of threads.

### 2）CUDA Core and Tensor Core

CUDA Core (CUDA核心) handles general numeric computation, while Tensor Core (张量核心) accelerates matrix operations used in Deep Learning (深度学习).

### 3）Warp

A Warp (线程束) is a group of 32 Threads (线程) scheduled together by an SM (流式多处理器).

```cuda
#include <iostream>

__global__ void printThreadInfo() {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / 32;

    printf("Thread %d belongs to warp %d\n", globalThreadId, warpId);
}

int main() {
    printThreadInfo<<<1, 8>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Possible Output:
// Thread 0 belongs to warp 0
// Thread 1 belongs to warp 0
// Thread 2 belongs to warp 0
// Thread 3 belongs to warp 0
// Thread 4 belongs to warp 0
// Thread 5 belongs to warp 0
// Thread 6 belongs to warp 0
// Thread 7 belongs to warp 0
```

### 4）SIMT Execution Model

SIMT (单指令多线程) means many Threads (线程) execute the same instruction on different data.

```cpp
#include <iostream>

__global__ void simtExample(int* data) {
    int i = threadIdx.x;
    data[i] = data[i] * 2;
}

int main() {
    int h_data[4] = {1, 2, 3, 4};
    int* d_data = nullptr;

    cudaMalloc((void**)&d_data, 4 * sizeof(int));
    cudaMemcpy(d_data, h_data, 4 * sizeof(int), cudaMemcpyHostToDevice);

    simtExample<<<1, 4>>>(d_data);

    cudaMemcpy(h_data, d_data, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_data[i] << " ";
    }

    cudaFree(d_data);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 2 4 6 8
```

## 3. CUDA Programming Model

The CUDA Programming Model (CUDA编程模型) organizes parallel work into Grid (网格), Block (线程块), and Thread (线程).

### 1）Grid, Block, and Thread

A Grid (网格) contains Blocks (线程块), and each Block (线程块) contains Threads (线程).

```cpp
#include <iostream>

__global__ void printHierarchy() {
    printf("blockIdx.x=%d, threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printHierarchy<<<2, 3>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Possible Output:
// blockIdx.x=0, threadIdx.x=0
// blockIdx.x=0, threadIdx.x=1
// blockIdx.x=0, threadIdx.x=2
// blockIdx.x=1, threadIdx.x=0
// blockIdx.x=1, threadIdx.x=1
// blockIdx.x=1, threadIdx.x=2
```

### 2）Built-in Variables

Built-in Variables (内置变量) such as `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` identify each Thread (线程) in the parallel layout.

| Variable (变量) | Meaning (含义)                                  |
| --------------- | ----------------------------------------------- |
| `threadIdx`     | Thread index inside a Block (线程块)            |
| `blockIdx`      | Block index inside a Grid (网格)                |
| `blockDim`      | Number of Threads (线程) in each Block (线程块) |
| `gridDim`       | Number of Blocks (线程块) in the Grid (网格)    |

```cpp
#include <iostream>

__global__ void printIndex() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("globalId=%d\n", globalId);
}

int main() {
    printIndex<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Possible Output:
// globalId=0
// globalId=1
// globalId=2
// globalId=3
// globalId=4
// globalId=5
// globalId=6
// globalId=7
```

### 3）1D, 2D, and 3D Thread Organization

Thread Organization (线程组织) can be one-dimensional, two-dimensional, or three-dimensional to match data shapes like arrays, matrices, and volumes.

```cpp
#include <iostream>

__global__ void print2DIndex() {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    printf("row=%d, col=%d\n", row, col);
}

int main() {
    dim3 block(2, 2);
    dim3 grid(2, 2);

    print2DIndex<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Possible Output:
// row=0, col=0
// row=0, col=1
// row=1, col=0
// row=1, col=1
// row=0, col=2
// row=0, col=3
// ...
```

## 4. CUDA Memory Hierarchy

CUDA Memory Hierarchy (CUDA内存层级) provides different memory spaces with different speed, size, and visibility.

### 1）Memory Types

Global Memory (全局内存) is large but slow, while Register Memory (寄存器内存) and Shared Memory (共享内存) are faster but limited.

| Memory (内存)       | Speed (速度)     | Size (大小) | Scope (作用域)     | Common Use (常见用途)         |
| ------------------- | ---------------- | ----------- | ------------------ | ----------------------------- |
| Register (寄存器)   | Fastest          | Very small  | One Thread (线程)  | Local scalar variables        |
| Shared (共享内存)   | Very fast        | Small       | One Block (线程块) | Data reused by Threads (线程) |
| Global (全局内存)   | Slow             | Large       | Whole Grid (网格)  | Main Device Memory (设备内存) |
| Local (局部内存)    | Slow             | Limited     | One Thread (线程)  | Spilled variables             |
| Constant (常量内存) | Fast when cached | Small       | Whole Grid (网格)  | Read-only constants           |
| Texture (纹理内存)  | Cached           | Medium      | Whole Grid (网格)  | Read-only spatial data        |

#### Register Memory Example

Register Memory (寄存器内存) stores private scalar variables for each Thread (线程).

```c p p
#include <iostream>

__global__ void registerExample(int* output) {
    int tid = threadIdx.x;
    int value = tid * 2;  // register variable
    output[tid] = value;
}

int main() {
    int h_output[4] = {0};
    int* d_output = nullptr;

    cudaMalloc((void**)&d_output, 4 * sizeof(int));

    registerExample<<<1, 4>>>(d_output);

    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_output);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 0 2 4 6
```



#### Shared Memory Example

Shared Memory (共享内存) allows Threads (线程) in the same Block (线程块) to reuse data quickly.

```cpp
#include <iostream>

__global__ void sharedExample(int* output) {
    __shared__ int sharedData[4];

    int tid = threadIdx.x;
    sharedData[tid] = tid + 1;

    __syncthreads();

    output[tid] = sharedData[tid] * 10;
}

int main() {
    int h_output[4] = {0};
    int* d_output = nullptr;

    cudaMalloc((void**)&d_output, 4 * sizeof(int));

    sharedExample<<<1, 4>>>(d_output);

    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_output);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 10 20 30 40
```

#### Global Memory Example

Global Memory (全局内存) stores data that can be accessed by all Threads (线程) in the Grid (网格).

```cpp
#include <iostream>

__global__ void globalExample(int* data) {
    int tid = threadIdx.x;
    data[tid] = data[tid] + 100;
}

int main() {
    int h_data[4] = {1, 2, 3, 4};
    int* d_data = nullptr;

    cudaMalloc((void**)&d_data, 4 * sizeof(int));
    cudaMemcpy(d_data, h_data, 4 * sizeof(int), cudaMemcpyHostToDevice);

    globalExample<<<1, 4>>>(d_data);

    cudaMemcpy(h_data, d_data, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_data[i] << " ";
    }

    cudaFree(d_data);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 101 102 103 104
```

#### Local Memory Example

Local Memory (局部内存) is private to one Thread (线程) and may be used when registers are not enough.

```cpp
#include <iostream>

__global__ void localExample(int* output) {
    int tid = threadIdx.x;

    int localArray[2];  // may be placed in local memory
    localArray[0] = tid;
    localArray[1] = tid + 10;

    output[tid] = localArray[0] + localArray[1];
}

int main() {
    int h_output[4] = {0};
    int* d_output = nullptr;

    cudaMalloc((void**)&d_output, 4 * sizeof(int));

    localExample<<<1, 4>>>(d_output);

    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_output);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 10 12 14 16
```

#### Constant Memory Example

Constant Memory (常量内存) stores read-only values shared by all Threads (线程).

```cpp
#include <iostream>

__constant__ int constantValue;

__global__ void constantExample(int* output) {
    int tid = threadIdx.x;
    output[tid] = tid + constantValue;
}

int main() {
    int h_output[4] = {0};
    int* d_output = nullptr;
    int h_constant = 1000;

    cudaMalloc((void**)&d_output, 4 * sizeof(int));

    cudaMemcpyToSymbol(constantValue, &h_constant, sizeof(int));

    constantExample<<<1, 4>>>(d_output);

    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaFree(d_output);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 1000 1001 1002 1003
```

#### Texture Memory Example

Texture Memory (纹理内存) is read-only cached memory optimized for spatial locality.

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void textureExample(cudaTextureObject_t texObj, int* output) {
    int tid = threadIdx.x;
    output[tid] = tex1Dfetch<int>(texObj, tid);
}

int main() {
    int h_data[4] = {5, 10, 15, 20};
    int h_output[4] = {0};

    int* d_data = nullptr;
    int* d_output = nullptr;

    cudaMalloc((void**)&d_data, 4 * sizeof(int));
    cudaMalloc((void**)&d_output, 4 * sizeof(int));
    cudaMemcpy(d_data, h_data, 4 * sizeof(int), cudaMemcpyHostToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
    resDesc.res.linear.sizeInBytes = 4 * sizeof(int);

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    textureExample<<<1, 4>>>(texObj, d_output);

    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
        std::cout << h_output[i] << " ";
    }

    cudaDestroyTextureObject(texObj);
    cudaFree(d_data);
    cudaFree(d_output);

    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// 5 10 15 20
```



### 2）Host Memory vs Device Memory

Host Memory (主机内存) belongs to the CPU (中央处理器), while Device Memory (设备内存) belongs to the GPU (图形处理器).

```cuda
#include <iostream>

int main() {
    int hostValue = 42;
    int* deviceValue = nullptr;

    cudaMalloc((void**)&deviceValue, sizeof(int));
    cudaMemcpy(deviceValue, &hostValue, sizeof(int), cudaMemcpyHostToDevice);

    int copiedBack = 0;
    cudaMemcpy(&copiedBack, deviceValue, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Value from device memory: " << copiedBack << std::endl;

    cudaFree(deviceValue);
    return 0;
}

// Compile:
// nvcc main.cu -o main
//
// Output:
// Value from device memory: 42
```

## 5. First CUDA Program

A CUDA Program (CUDA程序) usually allocates Device Memory (设备内存), copies data, launches a Kernel (核函数), copies results back, and frees memory.

### 1）Hello World

A Kernel (核函数) is launched from the Host (主机端) and executed on the Device (设备端).

```cpp
#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    std::cout << "Hello from CPU!" << std::endl;

    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Compile:
// nvcc hello.cu -o hello
//
// Output:
// Hello from CPU!
// Hello from GPU!
```

### 2）Vector Add

Vector Addition (向量加法) is the classic CUDA (统一计算设备架构) example because each output element can be computed independently.

```cpp
#include <iostream>

__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 5;
    const int size = n * sizeof(int);

    int h_a[n] = {1, 2, 3, 4, 5};
    int h_b[n] = {10, 20, 30, 40, 50};
    int h_c[n] = {0};

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << h_c[i] << " ";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// Compile:
// nvcc vector_add.cu -o vector_add
//
// Output:
// 11 22 33 44 55
```

### 3）Compile and Run with nvcc

`nvcc` (NVIDIA CUDA编译器) compiles CUDA Source Files (CUDA源文件) into executable programs.

```bash
nvcc vector_add.cu -o vector_add
./vector_add

# Output:
# 11 22 33 44 55
```

### 4）Basic CUDA Program Flow

The CUDA Program Flow (CUDA程序流程) is allocate memory, copy input, launch kernel, copy output, and release memory.

```cpp
Host data
  -> cudaMalloc
  -> cudaMemcpy HostToDevice
  -> kernel<<<grid, block>>>()
  -> cudaMemcpy DeviceToHost
  -> cudaFree
```

$$
\text{blocksPerGrid} = \left\lceil \frac{n}{\text{threadsPerBlock}} \right\rceil
$$

This formula ensures enough Blocks (线程块) are launched to cover all elements.

## 6. Interview Summary

CUDA (统一计算设备架构) is designed for Data Parallelism (数据并行), where many Threads (线程) perform the same operation on different data.

The Host (主机端) manages memory and launches Kernels (核函数), while the Device (设备端) performs massively parallel computation.

The Grid-Block-Thread (网格-线程块-线程) hierarchy maps logical parallel work onto GPU Hardware (GPU硬件).

The Warp (线程束) is the actual scheduling unit, and SIMT (单指令多线程) explains how GPU (图形处理器) executes many Threads (线程) together.

The Memory Hierarchy (内存层级) is critical because choosing the right memory space directly affects CUDA Performance (CUDA性能).
