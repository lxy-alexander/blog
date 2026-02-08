---
title: "CUDA Programming Guide"
published: 2026-02-07
description: "CUDA Programming Guide"
image: ""
tags: ["cuda","CUDA Programming Guide"]
category: cuda
draft: false
lang: ""
---





# Introduction

## CPU & GPU

**CPU**：专为**低延迟**设计，擅长快速执行串行任务（流控制和缓存占用大量晶体管）。

**GPU**：专为**高吞吐量**设计，擅长并行执行数千个线程（将更多晶体管用于数据处理单元）。



## **领域特定语言 (DSLs)**

使用 Triton 或 Warp 等语言进行高层级编程。







# Programming Model

## 软硬件映射

**线程块 (Thread Block)** 映射到 **SM (流多处理器)**。所有线程块内的线程在同一个 SM 上执行，共享资源。

**线程块集群 (Cluster)**（新特性）映射到 **GPC (图形处理集群)**，允许跨块的硬件级同步和分布式共享内存访问。



##  Warp 执行模型 (SIMT)

**Warp**：硬件调度的基本单位，包含 32 个线程。

**SIMT 机制**：Warp 内的线程“锁步”执行同一条指令。

**分支发散 (Divergence)**：如果 Warp 内的线程进入不同的 `if-else` 分支，硬件会 **串行化** 执行各路径，未选中的线程会被“掩盖 (Masked)”。**重点：** 这种发散会降低性能，应尽量让同一 Warp 内的线程走相同的控制流。

![image-20260207114508984](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260207114508984)





## 内存层次结构

高效利用内存与计算能力同等重要。

-   **寄存器 (Registers)**：速度最快，线程私有。
-   **共享内存 (Shared Memory)**：位于 SM 片上，速度极快，可编程控制。用于 **线程块内的数据交换**，是优化性能的关键。
-   **全局内存 (Global Memory)**：容量大但延迟高（DRAM），所有线程可见。
-   **统一内存 (Unified Memory)**：允许 CPU 和 GPU 访问同一地址空间，系统自动处理数据迁移。





![image-20260207114732122](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260207114732122)





# CUDA 平台

## 1. 计算能力 (Compute Capability)

-   **定义**：标识 GPU 硬件特性的版本号，格式为 `X.Y`（主版本.次版本）。
-   **对应关系**：计算能力 12.0 的 GPU 对应 sm_120





## 2. 软件栈结构

-   **NVIDIA Driver**：GPU 的“操作系统”，是运行任何 GPU 程序（包括图形和计算）的基础。
-   **CUDA Toolkit**：开发者用于构建应用的库、头文件和工具集。
-   **CUDA Runtime**：提供高级 API（如内存分配、内核启动），构建在底层的 **Driver API** 之上。



## 3. 编译与执行模型

CUDA 代码的编译流程旨在平衡**性能**与**兼容性**：

-   **PTX (Parallel Thread Execution)**：
    -   一种虚拟的汇编语言（中间表示层）。
    -   **作用**：提供对硬件的抽象，允许代码在未来的硬件上运行（通过 JIT 编译）。
-   **Cubin (CUDA Binary)**：
    -   针对特定 GPU 架构编译的实际二进制机器码。
-   **Fatbin**：
    -   一个容器，可以包含针对不同架构的多个 Cubin 和 PTX 代码。运行时，驱动会自动选择最适合当前 GPU 的版本。







4.   ## 兼容性规则 

| **类型**                 | **规则**                                                     | **示例**                                                     |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **二进制兼容性 (Cubin)** | **相同主版本号向下兼容**。GPU 的次版本号必须 $\ge$二进制代码的次版本号。跨主版本号**不兼容**。 | `sm_86` 的二进制代码可以在计算能力 8.6 和 8.9 的 GPU 上运行，但**不能**在 8.0 或 9.0 的 GPU 上运行。 |
| **PTX 兼容性 (JIT)**     | **完全向前兼容**。旧版 PTX 可以在任何新版 GPU 上即时编译运行。 | `compute_80` 的 PTX 代码可以在计算能力 8.0、8.6、9.0 甚至未来的 GPU 上运行。 |

>   **开发建议**：==为了最大化兼容性，应用程序通常在 Fatbin 中包含特定架构的优化二进制文件（Cubin），同时包含一份 PTX 代码以支持未来的 GPU。==





# CUDA C++

## 1. 编译与内核定义

-   **编译**：使用 `nvcc` 编译器驱动程序。
-   **内核 (Kernel)**：在 GPU 上运行的函数，由 CPU 调用。必须使用 `__global__` 修饰符，返回类型为 `void`。

```
// 内核定义示例
__global__ void vecAdd(float* A, float* B, float* C) { }
```



## 2.启动内核 (Kernel Launch)

使用 **三重尖括号(Triple Angle Brackets) `<<< >>>`** 指定执行配置。启动是**异步(synchronized)**的（CPU 不等待 GPU 完成）。

-   **一维启动示例**（1 个块，256 个线程）：

    ```
    vecAdd<<<1, 256>>>(A, B, C);
    ```

-   **多维启动示例**（使用 `dim3`）：

    `MatAdd`使用 16x16 线程块网格启动内核的过程，其中每个线程块为 8x8。

    ```
    dim3 grid(16, 16); // 网格维度
    dim3 block(8, 8);  // 线程块维度
    MatAdd<<<grid, block>>>(A, B, C);
    ```



## 3. 线程索引与全局位置计算

线程通过内置变量（`threadIdx`, `blockIdx`, `blockDim`, `gridDim`）识别自身身份。最常用的一维全局索引计算方式如下：

-   `threadIdx`返回线程在其线程块内的索引。线程块中的每个线程都有不同的索引。==范围是：`0` 到 `blockDim.x - 1`。==
-   `blockDim`给出线程块的尺寸，该尺寸在内核启动的执行配置中指定。
-   `blockIdx`返回网格中线程块的索引。每个线程块都有不同的索引。==范围是：`0` 到 `gridDim.x - 1`==
-   `gridDim`给出网格的尺寸，该尺寸是在内核启动时的执行配置中指定的。

### $$\text{全局索引} = \text{当前块内的线程ID} + (\text{块的大小} \times \text{当前块ID})$$

```
__global__ void vecAdd(float* A, float* B, float* C) {
   // 计算当前线程负责的全局索引
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
   
   // 执行计算
   C[workIndex] = A[workIndex] + B[workIndex];
}
```

### Example `vecAdd<<<4, 256>>>`

-   **任务**：两个 1024 元素的向量相加。
-   **启动配置**：`<<<4, 256>>>`
    -   `gridDim.x` = 4（共 4 个块）
    -   `blockDim.x` = 256（每个块 256 个线程）
    -   总线程数 = $4 \times 256 = 1024$，正好对应数据量。

以下是这 4 个块中线程如何计算出覆盖 `0` 到 `1023` 的索引：

| **线程块 (Block)** | **blockIdx.x** | **blockDim.x** | **基址偏移 (Offset)** | **线程索引范围 (threadIdx.x)** | **计算出的全局索引 (workIndex)**     |
| ------------------ | -------------- | -------------- | --------------------- | ------------------------------ | ------------------------------------ |
| **第 1 个块**      | 0              | 256            | $256 \times 0 = 0$    | 0 ~ 255                        | **0 ~ 255**                          |
| **第 2 个块**      | 1              | 256            | $256 \times 1 = 256$  | 0 ~ 255                        | $0+256$ ~ $255+256$ = **256 ~ 511**  |
| **第 3 个块**      | 2              | 256            | $256 \times 2 = 512$  | 0 ~ 255                        | $0+512$ ~ $255+512$ = **512 ~ 767**  |
| **第 4 个块**      | 3              | 256            | $256 \times 3 = 768$  | 0 ~ 255                        | $0+768$ ~ $255+768$ = **768 ~ 1023** |









## 4. 处理任意数据长度 (边界检查)

当数据长度 `vectorLength` 不是块大小的整数倍时，必须在内核中添加边界检查，并在主机端向上取整计算块数。

### **内核代码 (添加 `if` 检查)**

```
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    // 关键：防止越界访问
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

### **主机端代码 (计算块数)**

#### 方法A

计算网格尺寸 (向上取整)

为了覆盖所有数据，需要启动足够的线程块。计算公式为：**(总数据量 + 块大小 - 1) / 块大小**。

```
// vectorLength 是向量的总元素个数
int threads = 256; // 典型的块大小

// 向上取整公式：(N + M - 1) / M
int blocks = (vectorLength + threads - 1) / threads;

vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

#### 方法B

CUDA[核心计算库 (CCCL)](https://nvidia.github.io/cccl/)提供了一个便捷的实用程序，`cuda::ceil_div`用于执行向上取整以计算内核启动所需的块数。该实用程序可通过包含头文件来使用`<cuda/cmath>`。

```
#include <cuda/cmath>

int threads = 256;
// 使用辅助函数进行向上取整
int blocks = cuda::ceil_div(vectorLength, threads);

vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```











