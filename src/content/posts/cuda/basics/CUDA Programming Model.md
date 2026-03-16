---
title: "CUDA Programming Model"
published: 2026-03-14
description: "CUDA Programming Model"
image: ""
tags: ["cuda","basics","CUDA Programming Model"]
category: cuda / basics
draft: false
lang: ""
---

# I. CUDA Programming Model

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> CUDA (<span style="color:#E8600A;font-weight:700">Compute Unified Device Architecture</span> 统一计算设备架构) is a <span style="color:#2980B9;font-weight:700">parallel programming model</span> for NVIDIA GPUs. It enables <span style="color:#2980B9;font-weight:700">massive parallelism</span> by executing thousands of threads simultaneously on the <span style="color:#2980B9;font-weight:700">device (GPU)</span> while controlled by the <span style="color:#2980B9;font-weight:700">host (CPU)</span>. </div>

## 1. Kernel Function Specifiers

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Function specifiers</span> (函数限定符) determine <span style="color:#2980B9;font-weight:700">where a function executes</span> and <span style="color:#2980B9;font-weight:700">where it can be called from</span>, defining the execution flow between host and device. </div>

### 1) __global__

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The <span style="color:#E8600A;font-weight:700">__global__</span> qualifier defines a <span style="color:#E8600A;font-weight:700">kernel function</span> (内核函数) that <span style="color:#2980B9;font-weight:700">runs on the GPU</span> but is <span style="color:#2980B9;font-weight:700">called from the CPU</span>. It must return `void`. <span style="color:#2980B9;font-weight:700">Use when</span> defining parallel operations executed on the GPU. </div>

```cpp
// Define a kernel that adds two vectors
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Launch from host code
int main() {
    // ... memory allocation ...
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Kernel launch
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">__global__ functions cannot be called from other device functions</span>. They must be launched from host using `<<<...>>>` syntax. <span style="color:#C0392B;font-weight:700">Return type must be void</span>. </div>

### 2) __device__

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The <span style="color:#E8600A;font-weight:700">__device__</span> qualifier marks a function that <span style="color:#2980B9;font-weight:700">runs on the GPU</span> and can <span style="color:#2980B9;font-weight:700">only be called from other GPU functions</span>. <span style="color:#2980B9;font-weight:700">Use when</span> creating helper functions for kernels to organize parallel code. </div>

```cpp
// Device function for element-wise multiplication
__device__ float multiply(float a, float b) {
    return a * b;
}

// Kernel calling the device function
__global__ void elementWiseMul(float* A, float* B, float* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = multiply(A[i], B[i]);  // Calling __device__ function
    }
}
```

### 3) __host__

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The <span style="color:#E8600A;font-weight:700">__host__</span> qualifier (optional) indicates a function <span style="color:#2980B9;font-weight:700">runs on the CPU</span> and is called from the CPU. This is the <span style="color:#2980B9;font-weight:700">default for all functions</span>. <span style="color:#2980B9;font-weight:700">Use when</span> creating functions that exist on both CPU and GPU (combined with __device__). </div>

```cpp
// CPU-only function
__host__ void cpuFunction() {
    printf("Running on CPU\n");
}

// Function callable from both CPU and GPU
__host__ __device__ float square(float x) {
    return x * x;
}

// Usage in both contexts
__global__ void kernel(float* data) {
    int i = threadIdx.x;
    data[i] = square(data[i]);  // GPU version called
}

void hostFunction(float* data) {
    for(int i = 0; i < 10; i++) {
        data[i] = square(data[i]);  // CPU version called
    }
}
```

## 2. Launch Configuration

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The <span style="color:#E8600A;font-weight:700">launch configuration</span> (启动配置) syntax `<<<grid, block>>>` specifies <span style="color:#2980B9;font-weight:700">how many threads to create</span> and <span style="color:#2980B9;font-weight:700">how to organize them</span> in the parallel execution hierarchy. </div>

### 1) Grid and Block Dimensions

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The configuration `<<<grid, block>>>` defines a <span style="color:#E8600A;font-weight:700">grid</span> (网格) of <span style="color:#E8600A;font-weight:700">blocks</span> (线程块). Both can be 1D, 2D, or 3D using `dim3` type. <span style="color:#2980B9;font-weight:700">Use when</span> mapping threads to data structures like matrices or volumes. </div>

```cpp
// 1D configuration: 10 blocks, 256 threads per block
kernel<<<10, 256>>>();

// 2D configuration: 4x4 grid (16 blocks), 16x16 threads per block
dim3 gridDim(4, 4);
dim3 blockDim(16, 16);
kernel<<<gridDim, blockDim>>>();

// 3D configuration for volumetric data
dim3 grid3D(2, 2, 2);     // 8 blocks total
dim3 block3D(8, 8, 8);    // 512 threads per block
kernel<<<grid3D, block3D>>>();
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Maximum threads per block is 1024</span> for modern GPUs. Choose block sizes that are multiples of 32 (warp size) for optimal performance. </div>

## 3. Built-in Variables

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> CUDA provides <span style="color:#E8600A;font-weight:700">built-in variables</span> (内置变量) allowing each thread to <span style="color:#2980B9;font-weight:700">identify its position</span> within the grid and blocks for data mapping. </div>

### 1) threadIdx and blockIdx

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">threadIdx</span> (线程索引) gives the thread's index within its block. <span style="color:#E8600A;font-weight:700">blockIdx</span> (块索引) gives the block's index within the grid. Both are `uint3` types. <span style="color:#2980B9;font-weight:700">Use when</span> computing unique global indices for each thread. </div>

```cpp
__global__ void computeIndices() {
    // 1D indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 2D indices
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    
    // Calculate global 1D index
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Calculate global 2D indices
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
}
```

### 2) blockDim and gridDim

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">blockDim</span> (块维度) stores the dimensions of each block. <span style="color:#E8600A;font-weight:700">gridDim</span> (网格维度) stores the dimensions of the grid. <span style="color:#2980B9;font-weight:700">Use when</span> calculating bounds or mapping to multi-dimensional data. </div>

```cpp
__global__ void process2DMatrix(float* matrix, int width, int height) {
    // Calculate 2D global indices using blockDim and gridDim
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    // Check bounds
    if (col < width && row < height) {
        int idx = row * width + col;
        matrix[idx] *= 2.0f;
    }
}

// Launch for 1024x1024 matrix
dim3 blockSize(16, 16);  // 256 threads/block
dim3 gridSize(64, 64);    // 4096 blocks total
process2DMatrix<<<gridSize, blockSize>>>(d_matrix, 1024, 1024);
```

## 4. Synchronization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Synchronization</span> (同步) ensures threads in a block <span style="color:#2980B9;font-weight:700">reach the same execution point</span> before proceeding, preventing race conditions in shared memory. </div>

### 1) __syncthreads()

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">__syncthreads()</span> (线程块同步) acts as a <span style="color:#2980B9;font-weight:700">barrier</span> within a block. All threads must reach this point before any can continue. <span style="color:#2980B9;font-weight:700">Use when</span> threads exchange data through shared memory. </div>

```cpp
__global__ void parallelSum(float* input, float* output, int N) {
    __shared__ float sharedData[256];  // Shared memory for block
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    sharedData[threadIdx.x] = (i < N) ? input[i] : 0.0f;
    
    // Ensure all threads have loaded data
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        }
        // Synchronize after each step
        __syncthreads();
    }
    
    // Write block result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">__syncthreads() can cause deadlock in conditional code</span> where some threads don't reach it. All threads in a block must execute the same __syncthreads(). <span style="color:#C0392B;font-weight:700">Only synchronizes within a block, not across blocks</span>. </div>

## Built-in Variables Comparison

<div style="margin:20px 0">

| Variable                                     | <span style="color:#2980B9">Description</span>           | <span style="color:#E8600A">Type</span> | <span style="color:#2980B9">Range</span> | <span style="color:#E8600A">Use Case</span> |
| -------------------------------------------- | -------------------------------------------------------- | --------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| <span style="color:#2980B9">threadIdx</span> | <span style="color:#E8600A">Thread index in block</span> | uint3                                   | 0 to blockDim-1                          | Per-thread identification                   |
| <span style="color:#E8600A">blockIdx</span>  | <span style="color:#2980B9">Block index in grid</span>   | uint3                                   | 0 to gridDim-1                           | Block-level identification                  |
| <span style="color:#2980B9">blockDim</span>  | <span style="color:#E8600A">Dimensions of block</span>   | dim3                                    | Set by launch                            | Loop bounds calculation                     |
| <span style="color:#E8600A">gridDim</span>   | <span style="color:#2980B9">Dimensions of grid</span>    | dim3                                    | Set by launch                            | Grid size calculation                       |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">CUDA programming model</span> uses <span style="color:#2980B9;font-weight:700">function specifiers</span> for execution control, <span style="color:#2980B9;font-weight:700">launch configuration</span> for thread organization, <span style="color:#2980B9;font-weight:700">built-in variables</span> for thread identification, and <span style="color:#2980B9;font-weight:700">__syncthreads()</span> for block-level synchronization. </div>
