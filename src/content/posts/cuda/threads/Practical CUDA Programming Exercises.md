---
title: "Practical CUDA Programming Exercises"
published: 2026-03-15
description: "Practical CUDA Programming Exercises"
image: ""
tags: ["cuda","threads","Practical CUDA Programming Exercises"]
category: cuda / threads
draft: false
lang: ""
---

# I. Practical CUDA Programming Exercises

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Practical CUDA programming</span> involves writing kernel functions that execute on GPU threads. Starting with simple examples like <span style="color:#2980B9;font-weight:700">vector addition and matrix multiplication</span> builds understanding of thread hierarchy and memory access patterns. These exercises demonstrate how to <span style="color:#2980B9;font-weight:700">leverage thousands of threads</span> for parallel data processing. </div>

## 1. Simple Kernel Function Structure

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> A <span style="color:#E8600A;font-weight:700">CUDA kernel (CUDA内核)</span> is a function executed on the GPU. It uses <span style="color:#2980B9;font-weight:700">__global__ specifier</span> and must have <span style="color:#C0392B;font-weight:700">void return type</span>. Each thread computes its data index using built-in variables. </div>

### 1) Basic Kernel Template

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Kernel template</span> shows the standard pattern: compute global index, check boundaries, perform work. <span style="color:#2980B9;font-weight:700">Use this template</span> for any 1D data processing kernel. </div>

```cpp
// Basic kernel template for 1D data processing
__global__ void kernelTemplate(float* input, float* output, int n) {
    // Calculate global thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Boundary check - prevent out-of-bounds access
    if (idx < n) {
        // Perform work on data element
        output[idx] = process(input[idx]);
    }
}

// Host code to launch kernel
int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy data to device (assuming h_input is initialized)
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    kernelTemplate<<<gridSize, blockSize>>>(d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for completion and copy result back
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
```

## 2. Vector Addition

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Vector addition (向量加法)</span> is the "Hello World" of CUDA programming. Each thread computes one element of the result vector, demonstrating <span style="color:#2980B9;font-weight:700">data parallelism</span> and <span style="color:#2980B9;font-weight:700">coalesced memory access</span>. </div>

### 1) Basic Vector Addition

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Vector addition kernel</span> adds two input vectors element-wise. <span style="color:#2980B9;font-weight:700">Use this pattern</span> for any element-wise operation on arrays. </div>

```cpp
// Vector addition kernel: C = A + B
__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        C[idx] = A[idx] + B[idx];  // Element-wise addition
    }
}

// Complete example with main function
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int n = 1 << 20;  // 1,048,576 elements
    size_t bytes = n * sizeof(float);
    
    // Host arrays
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Launch configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

### 2) Vector Addition with Strided Access

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Grid-stride loop (网格步长循环)</span> allows each thread to process multiple elements. <span style="color:#2980B9;font-weight:700">Use this pattern</span> when processing large arrays to improve occupancy and handle arbitrary array sizes. </div>

```cpp
// Vector addition with grid-stride loop
__global__ void vectorAddStrided(float* A, float* B, float* C, int n) {
    // Calculate global thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Stride = total number of threads
    int stride = blockDim.x * gridDim.x;
    
    // Process elements with stride
    for (int i = idx; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Launch with fewer blocks but each thread does more work
int main() {
    int n = 1 << 20;
    
    // Use fewer blocks than needed to cover all elements
    int blockSize = 256;
    int gridSize = 1024;  // Fixed number of blocks
    
    vectorAddStrided<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    // Total threads = 1024 * 256 = 262,144 threads
    // Each thread processes ~4 elements (1M / 262k)
}
```

## 3. Matrix Multiplication

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Matrix multiplication (矩阵乘法)</span> is a classic example of 2D thread organization. It demonstrates <span style="color:#2980B9;font-weight:700">2D thread blocks</span> and the performance benefits of <span style="color:#2980B9;font-weight:700">shared memory tiling</span>. </div>

### 1) Naive Matrix Multiplication

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Naive matrix multiplication</span> uses 2D grid and blocks. Each thread computes one output element. <span style="color:#C0392B;font-weight:700">Performance is poor due to uncoalesced global memory access</span> - used only for learning. </div>

```cpp
// Naive matrix multiplication: C = A * B
// A: MxK, B: KxN, C: MxN
__global__ void matrixMulNaive(float* A, float* B, float* C, 
                                int M, int N, int K) {
    // Compute row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Dot product of row of A and column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}

// Host code for naive multiplication
int main() {
    int M = 1024, N = 1024, K = 1024;
    
    // Allocate and initialize matrices (not shown)
    
    // 2D block and grid configuration
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
```

### 2) Optimized Matrix Multiplication with Shared Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Tiled matrix multiplication (分块矩阵乘法)</span> uses shared memory to reduce global memory accesses. <span style="color:#2980B9;font-weight:700">Use this pattern</span> for high-performance matrix operations. </div>

```cpp
// Optimized matrix multiplication with shared memory tiling
// Assumes block size = TILE_SIZE x TILE_SIZE
#define TILE_SIZE 16

__global__ void matrixMulShared(float* A, float* B, float* C,
                                 int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread coordinates within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column for this thread
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        if (row < M && (tile * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        if ((tile * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();  // Wait for tile load
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host code for optimized multiplication
int main() {
    int M = 1024, N = 1024, K = 1024;
    
    // 2D block with tile size
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">The optimized version is 10-30x faster</span> than naive multiplication for large matrices because it reduces global memory accesses from O(N³) to O(N³/TILE_SIZE). </div>

## 4. Multi-Thread Data Processing Examples

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> Beyond basic operations, CUDA threads can handle <span style="color:#E8600A;font-weight:700">complex data processing tasks</span>. These examples show <span style="color:#2980B9;font-weight:700">parallel reduction</span> and <span style="color:#2980B9;font-weight:700">image processing</span> patterns. </div>

### 1) Parallel Reduction (Sum)

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Parallel reduction (并行归约)</span> combines many values into one using a tree-based approach. <span style="color:#2980B9;font-weight:700">Use this pattern</span> for sum, max, min, or any associative operation. </div>

```cpp
// Parallel sum reduction
__global__ void reduceSum(float* input, float* output, int n) {
    __shared__ float shared[256];  // Assume block size 256
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    int tid = threadIdx.x;
    
    // Load two elements with coalesced access
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    shared[tid] = sum;
    __syncthreads();
    
    // Tree-based reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Host code for reduction
int main() {
    int n = 1 << 20;
    int blockSize = 256;
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    
    // Allocate output for partial sums
    float *d_partial;
    cudaMalloc(&d_partial, gridSize * sizeof(float));
    
    // First reduction pass
    reduceSum<<<gridSize, blockSize>>>(d_input, d_partial, n);
    
    // Second reduction pass (if needed)
    if (gridSize > 1) {
        reduceSum<<<1, blockSize>>>(d_partial, d_output, gridSize);
    } else {
        cudaMemcpy(d_output, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }
}
```

### 2) Image Processing: RGB to Grayscale

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Image processing</span> with 2D thread mapping. Each thread handles one pixel. <span style="color:#2980B9;font-weight:700">Use 2D blocks</span> for natural mapping to image coordinates. </div>

```cpp
// RGB to Grayscale conversion
// Input: RGB image (width * height * 3 bytes)
// Output: Grayscale image (width * height bytes)
__global__ void rgbToGrayscale(unsigned char* rgb, unsigned char* gray,
                                int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int rgb_idx = (row * width + col) * 3;
        int gray_idx = row * width + col;
        
        // Standard grayscale conversion: 0.299R + 0.587G + 0.114B
        unsigned char r = rgb[rgb_idx];
        unsigned char g = rgb[rgb_idx + 1];
        unsigned char b = rgb[rgb_idx + 2];
        
        gray[gray_idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Host code for image processing
int main() {
    int width = 1920, height = 1080;
    
    // 2D block configuration
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    rgbToGrayscale<<<gridDim, blockDim>>>(d_rgb, d_gray, width, height);
}
```

## Comparison of Programming Patterns

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Pattern</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Thread Mapping</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Best For</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Example</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Element-wise</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">1 thread : 1 element</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Vector ops, additions</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Vector addition</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Strided</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">1 thread : multiple elements</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Large arrays, any size</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Grid-stride loop</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">2D Mapping</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">1 thread : 1 pixel/element</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Images, matrices</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Matrix multiplication</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#C0392B; font-weight:700">Reduction</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Tree-based</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Sum, max, min operations</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Parallel sum</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Start with vector addition (向量加法) for 1D patterns, progress to matrix multiplication (矩阵乘法) for 2D tiling</span>, and <span style="color:#2980B9;font-weight:700">always include boundary checks and error handling</span> for robust CUDA programs. </div>
