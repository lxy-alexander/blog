---
title: "Thread Hierarchy"
published: 2026-03-15
description: "Thread Hierarchy"
image: ""
tags: ["cuda","threads","Thread Hierarchy"]
category: cuda / threads
draft: false
lang: ""
---

# I. CUDA Thread Hierarchy

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">CUDA threads</span> are organized in a three-level hierarchy: <span style="color:#2980B9;font-weight:700">threads → blocks → grid</span>. This hierarchy enables <span style="color:#2980B9;font-weight:700">scalable parallelism</span> where threads in the same block can cooperate, while blocks execute independently. Each thread has unique identifiers to determine its data portion. </div>

## 1. Thread Fundamentals

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Threads</span> are the basic execution units in CUDA. Thousands of threads run simultaneously on the GPU, each executing the same kernel code but processing different data portions. This is the foundation of <span style="color:#E8600A;font-weight:700">SIMT (Single Instruction Multiple Thread)</span> architecture. </div>

### 1) Thread Concept in GPU

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">GPU threads</span> are extremely lightweight compared to CPU threads. <span style="color:#2980B9;font-weight:700">Context switching is free</span> because each thread has its own registers. <span style="color:#2980B9;font-weight:700">Use thousands of threads</span> to hide memory latency and maximize GPU utilization. </div>

```cpp
// Simple kernel showing thread-level parallelism
__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID
    if (i < n) {  // Boundary check
        y[i] = a * x[i] + y[i];  // Each thread processes one element
    }
}

// Launch configuration: 256 threads per block, enough blocks to cover n
// saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, d_x, d_y);
```

## 2. Built-in Variables for Thread Identification

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> CUDA provides special variables that threads use to identify themselves and their position in the hierarchy. These variables are <span style="color:#E8600A;font-weight:700">built-in</span> and automatically defined for each thread when a kernel executes. </div>

### 1) threadIdx (线程索引)

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">threadIdx</span> is a 3-component vector (x, y, z) identifying a thread within its block. <span style="color:#2980B9;font-weight:700">Use it for intra-block addressing</span>, like indexing into shared memory or determining which portion of block data to process. </div>

```cpp
__global__ void threadIdxExample() {
    // 1D block
    int tid_1d = threadIdx.x;
    
    // 2D block (e.g., for matrix operations)
    int row = threadIdx.y;
    int col = threadIdx.x;
    int tid_2d = row * blockDim.x + col;  // Linearized index
    
    // 3D block (e.g., for volume processing)
    int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
    int tid_3d = z * blockDim.x * blockDim.y + y * blockDim.x + x;
    
    printf("Thread in block: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}
```

### 2) blockIdx (块索引)

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">blockIdx</span> identifies which block the thread belongs to within the grid. <span style="color:#2980B9;font-weight:700">Combine with threadIdx to compute global data indices</span> and distribute work across the entire dataset. </div>

```cpp
__global__ void vectorOperation(float* data, int n) {
    // Compute global index for 1D grid of 1D blocks
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For 2D grid of 2D blocks (common in image processing)
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx_2d = global_row * width + global_col;  // Linearized
    
    if (global_idx < n) {
        data[global_idx] *= 2.0f;
    }
}
```

### 3) blockDim (块尺寸) and gridDim (网格尺寸)

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">blockDim</span> gives the dimensions of each block (number of threads per block). <span style="color:#E8600A;font-weight:700">gridDim</span> gives the dimensions of the grid (number of blocks). <span style="color:#2980B9;font-weight:700">Use them for boundary checking</span> and to calculate total threads. </div>

```cpp
__global__ void processMatrix(float* matrix, int width, int height) {
    // Calculate global position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check using grid and block dimensions
    if (x < width && y < height) {
        int idx = y * width + x;
        
        // Print thread's position in hierarchy
        printf("Thread at (%d,%d) in block (%d,%d) of grid (%d,%d)\n",
               threadIdx.x, threadIdx.y,
               blockIdx.x, blockIdx.y,
               gridDim.x, gridDim.y);
        
        matrix[idx] = process_pixel(matrix[idx]);
    }
}
```

## 3. Thread Hierarchy Organization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The hierarchy <span style="color:#E8600A;font-weight:700">Thread → Block → Grid</span> maps naturally to data decomposition. Threads in the same block can <span style="color:#2980B9;font-weight:700">synchronize and share data via shared memory</span>. Blocks are independent units that can execute in any order across SMs. </div>

### 1) Thread to Data Mapping Strategies

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Mapping strategies</span> determine how threads are assigned to data elements. <span style="color:#2980B9;font-weight:700">Choose based on data dimensionality</span>: 1D for vectors/arrays, 2D for matrices/images, 3D for volumes. <span style="color:#C0392B;font-weight:700">Poor mapping</span> can lead to memory access inefficiency. </div>

```cpp
// 1D Mapping: Each thread handles one element
__global__ void oneDimensional(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = compute(data[tid]);
    }
}

// 2D Mapping: Each thread handles one pixel
__global__ void twoDimensional(float* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = filter(image[idx]);
    }
}

// Strided Mapping: Each thread handles multiple elements (grid stride loop)
__global__ void stridedMapping(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  // Total threads
    
    for (int i = tid; i < n; i += stride) {
        data[i] = process(data[i]);
    }
}
```

### 2) Block and Grid Dimension Calculation

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Choosing block dimensions</span> affects occupancy and resource usage. <span style="color:#2980B9;font-weight:700">Block size should be a multiple of 32 (warp size)</span> for efficiency. <span style="color:#C0392B;font-weight:700">Grid size</span> should be large enough to keep all SMs busy. </div>

```cpp
// Host code showing launch configuration calculation
int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    // Allocate and initialize memory
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // Calculate launch configuration
    int blockSize = 256;  // Optimal block size (multiple of 32)
    
    // Calculate grid size to cover all elements
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // For 2D problems
    dim3 blockDim2D(16, 16);  // 256 threads total (16*16)
    dim3 gridDim2D(
        (width + blockDim2D.x - 1) / blockDim2D.x,
        (height + blockDim2D.y - 1) / blockDim2D.y
    );
    
    // Launch kernel
    myKernel<<<gridSize, blockSize>>>(d_data, N);
    myKernel2D<<<gridDim2D, blockDim2D>>>(d_image, width, height);
    
    cudaFree(d_data);
    return 0;
}
```

## Thread Hierarchy Summary

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Level</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Built-in Variables</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Communication</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Synchronization</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Thread</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;"><code>threadIdx</code></td>
    <td style="padding: 10px; border: 1px solid #ddd;">None (private)</td>
    <td style="padding: 10px; border: 1px solid #ddd;">None needed</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Block</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;"><code>blockIdx, blockDim</code></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Shared Memory</td>
    <td style="padding: 10px; border: 1px solid #ddd;"><code>__syncthreads()</code></td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Grid</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;"><code>gridDim</code></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Global Memory</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Kernel launch boundary</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Use threadIdx for intra-block indexing, blockIdx for inter-block distribution</span>, and always <span style="color:#2980B9;font-weight:700">combine them with blockDim/gridDim</span> to compute global data positions while <span style="color:#C0392B;font-weight:700">checking boundaries</span> to prevent out-of-bounds access. </div>
