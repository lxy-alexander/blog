---
title: "Thread Optimization Techniques"
published: 2026-03-15
description: "Thread Optimization Techniques"
image: ""
tags: ["cuda","threads","Thread Optimization Techniques"]
category: cuda / threads
draft: false
lang: ""
---

# I. Thread Optimization Techniques

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Thread optimization</span> is crucial for achieving peak GPU performance. Key techniques include <span style="color:#2980B9;font-weight:700">coalesced memory access</span>, <span style="color:#2980B9;font-weight:700">shared memory utilization</span>, <span style="color:#C0392B;font-weight:700">branch minimization</span>, and <span style="color:#E8600A;font-weight:700">occupancy optimization</span>. These techniques work together to maximize parallel efficiency. </div>

## 1. Memory Access Alignment and Coalescing

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Coalesced memory access (合并内存访问)</span> means threads in a warp access consecutive memory locations. This allows the hardware to <span style="color:#2980B9;font-weight:700">combine multiple memory requests into a single transaction</span>, maximizing bandwidth utilization. </div>

### 1) Coalesced vs. Non-Coalesced Access

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Coalesced access (合并访问)</span> occurs when thread i accesses address base + i * element_size. <span style="color:#C0392B;font-weight:700">Strided or random access</span> destroys coalescing and multiplies memory transactions. <span style="color:#2980B9;font-weight:700">Always design data layout and indexing for coalescing</span>. </div>

```cpp
// GOOD: Coalesced access - threads access consecutive addresses
__global__ void coalescedCopy(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = input[idx];  // Thread i accesses element i
    }
}

// BAD: Strided access - non-coalesced
__global__ void stridedCopy(float* input, float* output, int stride, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx * stride < n) {
        output[idx * stride] = input[idx * stride];  // Threads access spaced addresses
    }
}
```

## 2. Using Shared Memory for Thread Data Exchange

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory (共享内存)</span> is on-chip memory that enables thread cooperation and data reuse. <span style="color:#2980B9;font-weight:700">Use it to cache global memory data</span> that multiple threads need, reducing redundant global accesses by up to 100x. </div>

### 1) Shared Memory for Data Reuse

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory tiling (共享内存分块)</span> dramatically improves performance for operations with data reuse. <span style="color:#2980B9;font-weight:700">Load tiles from global to shared memory</span>, then compute using fast shared memory accesses. </div>

```cpp
// Matrix multiplication with shared memory tiling
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];  // Shared memory tiles
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < N / 16; ++tile) {
        // Load into shared memory (coalesced global loads)
        As[ty][tx] = A[row * N + tile * 16 + tx];
        Bs[ty][tx] = B[(tile * 16 + ty) * N + col];
        
        __syncthreads();  // Ensure tile is loaded
        
        // Compute using shared memory (fast)
        for (int k = 0; k < 16; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // Avoid overwriting before all threads finish
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

## 3. Reducing Thread Branching and Divergent Access

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Control flow divergence (控制流分歧)</span> and <span style="color:#E8600A;font-weight:700">non-uniform memory access patterns</span> reduce SIMD efficiency. <span style="color:#2980B9;font-weight:700">Restructure algorithms to maintain uniform execution paths</span> within warps. </div>

### 1) Branch Reduction Techniques

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Predication (预测执行)</span> and <span style="color:#E8600A;font-weight:700">data reorganization</span> help reduce branching. <span style="color:#2980B9;font-weight:700">Use branchless programming where possible</span>. <span style="color:#C0392B;font-weight:700">Avoid threadIdx-dependent conditions</span> that split warps. </div>

```cpp
// BAD: Excessive branching causing divergence
__global__ void branchingBad(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    
    if (data[idx] > 0) {
        data[idx] = processPositive(data[idx]);
    } else {
        data[idx] = processNegative(data[idx]);
    }
}

// GOOD: Branch reduction through predication
__global__ void branchingGood(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    
    float val = data[idx];
    int isPositive = (val > 0);
    
    // Compute both paths and select without branching
    float posResult = processPositive(val);
    float negResult = processNegative(val);
    
    data[idx] = isPositive ? posResult : negResult;
}
```

## 4. Occupancy Analysis

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Occupancy (线程占用率)</span> is the ratio of active warps to maximum possible warps on an SM. <span style="color:#2980B9;font-weight:700">Higher occupancy helps hide memory latency</span> but isn't always the primary performance factor. </div>

### 1) Calculating and Optimizing Occupancy

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Occupancy</span> is limited by block size, register usage, and shared memory usage. <span style="color:#2980B9;font-weight:700">Use CUDA occupancy APIs</span> to find optimal configurations. <span style="color:#C0392B;font-weight:700">Too many registers per thread</span> reduces occupancy. </div>

```cpp
// Host code for occupancy analysis
int main() {
    int blockSize;
    int minGridSize;
    
    // Get minimum grid size for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                        myKernel, 0, 0);
    
    // Get active blocks per SM for specific configuration
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, 
                                                   blockSize, 0);
    
    // Calculate occupancy
    int maxWarpsPerSM = 64;  // For compute capability 7.x
    int warpsPerBlock = (blockSize + 31) / 32;
    int activeWarps = numBlocks * warpsPerBlock;
    float occupancy = (float)activeWarps / maxWarpsPerSM * 100.0f;
    
    printf("Block size: %d, Occupancy: %.1f%%\n", blockSize, occupancy);
    
    // Launch with optimized configuration
    int gridSize = (N + blockSize - 1) / blockSize;
    myKernel<<<gridSize, blockSize>>>(d_data, N);
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Maximum occupancy doesn't always mean maximum performance</span>. Sometimes lower occupancy with more registers per thread (less register spilling) yields better results. </div>

## Optimization Techniques Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Technique</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Primary Benefit</th>
    <th style="padding: 10px; border: 1px solid #ddd;">When to Use</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Coalesced Access</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Maximizes bandwidth</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Always</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Shared Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Reduces global accesses</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Data reuse patterns</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Branch Reduction</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Maintains SIMD efficiency</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Data-dependent execution</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#C0392B; font-weight:700">High Occupancy</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Hides memory latency</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Memory-bound kernels</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Optimize CUDA threads through coalesced memory access (合并内存访问), shared memory tiling (共享内存分块), branch reduction</span>, and <span style="color:#2980B9;font-weight:700">balanced occupancy (线程占用率)</span> – always profile to find the true bottleneck. </div>
