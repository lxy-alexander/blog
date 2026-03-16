---
title: "Memory Access Patterns and Optimization"
published: 2026-03-15
description: "Memory Access Patterns and Optimization"
image: ""
tags: ["cuda","memory","Memory Access Patterns and Optimization"]
category: cuda / memory
draft: false
lang: ""
---

# I. Memory Access Patterns and Optimization

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Memory access optimization</span> is critical for CUDA performance. Three key techniques are <span style="color:#2980B9;font-weight:700">coalesced global memory access</span> to maximize bandwidth, <span style="color:#2980B9;font-weight:700">bank conflict avoidance</span> in shared memory, and <span style="color:#2980B9;font-weight:700">register reuse</span> to minimize slow memory accesses. These patterns determine whether your kernel achieves near-peak or only fractional memory throughput. </div>

## 1. Coalesced Memory Access

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Coalesced memory access (合并内存访问)</span> means consecutive threads in a warp access consecutive memory locations. This allows the hardware to <span style="color:#2980B9;font-weight:700">combine 32 memory requests into one or two cache-line transactions</span>, dramatically increasing effective bandwidth. </div>

### 1) Coalesced vs. Non-Coalesced Patterns

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Coalesced access</span> occurs when thread i accesses address base + i * element_size. <span style="color:#C0392B;font-weight:700">Non-coalesced access</span> (strided, random, or misaligned) can multiply memory transactions by up to 32x, severely degrading performance. </div>

```cpp
// GOOD: Perfect coalescing - consecutive threads access consecutive addresses
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Thread i accesses element i
    }
}

// BAD: Strided access - destroys coalescing
__global__ void stridedAccess(float* input, float* output, int stride, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx * stride < n) {
        output[idx * stride] = input[idx * stride];  // Threads access spaced addresses
        // Warp 0: accesses 0, stride, 2*stride, ... - not consecutive
    }
}

// BAD: Misaligned access - crosses cache line boundaries
__global__ void misalignedAccess(float* input, float* output, int offset, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        output[idx] = input[idx + offset];  // If offset not multiple of 32, access misaligned
    }
}

// GOOD: Structure-of-Arrays (SoA) for coalescing
struct SoA {
    float* x;
    float* y;
    float* z;
};

__global__ void soaKernel(SoA data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data.x[idx] = data.x[idx] * 2.0f;  // Coalesced access to x array
        data.y[idx] = data.y[idx] + 1.0f;  // Coalesced access to y array
        data.z[idx] = data.z[idx] - 3.0f;  // Coalesced access to z array
    }
}

// BAD: Array-of-Structures (AoS) - poor coalescing
struct AoS {
    float x, y, z;
};

__global__ void aosKernel(AoS* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx].x = data[idx].x * 2.0f;  // Threads access spaced by 12 bytes
        data[idx].y = data[idx].y + 1.0f;  // Even worse: offset within structure
        data[idx].z = data[idx].z - 3.0f;
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">For modern GPUs, coalescing requirements are more relaxed</span> - any access pattern where all 32 threads access the same 128-byte cache line can be coalesced. However, best performance still comes from perfectly consecutive access. </div>

### 2) Matrix Access Patterns

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Matrix memory layout</span> significantly affects coalescing. <span style="color:#2980B9;font-weight:700">Row-major access with row-major storage</span> achieves coalescing when consecutive threads access consecutive columns. </div>

```cpp
// GOOD: Row-major access with row-major storage (coalesced)
// Matrix stored as [row * width + col]
__global__ void rowMajorCoalesced(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x dimension = columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;  // Consecutive threads: different col, same row
        matrix[idx] = process(matrix[idx]);  // Coalesced! Threads access consecutive columns
    }
}

// BAD: Column-major access with row-major storage (non-coalesced)
__global__ void columnMajorNonCoalesced(float* matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = col * height + row;  // Consecutive threads: different col, spacing = height
        matrix[idx] = process(matrix[idx]);  // Non-coalesced! Threads access spaced addresses
    }
}

// FIX: Transpose before column-major processing
__global__ void transposeCoalesced(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < height) {
        // Coalesced read from input (row-major)
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;  // Swap block indices
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        // Coalesced write to output (now column-major data in row-major storage)
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

## 2. Bank Conflict Avoidance

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory bank conflicts (共享内存银行冲突)</span> occur when multiple threads access different addresses in the same memory bank. <span style="color:#C0392B;font-weight:700">This serializes the accesses</span>, reducing effective bandwidth from 32-way parallel to sequential. Modern GPUs have 32 banks, each 4 bytes wide. </div>

### 1) Understanding Bank Conflicts

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Bank index</span> is calculated as (byte_address / 4) % 32. <span style="color:#2980B9;font-weight:700">Conflicts happen when two threads access different addresses with the same bank index</span>. <span style="color:#2980B9;font-weight:700">No conflict if all threads access the same address</span> (broadcast). </div>

```cpp
// GOOD: No bank conflicts - each thread accesses different bank
__global__ void noBankConflict() {
    __shared__ float shared[32][32];  // 32x32 array
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each row: thread x accesses column x - different banks
    float val = shared[ty][tx];  // Bank = tx % 32 - all different
    shared[ty][tx] = val * 2.0f;
}

// BAD: 2-way bank conflicts - threads 0 and 16 access same bank
__global__ void twoWayBankConflict() {
    __shared__ float shared[32];  // 1D array
    
    int tx = threadIdx.x;
    
    // Threads 0 and 16 both access bank 0
    // Threads 1 and 17 both access bank 1, etc.
    float val = shared[tx * 2];  // Bank = (tx * 8) % 32? Actually: address/4 % 32
    // For tx=0: bank 0, tx=16: also bank 0 - 2-way conflict!
}

// BAD: 32-way bank conflict (worst case)
__global__ void worstBankConflict() {
    __shared__ float shared[32];
    
    int tx = threadIdx.x;
    
    // All threads access the same bank
    float val = shared[0];  // All access bank 0 - serialized 32x slower
}

// GOOD: Padding to avoid conflicts
__global__ void paddedAccess() {
    // Add padding to break power-of-two stride patterns
    __shared__ float shared[32][33];  // 33 columns instead of 32
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Access pattern that would conflict with 32 columns
    // With 33 columns, bank = (ty*33 + tx) % 32 - better distribution
    float val = shared[ty][tx * 2];  // Less likely to conflict
}
```

### 2) Common Conflict Patterns and Solutions

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Common conflict patterns</span> include power-of-two strides and transpose operations. <span style="color:#2980B9;font-weight:700">Solutions include padding, changing access patterns, or using vector types</span> to combine accesses. </div>

```cpp
// Matrix transpose with bank conflict analysis
__global__ void transposeConflicts(float* input, float* output, int width) {
    __shared__ float tile[32][32];  // 32x32 tile
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        // Coalesced read - no bank conflicts
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // PROBLEM: Writing transposed data causes bank conflicts!
    // Thread (tx,ty) writes to tile[tx][ty]
    // In a warp, ty varies, tx constant - all access different rows but same column
    // With 32 columns, this is column index = constant -> same bank for all threads!
    
    x = blockIdx.y * 32 + threadIdx.y;  // Note: swapped
    y = blockIdx.x * 32 + threadIdx.x;
    
    if (x < width && y < width) {
        // This write suffers bank conflicts
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// SOLUTION 1: Padding to avoid conflicts
__global__ void transposePadded(float* input, float* output, int width) {
    __shared__ float tile[32][33];  // Pad to 33 columns
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.y;
    y = blockIdx.x * 32 + threadIdx.x;
    
    if (x < width && y < width) {
        // With padding, column index doesn't map directly to bank
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];  // Reduced conflicts
    }
}

// SOLUTION 2: Use vector types to combine accesses
__global__ void transposeVector(float4* input, float4* output, int width) {
    __shared__ float4 tile[32][8];  // Each float4 holds 4 floats
    
    int x = blockIdx.x * 32 + threadIdx.x * 4;  // Each thread handles 4 columns
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * (width/4) + threadIdx.x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.y * 4;
    y = blockIdx.x * 32 + threadIdx.x;
    
    if (x < width && y < width) {
        output[y * (width/4) + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

## 3. Register Reuse

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Register reuse (寄存器重用)</span> means keeping frequently accessed data in registers rather than reloading from slower memory. <span style="color:#2980B9;font-weight:700">Registers are the fastest memory</span> (single-cycle access) and <span style="color:#2980B9;font-weight:700">reusing them reduces pressure on memory bandwidth</span>. </div>

### 1) Register vs. Memory Access

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Register reuse</span> is achieved by loading data once and using it multiple times in computation. <span style="color:#C0392B;font-weight:700">Excessive global memory access</span> without reuse leads to memory-bound performance. </div>

```cpp
// BAD: No register reuse - repeatedly accessing global memory
__global__ void poorReuse(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Each operation loads from global memory separately
        data[idx] = data[idx] * 2.0f;  // Load 1
        data[idx] = data[idx] + 1.0f;  // Load 2
        data[idx] = data[idx] / 3.0f;  // Load 3
    }
}

// GOOD: Register reuse - load once, use multiple times
__global__ void goodReuse(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        float val = data[idx];  // Single load into register
        
        // All operations use register value
        val = val * 2.0f;
        val = val + 1.0f;
        val = val / 3.0f;
        
        data[idx] = val;  // Single store
    }
}

// Stencil computation with register reuse
__global__ void stencil1D(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= 1 && idx < n-1) {
        // Load three values into registers
        float left = input[idx-1];
        float center = input[idx];
        float right = input[idx+1];
        
        // Reuse registers for multiple computations
        float sum = left + center + right;
        float avg = sum / 3.0f;
        float weighted = left * 0.25f + center * 0.5f + right * 0.25f;
        
        // Could do more computations without additional loads
        output[idx] = (avg + weighted) * 0.5f;
    }
}
```

### 2) Tiling for Register Reuse

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Tiling with register reuse</span> loads a block of data into registers and performs multiple computations. <span style="color:#2980B9;font-weight:700">Use this pattern for operations with high arithmetic intensity</span> like matrix multiplication and convolutions. </div>

```cpp
// Matrix multiplication with register reuse
// Each thread computes multiple output elements
#define TILE_SIZE 16
#define NUM_ELEMS 4  // Each thread computes 4 outputs

__global__ void matrixMulRegisterReuse(float* A, float* B, float* C,
                                        int M, int N, int K) {
    // Each thread computes 2x2 output tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty * 2;  // Each thread handles 2 rows
    int col = blockIdx.x * TILE_SIZE + tx * 2;  // Each thread handles 2 cols
    
    // Registers for partial sums (4 per thread)
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        // Load tile (coalesced)
        As[ty][tx] = A[row * K + tile + tx];
        Bs[ty][tx] = B[(tile + ty) * N + col];
        __syncthreads();
        
        // Compute using registers - reuse each loaded value multiple times
        for (int k = 0; k < TILE_SIZE; k++) {
            float a00 = As[ty*2][k];      // Load A values into registers
            float a01 = As[ty*2+1][k];
            float b00 = Bs[k][tx*2];       // Load B values into registers
            float b01 = Bs[k][tx*2+1];
            
            // Update all 4 sums with these values
            sum[0] += a00 * b00;  // (0,0)
            sum[1] += a00 * b01;  // (0,1)
            sum[2] += a01 * b00;  // (1,0)
            sum[3] += a01 * b01;  // (1,1)
        }
        
        __syncthreads();
    }
    
    // Write results (coalesced)
    if (row < M && col < N) {
        C[row * N + col] = sum[0];
        if (col+1 < N) C[row * N + col+1] = sum[1];
        if (row+1 < M) C[(row+1) * N + col] = sum[2];
        if (row+1 < M && col+1 < N) C[(row+1) * N + col+1] = sum[3];
    }
}
```

## Memory Optimization Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Technique</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Memory Type</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Impact</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Common Pitfall</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Coalescing</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Global</td>
    <td style="padding: 10px; border: 1px solid #ddd;">1x to 32x bandwidth</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Strided access patterns</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Bank Conflict</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Shared</td>
    <td style="padding: 10px; border: 1px solid #ddd;">1x to 32x slowdown</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Power-of-two strides</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Register Reuse</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Register</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Reduces memory traffic</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Register spilling</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Maximize coalesced global access, pad shared memory to avoid bank conflicts</span>, and <span style="color:#2980B9;font-weight:700">reuse data in registers</span> – these three optimizations transform memory-bound kernels into compute-bound ones. </div>
