---
title: "CUDA Memory Hierarchy"
published: 2026-03-15
description: "CUDA Memory Hierarchy"
image: ""
tags: ["cuda","memory","CUDA Memory Hierarchy"]
category: cuda / memory
draft: false
lang: ""
---


# I. CUDA Memory Hierarchy

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">CUDA memory hierarchy</span> consists of multiple memory types with different speeds, scopes, and purposes. Understanding this hierarchy is crucial for <span style="color:#2980B9;font-weight:700">optimizing GPU performance</span>. The fastest memory is closest to the <span style="color:#E8600A;font-weight:700">Streaming Multiprocessors (SMs)</span>, while larger but slower memory resides off-chip. </div>

## 1. Registers

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Registers</span> are the fastest memory in the GPU hierarchy, located on-chip within each <span style="color:#E8600A;font-weight:700">Streaming Multiprocessor (SM)</span>. They are <span style="color:#2980B9;font-weight:700">private to each thread</span> and automatically managed by the compiler. </div>

### 1) Register Characteristics

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Registers</span> store local variables and intermediate computation results. <span style="color:#2980B9;font-weight:700">Use them for frequently accessed data</span> within a thread. <span style="color:#C0392B;font-weight:700">Excessive register usage reduces occupancy</span> (active threads per SM) and limits parallel execution. </div>

```cpp
// Good register usage - local variables stored in registers
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Stored in register
    float local_sum = 0.0f;  // Stored in register
    
    if (idx < n) {
        local_sum = a[idx] + b[idx];  // Register operations
        c[idx] = local_sum;
    }
}
```

## 2. Local Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Local memory</span> physically resides in <span style="color:#E8600A;font-weight:700">DRAM</span> (global memory) but is managed by the compiler. It's used when registers are insufficient or for arrays with <span style="color:#C0392B;font-weight:700">non-compile-time indices</span>. </div>

### 1) Register Spilling

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Register spilling</span> occurs when a thread uses more variables than available registers. <span style="color:#C0392B;font-weight:700">This forces data into local memory</span>, dramatically reducing performance. <span style="color:#2980B9;font-weight:700">Check compiler reports</span> to identify register spilling. </div>

```cpp
// Bad example - may cause register spilling since Exceeding the total register limit of a single SM (streaming multiprocessor)
__global__ void spillExample() {
    int large_array[100];  // Too large for registers -> local memory
    for (int i = 0; i < 100; i++) {  // the index i is dynamic, 
        large_array[i] = i;  // Slow local memory access
    }
}

// Better approach - use shared memory for large arrays
__global__ void betterExample() {
    __shared__ int shared_array[100];  // Shared memory instead
    // Process data...
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> Use <span style="color:#2980B9;font-weight:700">nvcc --ptxas-options=-v</span> to see register and local memory usage during compilation. </div>

## 3. Shared Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory</span> is on-chip memory accessible by all threads in a block. It's <span style="color:#2980B9;font-weight:700">~100x faster than global memory</span> and enables <span style="color:#2980B9;font-weight:700">inter-thread communication</span> and data reuse. </div>

### 1) Inter-thread Communication

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory</span> allows threads in a block to exchange data and collaborate. <span style="color:#2980B9;font-weight:700">Use __syncthreads()</span> to ensure all threads have finished accessing shared memory before proceeding. </div>

```cpp
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int N) {
    // N = matrix dimension, e.g. N=64 means A/B/C are all 64x64
    // assumes N is a multiple of 16 (tile size)
    // element A[i][j] is stored at A[i * N + j] (row-major)
    __shared__ float sharedA[16][16];       // tile of A in shared memory (fast, block-private)
    __shared__ float sharedB[16][16];       // tile of B in shared memory (fast, block-private)

    int row = blockIdx.y * 16 + threadIdx.y; // global row index of C this thread owns
    int col = blockIdx.x * 16 + threadIdx.x; // global col index of C this thread owns
    int tx = threadIdx.x, ty = threadIdx.y;  // local position within the 16x16 block

    float sum = 0.0f;                        // accumulator in register, reset once per thread

    for (int t = 0; t < N / 16; t++) {      // iterate over tiles along K dimension
        sharedA[ty][tx] = A[row * N + (t * 16 + tx)];   // load one element of A tile
        sharedB[ty][tx] = B[(t * 16 + ty) * N + col];   // load one element of B tile

        __syncthreads();                     // wait: all threads must finish loading before compute

        for (int k = 0; k < 16; k++)
            sum += sharedA[ty][k] * sharedB[k][tx]; // dot product from shared memory (fast)

        __syncthreads();                     // wait: all threads done reading before next tile load
    }

    C[row * N + col] = sum;                  // write final result to global memory once
}
```

>### Analysis: Memory Latency Amortization
>
>#### 1. Naive Implementation (Direct Global Access)
>
>For each element in $C$, we perform a dot product of length $N$.
>
>-   **Total Global Reads**: $2N^3$ (Each of the $N^2$ elements requires $N$ reads from $A$ and $N$ from $B$).
>-   **Estimated Latency**: $2N^3 \times 500$ cycles.
>-   **Bottleneck**: Extremely **Memory-Bound**. The compute units spend most of their time idling, waiting for data.
>
>#### 2. Tiled Implementation (Shared Memory)
>
>By loading data into $16 \times 16$ tiles:
>
>-   **Global to Shared Transfer**:
>    -   Each element is loaded into Shared Memory once per tile.
>    -   Since each element is reused 16 times within the block, the total global reads are reduced by a factor of 16.
>    -   **Global Reads**: $\frac{2N^3}{16}$
>    -   **Global Latency**: $\frac{2N^3}{16} \times 500 = 62.5N^3$ cycles.
>-   **Shared Memory Reads**:
>    -   The compute units still perform $2N^3$ reads, but they hit the fast Shared Memory.
>    -   **Shared Latency**: $2N^3 \times 5 = 10N^3$ cycles.
>
>#### 3. Final Comparison
>
>-   **Total Tiled Latency**: $62.5N^3 + 10N^3 = \mathbf{72.5N^3}$
>-   **Effective Speedup**: $\frac{1000N^3}{72.5N^3} \approx \mathbf{13.8x}$

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Bank conflicts</span> occur when multiple threads access the same shared memory bank, causing serialization. Design access patterns to avoid this. </div>

## 4. Global Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Global memory</span> is the main GPU DRAM, accessible by all threads and the host. It has <span style="color:#C0392B;font-weight:700">high latency (~400-800 cycles)</span> but high bandwidth. <span style="color:#2980B9;font-weight:700">Manual allocation (cudaMalloc) and deallocation (cudaFree)</span> are required. </div>

### 1) Coalesced Access Pattern

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Coalesced access</span> means threads in a warp access consecutive memory addresses. <span style="color:#2980B9;font-weight:700">This maximizes memory bandwidth utilization</span>. <span style="color:#C0392B;font-weight:700">Non-coalesced access</span> can severely degrade performance. </div>

```cpp
// Good: Coalesced access (thread i accesses index i)
__global__ void coalescedAccess(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;  // Consecutive addresses
    }
}

// Bad: Strided access (non-coalesced)
__global__ void stridedAccess(float* input, float* output, int stride, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx * stride] = input[idx * stride];  // Non-consecutive
    }
}
```

## 5. Constant Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Constant memory</span> is 64KB of cached read-only memory. <span style="color:#2980B9;font-weight:700">Best when all threads in a warp access the same address</span> - data is broadcast in one cycle. <span style="color:#C0392B;font-weight:700">Different addresses cause serialization</span>. </div>

### 1) Constant Memory Usage

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Constant memory</span> is ideal for lookup tables and kernel parameters that are read-only and accessed uniformly by all threads. <span style="color:#2980B9;font-weight:700">Declare with __constant__</span> and copy data using <span style="color:#2980B9;font-weight:700">cudaMemcpyToSymbol</span>. </div>

```cpp
// Constant memory declaration
__constant__ float lookupTable[256];

// Host code
float h_table[256] = { /* initialize values */ };
cudaMemcpyToSymbol(lookupTable, h_table, sizeof(float) * 256);

// Kernel using constant memory
__global__ void kernelUsingConstant(float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // All threads in warp access same constant address - fast!
        output[idx] = lookupTable[idx % 256];
    }
}
```

## 6. Texture Memory / Read-Only Cache

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Texture memory</span> and the <span style="color:#E8600A;font-weight:700">read-only cache</span> provide optimized access for 2D spatial locality. Modern GPUs can use <span style="color:#2980B9;font-weight:700">__ldg() intrinsic</span> to leverage the read-only cache directly. </div>

### 1) Read-Only Cache with __ldg()

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">__ldg()</span> forces loading through the read-only data cache. <span style="color:#2980B9;font-weight:700">Use it for read-only global memory accesses</span> to reduce latency and avoid polluting L1 cache. </div>

```cpp
// Using read-only cache for texture-like access
__global__ void textureLikeAccess(const float* __restrict__ input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // __ldg() ensures access through read-only cache
        float val = __ldg(&input[idx]);
        output[idx] = val * 2.0f;
    }
}
```

## Memory Type Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Memory Type</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Location</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Scope</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Speed</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Registers</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">SM On-chip</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Per-thread</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Fastest</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Shared Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">SM On-chip</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Per-block</td>
    <td style="padding: 10px; border: 1px solid #ddd;">~1-2 cycles</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Global Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">DRAM</td>
    <td style="padding: 10px; border: 1px solid #ddd;">All threads + host</td>
    <td style="padding: 10px; border: 1px solid #ddd;">~400-800 cycles</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Constant Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">DRAM + Cache</td>
    <td style="padding: 10px; border: 1px solid #ddd;">All threads</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Fast if uniform</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Use registers for private data, shared memory for block collaboration, and ensure coalesced global memory access</span> while <span style="color:#C0392B;font-weight:700">avoiding register spilling and bank conflicts</span> for optimal CUDA performance. </div>
