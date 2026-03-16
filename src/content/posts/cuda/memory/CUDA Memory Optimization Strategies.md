---
title: "CUDA Memory Optimization Strategies"
published: 2026-03-15
description: "CUDA Memory Optimization Strategies"
image: ""
tags: ["cuda","memory","CUDA Memory Optimization Strategies"]
category: cuda / memory
draft: false
lang: ""
---

# I. Common CUDA Optimization Strategies

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">CUDA optimization strategies</span> focus on maximizing memory throughput and minimizing latency. Key approaches include <span style="color:#2980B9;font-weight:700">using shared memory instead of global memory</span>, <span style="color:#2980B9;font-weight:700">storing frequently used data in registers</span>, <span style="color:#2980B9;font-weight:700">leveraging constant memory for read-only data</span>, and <span style="color:#2980B9;font-weight:700">using pinned memory for faster host-device transfers</span>. These techniques can yield 10-100x performance improvements. </div>

## 1. Using Shared Memory Instead of Global Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory (共享内存)</span> is on-chip memory with ~100x lower latency than global memory. <span style="color:#2980B9;font-weight:700">Use it to cache data that is accessed multiple times by threads in the same block</span>, reducing redundant global memory traffic. </div>

### 1) Shared Memory for Data Reuse

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Shared memory tiling</span> loads a block of data once and reuses it many times. <span style="color:#2980B9;font-weight:700">Use for stencil(masked) computations, matrix multiplication, and any algorithm with data locality</span>. </div>

```cpp
// BAD: Repeated global memory accesses
__global__ void stencilBad(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= 1 && idx < n-1) {
        // Each thread loads the same neighboring elements multiple times
        // Different threads reload the same data from different blocks
        // 相邻线程确实重复读了同一个地址，但现代 GPU 有 L1/L2 cache，input[idx-1] 很可能已经被相邻线程缓存，实际代价没有注释暗示的那么大。stencil 用 shared memory 的主要收益是减少 cache 压力，不是完全避免重复读。
        output[idx] = (input[idx-1] + input[idx] + input[idx+1]) / 3.0f;
    }
}

// GOOD: Shared memory for stencil
__global__ void stencilGood(float* input, float* output, int n) {
    __shared__ float shared[256 + 2];  // Halo cells for neighbors
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    // Load data into shared memory (with halo)
    if (idx < n) {
        shared[tid + 1] = input[idx];  // Center
    }
    if (tid == 0 && blockIdx.x > 0) {
        // Load left halo from previous block
        shared[0] = input[idx - 1];
    }
    if (tid == blockDim.x - 1 && idx + 1 < n) {
        // Load right halo from next block
        shared[blockDim.x + 1] = input[idx + 1];
    }
    
    __syncthreads();  // Ensure all data loaded
    
    // Compute using fast shared memory
    if (idx >= 1 && idx < n-1) {
        float left = shared[tid];
        float center = shared[tid + 1];
        float right = shared[tid + 2];
        output[idx] = (left + center + right) / 3.0f;
    }
}

// Matrix multiplication with shared memory
#define TILE_SIZE 16
__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < N / TILE_SIZE; tile++) {
        // Load tile into shared memory (coalesced)
        As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        
        __syncthreads();  // Wait for tile load
        
        // Compute using shared memory (fast)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Shared memory is limited (typically 48-96KB per SM)</span>. Too much shared memory per block reduces occupancy. Balance shared memory usage with block size for optimal performance. </div>

## 2. Storing Frequently Used Data in Registers

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Registers (寄存器)</span> are the fastest memory in the GPU hierarchy with single-cycle access. <span style="color:#2980B9;font-weight:700">Store frequently accessed variables in registers</span> by declaring them as local variables. <span style="color:#C0392B;font-weight:700">Avoid register spilling</span> by not exceeding the register limit per thread. </div>

### 1) Register Optimization Techniques

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Register reuse</span> loads data once and performs multiple operations. <span style="color:#2980B9;font-weight:700">Use compiler hints with __launch_bounds__</span> to control register usage. <span style="color:#2980B9;font-weight:700">Manual unrolling</span> can increase register reuse. </div>

```cpp
// BAD: No register reuse - repeated global loads
__global__ void badRegisterUse(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Each operation loads from global memory
        data[idx] = data[idx] * 2.0f;  // Load 1
        data[idx] = data[idx] + 1.0f;  // Load 2
        data[idx] = data[idx] / 3.0f;  // Load 3
    }
}

// GOOD: Register reuse - load once, use multiple times
__global__ void goodRegisterUse(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        float val = data[idx];  // Single load into register
        
        // All operations use register (fast)
        val = val * 2.0f;
        val = val + 1.0f;
        val = val / 3.0f;
        
        data[idx] = val;  // Single store
    }
}

// Using registers for multiple values per thread
__global__ void vectorMult(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;  // Each thread handles 4 elements
    
    if (idx + 3 < n) {
        // Load 4 values into registers
        float4 vals = *reinterpret_cast<float4*>(&input[idx]);
        
        // Process in registers
        vals.x = vals.x * 2.0f;
        vals.y = vals.y * 2.0f;
        vals.z = vals.z * 2.0f;
        vals.w = vals.w * 2.0f;
        
        // Store results
        *reinterpret_cast<float4*>(&output[idx]) = vals;
    }
}

// Launch bounds to control register usage
__launch_bounds__(256, 4)  // Min 4 blocks per SM, max 256 threads/block
__global__ void registerOptimizedKernel(float* data, int n) {
    // Compiler will try to limit register usage to achieve
    // at least 4 blocks (1024 threads) per SM
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        float a[8], b[8], c[8];  // Compiler may keep in registers
        for (int i = 0; i < 8; i++) {
            a[i] = data[idx * 8 + i];
            b[i] = a[i] * 2.0f;
            c[i] = b[i] + a[i];
            data[idx * 8 + i] = c[i];
        }
    }
}
```

## 3. Using Constant Memory for Read-Only Data

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Constant memory (常量内存)</span> is 64KB of cached read-only memory. <span style="color:#2980B9;font-weight:700">Best for data that is read uniformly by all threads</span> - when all threads in a warp access the same address, it's broadcast in one cycle. <span style="color:#C0392B;font-weight:700">Different addresses cause serialization</span>. </div>

### 1) Constant Memory Applications

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Constant memory</span> is ideal for lookup tables, filter coefficients, and kernel parameters. <span style="color:#2980B9;font-weight:700">Declare with __constant__</span> and copy using <span style="color:#2980B9;font-weight:700">cudaMemcpyToSymbol</span>. </div>

```cpp
// Constant memory declaration (global scope)
__constant__ float filterCoeff[256];
__constant__ int lookupTable[1024];
__constant__ float params[8];

// Kernel using constant memory
__global__ void convolutionConstant(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        float sum = 0.0f;
        
        // All threads in warp access same coefficient addresses - fast!
        for (int k = 0; k < 5; k++) {
            if (idx + k < n) {
                sum += input[idx + k] * filterCoeff[k];
            }
        }
        
        output[idx] = sum;
    }
}

// Host code to set constant memory
int main() {
    // Initialize host arrays
    float h_filter[256];
    for (int i = 0; i < 256; i++) {
        h_filter[i] = i * 0.01f;
    }
    
    // Copy to constant memory
    cudaMemcpyToSymbol(filterCoeff, h_filter, sizeof(float) * 256);
    
    // Now all kernels can use filterCoeff
    convolutionConstant<<<grid, block>>>(d_input, d_output, n);
    
    return 0;
}

// Multiple parameters in constant memory
struct KernelParams {
    float alpha;
    float beta;
    int mode;
    int pad;
};

__constant__ KernelParams d_params;

__global__ void kernelWithParams(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Access parameters from constant memory
        if (d_params.mode == 0) {
            data[idx] = data[idx] * d_params.alpha;
        } else {
            data[idx] = data[idx] * d_params.beta;
        }
    }
}

// Set parameters
KernelParams params = {2.0f, 3.0f, 0, 0};
cudaMemcpyToSymbol(d_params, &params, sizeof(KernelParams));
```

## 4. Using Pinned Memory for Faster Host-Device Transfers

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Pinned memory (页锁定内存)</span> is host memory that cannot be paged to disk. <span style="color:#2980B9;font-weight:700">It enables faster host-device transfers (2x-4x speedup)</span> and is required for asynchronous operations. <span style="color:#C0392B;font-weight:700">Allocate with cudaHostAlloc() instead of malloc()</span>. </div>

### 1) Pinned Memory Benefits

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Pinned memory</span> allows the GPU to directly access host memory via DMA without CPU intervention. <span style="color:#2980B9;font-weight:700">Use for all data that will be transferred frequently</span>, especially with streams for overlapping. </div>

```cpp
// BAD: Pageable memory (malloc) - slower transfers
void badMemoryTransfer() {
    float* h_data = (float*)malloc(N * sizeof(float));
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Slower transfer - GPU must copy through pinned staging buffer
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    free(h_data);
    cudaFree(d_data);
}

// GOOD: Pinned memory - faster transfers
void goodMemoryTransfer() {
    float* h_data;
    float* d_data;
    
    // Allocate pinned memory
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Faster direct transfer
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaFreeHost(h_data);  // Special free for pinned memory
    cudaFree(d_data);
}

// Pinned memory with streams for overlapping
void pinnedWithStreams() {
    const int N_STREAMS = 4;
    const int CHUNK_SIZE = N / N_STREAMS;
    
    float *h_data;
    float *d_data;
    
    // Allocate pinned memory
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocDefault);
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Create streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Launch overlapping transfers and kernels
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * CHUNK_SIZE;
        
        // Asynchronous copy requires pinned memory
        cudaMemcpyAsync(d_data + offset, h_data + offset,
                       CHUNK_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice, streams[i]);
        
        kernel<<<grid, block, 0, streams[i]>>>(d_data + offset, CHUNK_SIZE);
        
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                       CHUNK_SIZE * sizeof(float),
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_data);
    cudaFree(d_data);
}

// Zero-copy memory (mapped pinned memory)
void zeroCopyExample() {
    float* h_data;
    
    // Allocate mapped pinned memory
    cudaHostAlloc(&h_data, N * sizeof(float), 
                  cudaHostAllocMapped);  // Mapped flag
    
    float* d_data;
    // Get device pointer to same memory (zero-copy)
    cudaHostGetDevicePointer(&d_data, h_data, 0);
    
    // Now GPU can directly access host memory
    kernel<<<grid, block>>>(d_data, N);  // No explicit copy needed!
    
    cudaDeviceSynchronize();
    cudaFreeHost(h_data);  // Data already on host
}
```

## Optimization Strategies Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Strategy</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Memory Type</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Speed Gain</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Best Use Case</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Shared Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">On-chip</td>
    <td style="padding: 10px; border: 1px solid #ddd;">~100x vs global</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Data reuse within blocks</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Registers</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Register file</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Fastest (1 cycle)</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Per-thread private data</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Constant Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Cached DRAM</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Broadcast speed</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Uniform read-only data</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#C0392B; font-weight:700">Pinned Memory</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Host DRAM</td>
    <td style="padding: 10px; border: 1px solid #ddd;">2-4x faster transfers</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Frequent host-device transfers</td>
  </tr>
</table>

## Combined Optimization Example

```cpp
// Comprehensive example using all optimization strategies
#define TILE_SIZE 16
#define N_FILTER 5

// Constant memory for filter coefficients
__constant__ float d_filter[N_FILTER];

// Kernel using shared memory, registers, and constant memory
__global__ void optimizedFilter(float* input, float* output, int width, int height) {
    // Shared memory for input tile with halo
    __shared__ float tile[TILE_SIZE][TILE_SIZE + N_FILTER - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;
    
    // Load data into shared memory (including halo)
    if (row < height && col < width) {
        tile[ty][tx + N_FILTER/2] = input[row * width + col];
    }
    
    // Load halo cells
    if (tx < N_FILTER/2) {
        if (row < height && col - N_FILTER/2 >= 0) {
            tile[ty][tx] = input[row * width + col - N_FILTER/2];
        }
        if (row < height && col + TILE_SIZE < width) {
            tile[ty][tx + TILE_SIZE + N_FILTER/2] = 
                input[row * width + col + TILE_SIZE];
        }
    }
    
    __syncthreads();
    
    // Compute using registers and constant memory
    if (row < height && col < width) {
        // Load into registers
        float sum = 0.0f;
        
        // Use constant memory filter (broadcast efficient)
        for (int k = 0; k < N_FILTER; k++) {
            sum += tile[ty][tx + k] * d_filter[k];  // Fast constant access
        }
        
        // Store result
        output[row * width + col] = sum;
    }
}

// Host code with pinned memory
int main() {
    int width = 1920, height = 1080;
    size_t bytes = width * height * sizeof(float);
    
    // Initialize filter on host
    float h_filter[N_FILTER] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};
    
    // Copy to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(float) * N_FILTER);
    
    // Allocate pinned host memory
    float *h_input, *h_output;
    cudaHostAlloc(&h_input, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_output, bytes, cudaHostAllocDefault);
    
    // Initialize input...
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy to device (fast with pinned memory)
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with shared memory optimization
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE,
                 (height + TILE_SIZE - 1) / TILE_SIZE);
    
    optimizedFilter<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    
    // Copy back
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    
    return 0;
}
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Use shared memory for block-level data reuse, registers for per-thread temporaries, constant memory for uniform read-only data</span>, and <span style="color:#2980B9;font-weight:700">pinned memory for all host-device transfers</span> to achieve near-peak GPU performance. </div>
