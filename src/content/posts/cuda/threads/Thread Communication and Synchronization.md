---
title: "Thread Communication and Synchronization"
published: 2026-03-15
description: "Thread Communication and Synchronization"
image: ""
tags: ["cuda","threads","Thread Communication and Synchronization"]
category: cuda / threads
draft: false
lang: ""
---

# I. Thread Communication and Synchronization

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Thread communication and synchronization</span> enable cooperation between threads in CUDA. <span style="color:#2980B9;font-weight:700">Block-level synchronization with __syncthreads()</span> ensures data consistency in shared memory. <span style="color:#E8600A;font-weight:700">Atomic operations</span> provide safe concurrent updates to shared data. <span style="color:#C0392B;font-weight:700">Avoiding warp divergence</span> maintains SIMT efficiency. </div>

## 1. Block-Level Synchronization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">__syncthreads()</span> is a barrier synchronization primitive that ensures all threads in a block have reached the same point before any thread proceeds. It's essential for <span style="color:#2980B9;font-weight:700">coordinating shared memory access</span> and preventing race conditions. </div>

### 1) __syncthreads() Usage

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">__syncthreads()</span> creates a barrier where all threads must wait. <span style="color:#2980B9;font-weight:700">Use it between shared memory writes and reads</span> to ensure data is fully written before consumption. <span style="color:#C0392B;font-weight:700">Never put __syncthreads() in conditional code</span> that varies across threads. </div>

```cpp
// Matrix transpose using shared memory with synchronization
__global__ void transposeSync(float* input, float* output, int width) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        // Load data into shared memory
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    // Barrier: ensure all threads have loaded data
    __syncthreads();
    
    // Now safe to read from shared memory written by other threads
    x = blockIdx.y * 32 + threadIdx.x;  // Transposed coordinates
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        // Write transposed data
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">__syncthreads() in conditional code can cause deadlock</span> because threads not reaching the barrier will never release waiting threads. Always ensure all threads in the block execute the same __syncthreads(). </div>

## 2. Atomic Operations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Atomic operations</span> perform read-modify-write sequences without interference from other threads. They're essential for <span style="color:#2980B9;font-weight:700">safe concurrent updates to shared variables</span> in global or shared memory. </div>

### 1) Basic Atomic Operations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">atomicAdd()</span> adds a value atomically. <span style="color:#E8600A;font-weight:700">atomicCAS()</span> (Compare And Swap) enables custom atomic operations. <span style="color:#2980B9;font-weight:700">Use atomics for counters, histograms, and reductions</span> where multiple threads update the same location. </div>

```cpp
// Histogram computation using atomic operations
__global__ void histogramAtomic(int* data, int* hist, int n, int numBins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % numBins;
        // Safely increment histogram bin
        atomicAdd(&hist[bin], 1);
    }
}

// Custom atomic max operation using atomicCAS
__global__ void atomicMaxCustom(int* data, int* maxVal, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        int oldVal = *maxVal;
        int newVal = max(oldVal, data[idx]);
        
        // Keep trying until we successfully update
        while (oldVal < data[idx]) {
            if (atomicCAS(maxVal, oldVal, newVal) == oldVal) {
                break;  // Success
            }
            oldVal = *maxVal;  // Re-read current value
            newVal = max(oldVal, data[idx]);
        }
    }
}

// Shared memory atomics for block-level reduction
__global__ void blockReduceAtomic(float* input, float* output, int n) {
    __shared__ float sharedMax;
    
    if (threadIdx.x == 0) {
        sharedMax = 0.0f;
    }
    __syncthreads();
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Atomic update in shared memory (fast!)
        atomicMax((int*)&sharedMax, __float_as_int(input[idx]));
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicMax((int*)&output[blockIdx.x], __float_as_int(sharedMax));
    }
}
```

### 2) Atomic Operation Performance Considerations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Atomic operations</span> can become bottlenecks when many threads contend for the same address. <span style="color:#2980B9;font-weight:700">Use privatization techniques</span> (per-thread or per-block accumulators) to reduce contention, then combine results. </div>

```cpp
// Efficient reduction using privatization to avoid atomic contention
__global__ void histogramEfficient(int* data, int* hist, int n, int numBins) {
    // Per-block private histogram in shared memory
    __shared__ int privateHist[256];  // Assume numBins <= 256
    
    // Initialize private histogram
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        privateHist[i] = 0;
    }
    __syncthreads();
    
    // Process data, updating private histogram
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % numBins;
        atomicAdd(&privateHist[bin], 1);  // Fast: shared memory atomic
    }
    __syncthreads();
    
    // Merge private histogram to global
    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        if (privateHist[i] > 0) {
            atomicAdd(&hist[i], privateHist[i]);  // One atomic per bin per block
        }
    }
}
```

## 3. Warp Divergence Avoidance

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Warp divergence</span> occurs when threads in the same warp take different control flow paths. <span style="color:#C0392B;font-weight:700">All paths execute serially</span>, wasting execution resources. <span style="color:#2980B9;font-weight:700">Restructure code to make branch conditions uniform within warps</span>. </div>

### 1) Identifying and Fixing Divergence

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Divergence patterns</span> often come from threadIdx-dependent conditions. <span style="color:#2980B9;font-weight:700">Use warp-level operations</span> or restructure data to keep warps together. <span style="color:#2980B9;font-weight:700">Predication</span> can sometimes replace branching. </div>

```cpp
// BAD: Divergent warp
__global__ void divergentKernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // This causes divergence within each warp
        if (threadIdx.x < 16) {
            data[idx] = processFirstHalf(data[idx]);
        } else {
            data[idx] = processSecondHalf(data[idx]);
        }
        // Warp executes both paths sequentially!
    }
}

// GOOD: Divergence avoided by block-level condition
__global__ void nonDivergentKernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Condition based on blockIdx - uniform across warp
        if (blockIdx.x % 2 == 0) {
            data[idx] = processEvenBlocks(data[idx]);
        } else {
            data[idx] = processOddBlocks(data[idx]);
        }
        // No divergence within warp - all threads take same path
    }
}

// GOOD: Using predication instead of branching
__global__ void predicatedKernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // Predicated execution - both expressions computed,
        // but results masked based on condition
        float val = data[idx];
        float result1 = processFirstHalf(val);
        float result2 = processSecondHalf(val);
        
        // Select based on condition without branching
        data[idx] = (threadIdx.x < 16) ? result1 : result2;
    }
}

// GOOD: Warp-level uniform processing
__global__ void warpUniformKernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    if (idx < n) {
        // Process differently per warp, but uniform within warp
        switch (warpId % 3) {
            case 0:  // Entire warp 0,3,6... takes same path
                data[idx] = processTypeA(data[idx]);
                break;
            case 1:  // Entire warp 1,4,7... takes same path
                data[idx] = processTypeB(data[idx]);
                break;
            case 2:  // Entire warp 2,5,8... takes same path
                data[idx] = processTypeC(data[idx]);
                break;
        }
    }
}
```

### 2) Advanced Divergence Avoidance Techniques

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Warp shuffle instructions</span> enable communication without shared memory. <span style="color:#E8600A;font-weight:700">Cooperative groups</span> provide modern APIs for warp-level programming. <span style="color:#2980B9;font-weight:700">Use these to avoid divergence in reduction operations</span>. </div>

```cpp
// Divergent reduction (BAD)
__global__ void divergentReduce(float* input, float* output, int n) {
    __shared__ float shared[256];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared[threadIdx.x] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Divergent loop - threads take different paths
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {  // Divergent!
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) output[blockIdx.x] = shared[0];
}

// Non-divergent reduction using warp shuffles (GOOD)
__global__ void nonDivergentReduce(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data
    float sum = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle (no divergence)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Each warp's result is in lane 0
    __shared__ float warpSums[32];  // Max 32 warps per block
    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    if (laneId == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();
    
    // Final reduction (only one warp active)
    if (warpId == 0) {
        float blockSum = (threadIdx.x < (blockDim.x / 32)) ? 
                         warpSums[laneId] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockSum += __shfl_down_sync(0xFFFFFFFF, blockSum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(output, blockSum);
        }
    }
}
```

## Synchronization and Atomic Operations Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Method</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Scope</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Use Case</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Performance Impact</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">__syncthreads()</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Block</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Shared memory consistency</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Medium (all threads wait)</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">Global Memory Atomics</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Grid</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Counters, histograms</td>
    <td style="padding: 10px; border: 1px solid #ddd;">High (global contention)</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">Shared Memory Atomics</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Block</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Block-level aggregation</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Low (fast on-chip)</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#C0392B; font-weight:700">Warp Shuffle</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Warp</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Fast reductions, broadcasts</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Lowest (register-only)</td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Use __syncthreads() for shared memory consistency</span>, <span style="color:#2980B9;font-weight:700">atomic operations for safe concurrent updates</span>, and <span style="color:#C0392B;font-weight:700">restructure code to avoid warp divergence</span> by keeping branch conditions uniform within each warp. </div>
