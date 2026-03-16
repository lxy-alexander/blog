---
title: "L1 and L2 Cache in CUDA"
published: 2026-03-15
description: "L1 and L2 Cache in CUDA"
image: ""
tags: ["cuda","memory","L1 and L2 Cache in CUDA"]
category: cuda / memory
draft: false
lang: ""
---

# I. L1 and L2 Cache in CUDA

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">L1 and L2 caches (高速缓存)</span> are on-chip memories that reduce latency to global memory. L1 cache is <span style="color:#2980B9;font-weight:700">private to each Streaming Multiprocessor (SM)</span>, while L2 cache is <span style="color:#2980B9;font-weight:700">shared by all SMs</span>. Understanding their relationship with other memories and how to control them is essential for performance tuning. </div>

## 1. Cache Hierarchy and Relationships

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">L1 cache</span> is SM-private (~192KB on H100), <span style="color:#E8600A;font-weight:700">L2 cache</span> is shared by all SMs (~50MB on H100). <span style="color:#2980B9;font-weight:700">Register ~1 cycle / per-thread, L1 ~20-30 cycles / per-SM, L2 ~100 cycles / all SMs, DRAM ~500 cycles / off-chip</span>. L1 physically shares the same on-chip SRAM as <span style="color:#E8600A;font-weight:700">shared memory (共享内存)</span> on pre-Ampere — more shared memory means less L1. <span style="color:#C0392B;font-weight:700">On Ampere and later, L1 and shared memory are independent.</span> L2 sits between L1 and <span style="color:#E8600A;font-weight:700">global memory (全局内存, HBM)</span> — all L1 misses go to L2 before hitting DRAM. <span style="color:#2980B9;font-weight:700">Constant memory (常量内存) and texture memory (纹理内存) have their own read-only cache paths that bypass L1/L2.</span> </div>

### 1) Cache Position in Memory Hierarchy

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">L1 cache</span> is SM-private, <span style="color:#E8600A;font-weight:700">L2 cache</span> is shared by all SMs. <span style="color:#2980B9;font-weight:700">Register ~1 cycle, L1 ~20-30 cycles, L2 ~100 cycles, DRAM ~500 cycles</span>. <span style="color:#C0392B;font-weight:700">L1 latency is often confused with register latency — they differ by ~20x.</span> </div>

```cpp
// Memory hierarchy visualization in code
__global__ void demonstrateHierarchy(float* global_mem) {
    // 1. Global memory access (slowest) - goes through L2 → L1 → register
    float val = global_mem[threadIdx.x];
    
    // 2. Register access (fastest) - already in register
    float reg_val = val * 2.0f;
    
    // 3. Shared memory (closely related to L1 - shares same hardware)
    __shared__ float shared[256];
    shared[threadIdx.x] = reg_val;
    
    __syncthreads();
    
    // 4. L1 cache would automatically cache frequently accessed global data
    global_mem[threadIdx.x] = shared[threadIdx.x];
}
```

### 2) Relationship with Shared Memory

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> On many architectures (especially <span style="color:#E8600A;font-weight:700">Ampere and later</span>), <span style="color:#C0392B;font-weight:700">L1 cache and shared memory share the same on-chip storage</span>. This creates a trade-off: more L1 means less shared memory, and vice versa . </div>

```cpp
// Shared memory and L1 share hardware resources
__global__ void sharedVsL1Kernel(float* data) {
    // This shared memory allocation reduces available L1 cache
    __shared__ float large_tile[256][256];  // 256KB shared memory
    
    // L1 cache will be smaller because shared memory uses the space
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // This global load may have less L1 cache available
    float val = data[idx];
    
    // Process using shared memory
    large_tile[threadIdx.y][threadIdx.x] = val;
    __syncthreads();
    
    data[idx] = large_tile[threadIdx.y][threadIdx.x] * 2.0f;
}
```

## 2. Cache Configuration and Control

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> CUDA provides several ways to control cache behavior: <span style="color:#2980B9;font-weight:700">compiler flags for enabling/disabling L1</span>, <span style="color:#2980B9;font-weight:700">API functions for L1/shared memory partitioning</span>, and <span style="color:#2980B9;font-weight:700">PTX instructions for per-access control</span>. </div>

### 1) Enabling/Disabling L1 Cache Globally

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Compiler flags</span> control L1 cache usage for all global memory loads. <span style="color:#2980B9;font-weight:700">-Xptxas -dlcm=cg</span> disables L1 (cache global), <span style="color:#2980B9;font-weight:700">-Xptxas -dlcm=ca</span> enables L1 (cache all) . </div>

```bash
# Compiler flags to control L1 cache
# Disable L1 cache for global loads (use L2 only)
nvcc -Xptxas -dlcm=cg mykernel.cu -o mykernel

# Enable L1 cache for global loads (default behavior)
nvcc -Xptxas -dlcm=ca mykernel.cu -o mykernel
```

```cpp
// Impact of cache settings on memory transactions
__global__ void cacheAwareKernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // With -dlcm=ca: L1 cached load - 128-byte transactions
        // With -dlcm=cg: L2 only - 32-byte transactions
        float val = data[idx];
        
        // For scattered access patterns, disabling L1 can be better
        // because 32-byte segments waste less bandwidth 
        data[idx] = val * 2.0f;
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> When L1 is enabled, memory transactions are <span style="color:#C0392B;font-weight:700">128 bytes (L1 cache line size)</span>. When disabled, transactions are <span style="color:#2980B9;font-weight:700">32-byte segments</span>. For uncoalesced or misaligned access, disabling L1 can actually improve performance by reducing wasted bandwidth . </div>

### 2) L1/Shared Memory Partitioning

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">cudaDeviceSetCacheConfig()</span> and <span style="color:#E8600A;font-weight:700">cudaFuncSetCacheConfig()</span> control the split between L1 and shared memory. <span style="color:#2980B9;font-weight:700">Use PreferL1 for more L1 cache, PreferShared for more shared memory</span> . </div>

```cpp
// Controlling L1/shared memory partition
#include <cuda_runtime.h>

// Set device-wide cache configuration
void setDeviceCacheConfig() {
    // Prefer larger L1 cache (smaller shared memory)
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    // Prefer larger shared memory (smaller L1 cache)
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    
    // Default - no preference
    cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    
    // Some devices support equal split (PreferEqual) 
}

// Per-kernel cache configuration
__global__ void l1IntensiveKernel(float* data) {
    // This kernel benefits from large L1 cache
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Repeated access to same data - L1 helps
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        sum += data[idx + i * blockDim.x * gridDim.x];
    }
    data[idx] = sum;
}

__global__ void sharedIntensiveKernel(float* data) {
    // This kernel benefits from large shared memory
    __shared__ float tile[256][256];  // Needs lots of shared memory
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load into shared memory
    tile[ty][tx] = data[ty * blockDim.x + tx];
    __syncthreads();
    
    // Compute using shared memory
    data[ty * blockDim.x + tx] = tile[tx][ty];  // Transpose
}

// Host code setting per-kernel cache config
int main() {
    // Set different cache config for different kernels
    cudaFuncSetCacheConfig(l1IntensiveKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(sharedIntensiveKernel, cudaFuncCachePreferShared);
    
    // Launch kernels - each uses its preferred config
    l1IntensiveKernel<<<grid, block>>>(d_data);
    sharedIntensiveKernel<<<grid2, block2>>>(d_data);
    
    return 0;
}
```

## 3. Fine-Grained Cache Control

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> For precise control, CUDA provides <span style="color:#E8600A;font-weight:700">intrinsic functions and PTX instructions</span> to control caching for individual loads. This allows mixing cached and uncached accesses in the same kernel . </div>

### 1) Per-Access Cache Control with Intrinsics

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">__ldcg()</span> forces a load to bypass L1 (cache global). <span style="color:#E8600A;font-weight:700">__ldca()</span> enables L1 caching (cache all). <span style="color:#2980B9;font-weight:700">Use these when different data have different access patterns</span> . </div>

```cpp
// Per-access cache control with intrinsics
__device__ __inline__ float ld_gbl_cg(const float* addr) {
    // PTX: ld.global.cg - bypass L1, cache in L2 only 
    float val;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

__device__ __inline__ float ld_gbl_ca(const float* addr) {
    // PTX: ld.global.ca - cache in both L1 and L2
    float val;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

__global__ void mixedCacheKernel(float* data1, float* data2, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // data1: accessed once - bypass L1 to avoid cache pollution
        float val1 = __ldcg(&data1[idx]);  // CUDA intrinsic (simpler than inline PTX)
        
        // data2: accessed repeatedly - use L1 caching
        float val2 = 0.0f;
        for (int i = 0; i < 5; i++) {
            val2 += __ldca(&data2[idx + i * n]);
        }
        
        data1[idx] = val1 + val2;
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> Modern CUDA provides built-in intrinsics like <span style="color:#2980B9;font-weight:700">__ldcg(), __ldca(), __ldcs()</span> in the CUDA C++ Programming Guide, which are easier than writing inline PTX. These map directly to the cache operators described in PTX ISA . </div>

### 2) When to Disable L1 Cache

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#2980B9;font-weight:700">Disable L1 cache when:</span> access patterns are random/scattered, data is accessed only once, or to prevent cache pollution. <span style="color:#2980B9;font-weight:700">Enable L1 when:</span> data has locality, is accessed repeatedly, or patterns are coalesced . </div>

```cpp
// Comparison: when to disable vs enable L1
__global__ void cacheDecisionKernel(float* random_access, float* sequential, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        // SCENARIO 1: Random access pattern - disable L1
        // Each thread accesses a random location
        // L1 would thrash and waste bandwidth
        int random_idx = (idx * 129) % n;  // Poor locality
        float rand_val = __ldcg(&random_access[random_idx]);
        
        // SCENARIO 2: Sequential access pattern - enable L1
        // Threads access consecutive addresses - great locality
        // L1 will cache subsequent accesses
        float seq_val = __ldca(&sequential[idx]);
        
        // SCENARIO 3: Mixed - use appropriate access for each
        random_access[random_idx] = seq_val;
        sequential[idx] = rand_val;
    }
}
```

## Cache Configuration Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Control Method</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Scope</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Effect</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Use Case</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">-Xptxas -dlcm=cg</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Entire compilation</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Disable L1 globally</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Random access patterns </td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">cudaFuncSetCacheConfig</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Per kernel</td>
    <td style="padding: 10px; border: 1px solid #ddd;">L1/shared split</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Kernels with different needs </td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">__ldcg() intrinsic</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Per load</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Bypass L1 for specific access</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Mixed access patterns </td>
  </tr>
</table>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">L1 cache is SM-private and shares hardware with shared memory</span>; <span style="color:#2980B9;font-weight:700">disable it globally with -dlcm=cg for random access, partition it with cache config APIs</span>, or <span style="color:#C0392B;font-weight:700">control per-load with __ldcg() intrinsics</span> – the right choice depends entirely on your access pattern and data reuse. </div>
