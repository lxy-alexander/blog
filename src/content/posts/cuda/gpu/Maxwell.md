---
title: "Maxwell"
published: 2026-04-27
description: "Maxwell"
image: ""
tags: ["cuda","gpu","Maxwell"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:54:24.259.903477228Z"
---
# Maxwell Architecture (麦克斯韦架构)

The Maxwell Architecture (麦克斯韦架构, 2014) is NVIDIA's "perf-per-watt" (每瓦性能) generation that redesigned the SM (流式多处理器) into smaller partitions, dedicated 96 KB of pure Shared Memory (共享内存) per SM, and added Native Shared Memory Atomics (原生共享内存原子操作) — making histograms (直方图) and reductions (归约) several times faster.

**Representative GPU Models**:

- GeForce GTX 750 / 750 Ti (GM107, "first Maxwell")
- GeForce GTX 970 / 980 / 980 Ti / TITAN X (GM204 / GM200) — 16–24 SMM, 2048–3072 CUDA Cores
- Tesla M40 / M4 (data center / inference)
- Quadro M6000
- Jetson TX1 (embedded, 嵌入式)

**Architecture Diagram** — GM200 Example (24 SMM):

```
┌────────────────────────────────────────────────────────────────────┐
│                       GM200 GPU (Maxwell)                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Host Interface (PCIe 3.0) + Hyper-Q             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      GigaThread Engine                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│   GPC 0          GPC 1         ...                       GPC 5     │
│  ┌───────┐      ┌───────┐                              ┌───────┐   │
│  │SMM 0  │      │SMM 4  │                              │SMM 20 │   │
│  │SMM 1  │      │SMM 5  │                              │SMM 21 │   │
│  │SMM 2  │      │SMM 6  │            ...               │SMM 22 │   │
│  │SMM 3  │      │SMM 7  │                              │SMM 23 │   │
│  │       │      │       │                              │       │   │
│  │ Each  │      │ Each  │                              │ Each  │   │
│  │ SMM:  │      │ SMM:  │                              │ SMM:  │   │
│  │ 4 x 32│      │ 4 x 32│                              │ 4 x 32│   │
│  │ = 128 │      │ = 128 │                              │ = 128 │   │
│  │ SP    │      │ SP    │                              │ SP    │   │
│  │ 96 KB │      │ 96 KB │                              │ 96 KB │   │
│  │ Shmem │      │ Shmem │                              │ Shmem │   │
│  └───────┘      └───────┘                              └───────┘   │
│       │              │                                      │      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Unified L2 Cache (3 MB)                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Global Memory (GDDR5, 384-bit)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘

Total: 24 SMM × 128 SP = 3072 CUDA Cores
Each SMM: split into 4 partitions of 32 SP, 96 KB dedicated Shared Memory
(L1 is now separate from Shared Memory, unlike Fermi/Kepler)
```

**Solved**: Kepler shared 64 KB between L1 and Shared Memory (capping shared to 48 KB) and routed shared-memory atomics through a slow lock/CAS path — Maxwell separates L1 from Shared Memory (giving full 96 KB of Shared Memory) and implements atomics in hardware directly inside Shared Memory.

**Best Suited For**: Histogram-heavy workloads (直方图密集型负载), per-block reductions (块内归约), image processing (图像处理), and power-constrained / embedded inference (受功耗约束的推理 / 嵌入式推理).

<br>

## 1. Native Shared Memory Atomics (原生共享内存原子操作)

Native Shared Memory Atomics (原生共享内存原子操作) execute directly inside the SM's Shared Memory hardware, with much lower latency (延迟) and higher throughput (吞吐量) than Kepler's lock-based emulation.

**Solved**: On Fermi/Kepler, `atomicAdd` on `__shared__` memory was emulated using a lock per bank — under heavy contention (高竞争), throughput collapsed.

### 1) 32-bit Integer Shared Memory Atomics (32 位整数共享内存原子操作)

Maxwell implements 32-bit integer atomics on Shared Memory at near-native speed, making per-block counters and bin updates (桶更新) extremely fast.

**Old**: Use Global Memory atomics for counters because shared-memory atomics are too slow on Kepler.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void countGlobal(const int* data, int N, int* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (data[i] % 2 == 0)) {
        atomicAdd(counter, 1);   // global memory atomic — slow under contention
    }
}

int main() {
    const int N = 10000;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i;   // 5000 even values

    int *d, *dCount;
    int hCount = 0;
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &hCount, sizeof(int), cudaMemcpyHostToDevice);

    countGlobal<<<(N+255)/256, 256>>>(d, N, dCount);
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Even count (global atomics): %d\n", hCount);

    free(h); cudaFree(d); cudaFree(dCount);
    return 0;
}

/* Expected Output:
Even count (global atomics): 5000
*/
```

**New**: Each block maintains a local counter in Shared Memory using fast native atomics, then merges once into Global Memory.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void countShared(const int* data, int N, int* counter) {
    __shared__ int sCount;
    if (threadIdx.x == 0) sCount = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && (data[i] % 2 == 0)) {
        atomicAdd(&sCount, 1);   // native shared-memory atomic on Maxwell
    }
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(counter, sCount);  // one global update per block
}

int main() {
    const int N = 10000;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i;   // 5000 even values

    int *d, *dCount;
    int hCount = 0;
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCount, &hCount, sizeof(int), cudaMemcpyHostToDevice);

    countShared<<<(N+255)/256, 256>>>(d, N, dCount);
    cudaMemcpy(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Even count (shared atomics + 1 global merge): %d\n", hCount);

    free(h); cudaFree(d); cudaFree(dCount);
    return 0;
}

/* Expected Output:
Even count (shared atomics + 1 global merge): 5000
*/
```

<br>

## 2. Histogram-friendly Atomic Operations (直方图友好的原子操作)

Histogram (直方图) computation has many threads updating a small set of bins (桶) — the canonical contention-heavy workload (高竞争负载) where Maxwell's fast Shared Memory atomics shine.

**Solved**: A pure global-memory histogram serializes updates to popular bins — extremely slow. Per-block private histograms in Shared Memory + native atomics scale much better.

**Old**: Every thread does `atomicAdd` directly on the global histogram — heavy contention.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

#define BINS 16

__global__ void histGlobal(const int* data, int N, int* hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = data[i] % BINS;
        atomicAdd(&hist[bin], 1);   // global atomic — heavy contention
    }
}

int main() {
    const int N = 4096;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i;   // each bin gets N/BINS = 256

    int *d, *dHist;
    int hHist[BINS] = {0};
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dHist, BINS*sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dHist, hHist, BINS*sizeof(int), cudaMemcpyHostToDevice);

    histGlobal<<<(N+255)/256, 256>>>(d, N, dHist);
    cudaMemcpy(hHist, dHist, BINS*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram (global atomics): ");
    for (int b = 0; b < BINS; ++b) printf("%d ", hHist[b]);
    printf("\n");

    free(h); cudaFree(d); cudaFree(dHist);
    return 0;
}

/* Expected Output:
Histogram (global atomics): 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256
*/
```

**New**: Per-block private histogram in Shared Memory using native atomics, then merge to global once.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

#define BINS 16

__global__ void histShared(const int* data, int N, int* hist) {
    __shared__ int sHist[BINS];
    int tid = threadIdx.x;

    // Initialize private histogram
    if (tid < BINS) sHist[tid] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) {
        int bin = data[i] % BINS;
        atomicAdd(&sHist[bin], 1);   // fast native shared-memory atomic
    }
    __syncthreads();

    // Merge once into global histogram
    if (tid < BINS) atomicAdd(&hist[tid], sHist[tid]);
}

int main() {
    const int N = 4096;
    int* h = (int*)malloc(N*sizeof(int));
    for (int i = 0; i < N; ++i) h[i] = i;   // each bin gets N/BINS = 256

    int *d, *dHist;
    int hHist[BINS] = {0};
    cudaMalloc(&d, N*sizeof(int));
    cudaMalloc(&dHist, BINS*sizeof(int));
    cudaMemcpy(d, h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dHist, hHist, BINS*sizeof(int), cudaMemcpyHostToDevice);

    histShared<<<(N+255)/256, 256>>>(d, N, dHist);
    cudaMemcpy(hHist, dHist, BINS*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Histogram (shared atomics): ");
    for (int b = 0; b < BINS; ++b) printf("%d ", hHist[b]);
    printf("\n");

    free(h); cudaFree(d); cudaFree(dHist);
    return 0;
}

/* Expected Output:
Histogram (shared atomics): 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256 256
*/
```

<br>
<br>