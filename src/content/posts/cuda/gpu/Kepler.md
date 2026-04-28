---
title: "Kepler"
published: 2026-04-27
description: "Kepler"
image: ""
tags: ["cuda","gpu","Kepler"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:52:30.335.569021499Z"
---
# Kepler Architecture (开普勒架构)

The Kepler Architecture (开普勒架构, 2012) is NVIDIA's first architecture designed for energy efficiency (能效) at scale, introducing Warp Shuffle (线程束洗牌), Dynamic Parallelism (动态并行), and Hyper-Q (超级队列) — letting GPUs schedule themselves and host more concurrent work.

**Representative GPU Models**:

- GeForce GTX 680 / GTX 770 (GK104) — 8 SMX, 1536 CUDA Cores
- GeForce GTX 780 / GTX 780 Ti / TITAN (GK110) — 14–15 SMX, 2688–2880 CUDA Cores
- Tesla K20 / K20X / K40 / K80 (data center)
- Quadro K6000

**Architecture Diagram** — GK110 Example (15 SMX):

```
┌────────────────────────────────────────────────────────────────────┐
│                       GK110 GPU (Kepler)                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Host Interface (PCIe 3.0) + 32 Hyper-Q queues   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           GigaThread Engine (with Dynamic Parallelism)       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  GPC 0          GPC 1         ...                       GPC 4      │
│  ┌──────┐      ┌──────┐                                ┌──────┐    │
│  │SMX 0 │      │SMX 3 │                                │SMX 12│    │
│  │SMX 1 │      │SMX 4 │                                │SMX 13│    │
│  │SMX 2 │      │SMX 5 │             ...                │SMX 14│    │
│  │ 192  │      │ 192  │                                │ 192  │    │
│  │ SP   │      │ SP   │                                │ SP   │    │
│  │ each │      │ each │                                │ each │    │
│  │ 64KB │      │ 64KB │                                │ 64KB │    │
│  │L1+Sh │      │L1+Sh │                                │L1+Sh │    │
│  └──────┘      └──────┘                                └──────┘    │
│       │             │                                       │      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Unified L2 Cache (1.5 MB)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             Global Memory (GDDR5, 384-bit, ECC)              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘

Total: 15 SMX × 192 SP = 2880 CUDA Cores
Each SMX: 192 SP + 64 DP units + 32 SFU + 32 LD/ST units
```

**Solved**: Fermi was power-limited (功耗受限) and had only one CPU-to-GPU work queue — Kepler greatly increased core density per watt, added register-to-register communication (`__shfl`), let kernels launch kernels (Dynamic Parallelism), and exposed 32 hardware work queues (Hyper-Q).

**Best Suited For**: Recursive / nested algorithms (递归 / 嵌套算法), warp-cooperative reductions (线程束协作归约), and multi-process MPI / multi-stream workloads sharing one GPU.

<br>

## 1. Warp-level Primitives (线程束级原语)

Warp-level Primitives (线程束级原语) are intrinsics that exchange data directly between the 32 threads of a Warp (线程束) using on-chip register paths, bypassing Shared Memory entirely.

**Solved**: Pre-Kepler intra-warp data exchange required Shared Memory (共享内存) round-trips with `__syncthreads()` — slow and wastes shared memory.

### 1) `__shfl` (洗牌指令)

`__shfl` (and variants `__shfl_up`, `__shfl_down`, `__shfl_xor`) lets a thread read another thread's register value within the same warp in one instruction.

**Old**: Shared memory + `__syncthreads()` for warp-internal communication.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpReduceShared(const int* in, int* out) {
    __shared__ int sdata[32];
    int tid = threadIdx.x;
    sdata[tid] = in[tid];
    __syncthreads();

    // Tree reduction inside one warp via shared memory
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpReduceShared<<<1, 32>>>(d, o);

    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Warp sum (shared memory): %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Warp sum (shared memory): 528
*/
```

**New**: `__shfl_down_sync` performs the reduction with no shared memory and no `__syncthreads()`.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warpReduceShfl(const int* in, int* out) {
    int tid = threadIdx.x;
    int v = in[tid];

    // Register-to-register reduction inside the warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }
    if (tid == 0) *out = v;
}

int main() {
    int h[32];
    for (int i = 0; i < 32; ++i) h[i] = i + 1;   // sum = 528

    int *d, *o;
    cudaMalloc(&d, 32*sizeof(int));
    cudaMalloc(&o, sizeof(int));
    cudaMemcpy(d, h, 32*sizeof(int), cudaMemcpyHostToDevice);

    warpReduceShfl<<<1, 32>>>(d, o);

    int r;
    cudaMemcpy(&r, o, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Warp sum (shfl): %d\n", r);

    cudaFree(d); cudaFree(o);
    return 0;
}

/* Expected Output:
Warp sum (shfl): 528
*/
```

<br>

## 2. Dynamic Parallelism (动态并行)

Dynamic Parallelism (动态并行) lets a GPU kernel launch other kernels directly from the device (设备端), without returning to the host.

**Solved**: Recursive / data-dependent algorithms (数据依赖算法) such as adaptive mesh refinement (自适应网格细化) previously required round-trips to the CPU between launches — Kepler removes that bottleneck.

**Old**: Host re-launches a child kernel per element after inspecting parent results.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void parent(const int* counts, int* out, int parentIdx) {
    int i = threadIdx.x;
    if (i < counts[parentIdx]) out[i] = parentIdx * 10 + i;
}

int main() {
    const int N = 3;
    int hCounts[N] = {2, 3, 4};
    int *dCounts;
    cudaMalloc(&dCounts, N*sizeof(int));
    cudaMemcpy(dCounts, hCounts, N*sizeof(int), cudaMemcpyHostToDevice);

    // Host loops: one child launch per parent index
    for (int p = 0; p < N; ++p) {
        int *dOut;
        cudaMalloc(&dOut, hCounts[p]*sizeof(int));
        parent<<<1, hCounts[p]>>>(dCounts, dOut, p);

        int hOut[8] = {0};
        cudaMemcpy(hOut, dOut, hCounts[p]*sizeof(int), cudaMemcpyDeviceToHost);
        printf("parent %d: ", p);
        for (int i = 0; i < hCounts[p]; ++i) printf("%d ", hOut[i]);
        printf("\n");
        cudaFree(dOut);
    }
    cudaFree(dCounts);
    return 0;
}

/* Expected Output:
parent 0: 0 1
parent 1: 10 11 12
parent 2: 20 21 22 23
*/
```

**New**: A device-side kernel launches its own child kernels — no host involvement.

```cpp
// Compile: nvcc -arch=sm_35 -rdc=true file.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void child(int parentIdx, int* out) {
    int i = threadIdx.x;
    out[i] = parentIdx * 10 + i;
}

__global__ void parent(const int* counts, int** outs) {
    int p = threadIdx.x;
    // Launch child kernel from the device
    child<<<1, counts[p]>>>(p, outs[p]);
}

int main() {
    const int N = 3;
    int hCounts[N] = {2, 3, 4};
    int *dCounts;
    cudaMalloc(&dCounts, N*sizeof(int));
    cudaMemcpy(dCounts, hCounts, N*sizeof(int), cudaMemcpyHostToDevice);

    int* hOutPtrs[N];
    for (int i = 0; i < N; ++i) cudaMalloc(&hOutPtrs[i], hCounts[i]*sizeof(int));
    int** dOutPtrs;
    cudaMalloc(&dOutPtrs, N*sizeof(int*));
    cudaMemcpy(dOutPtrs, hOutPtrs, N*sizeof(int*), cudaMemcpyHostToDevice);

    parent<<<1, N>>>(dCounts, dOutPtrs);
    cudaDeviceSynchronize();

    for (int p = 0; p < N; ++p) {
        int hOut[8] = {0};
        cudaMemcpy(hOut, hOutPtrs[p], hCounts[p]*sizeof(int), cudaMemcpyDeviceToHost);
        printf("parent %d: ", p);
        for (int i = 0; i < hCounts[p]; ++i) printf("%d ", hOut[i]);
        printf("\n");
        cudaFree(hOutPtrs[p]);
    }
    cudaFree(dCounts); cudaFree(dOutPtrs);
    return 0;
}

/* Expected Output:
parent 0: 0 1
parent 1: 10 11 12
parent 2: 20 21 22 23
*/
```

<br>

## 3. Hyper-Q (超级队列)

Hyper-Q (超级队列) provides 32 independent hardware work queues (硬件工作队列) on the GPU, so multiple host threads, MPI ranks, or CUDA streams can submit work concurrently without serializing in a single front-end queue.

**Solved**: Pre-Kepler GPUs had only one hardware work queue (一个硬件队列) — launches from independent streams collapsed into a single queue, killing concurrency between MPI ranks (MPI 进程) sharing one GPU.

**Old**: Single default stream forces serial execution across independent contexts.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernelA(int* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 1;
}
__global__ void kernelB(int* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 2;
}

int main() {
    const int N = 1024;
    int *dA, *dB;
    cudaMalloc(&dA, N*sizeof(int));
    cudaMalloc(&dB, N*sizeof(int));

    // Both on default stream — strictly serial, even though independent
    kernelA<<<4, 256>>>(dA, N);
    kernelB<<<4, 256>>>(dB, N);
    cudaDeviceSynchronize();

    int hA, hB;
    cudaMemcpy(&hA, dA, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hB, dB, sizeof(int), cudaMemcpyDeviceToHost);
    printf("A=%d B=%d (executed serially)\n", hA, hB);

    cudaFree(dA); cudaFree(dB);
    return 0;
}

/* Expected Output:
A=1 B=2 (executed serially)
*/
```

**New**: Multiple non-default streams map to independent Hyper-Q hardware queues — true concurrency.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernelA(int* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 1;
}
__global__ void kernelB(int* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = 2;
}

int main() {
    const int N = 1024;
    int *dA, *dB;
    cudaMalloc(&dA, N*sizeof(int));
    cudaMalloc(&dB, N*sizeof(int));

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Independent Hyper-Q queues — concurrent on Kepler+
    kernelA<<<4, 256, 0, s1>>>(dA, N);
    kernelB<<<4, 256, 0, s2>>>(dB, N);
    cudaDeviceSynchronize();

    int hA, hB;
    cudaMemcpy(&hA, dA, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hB, dB, sizeof(int), cudaMemcpyDeviceToHost);
    printf("A=%d B=%d (concurrent via Hyper-Q)\n", hA, hB);

    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(dA); cudaFree(dB);
    return 0;
}

/* Expected Output:
A=1 B=2 (concurrent via Hyper-Q)
*/
```

<br>
<br>
