---
title: "Pascal"
published: 2026-04-27
description: "Pascal"
image: ""
tags: ["cuda","gpu","Pascal"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:54:34.191.530739267Z"
---
# Pascal Architecture (帕斯卡架构)

The Pascal Architecture (帕斯卡架构, 2016) introduced HBM2 (高带宽显存), NVLink (NV 高速互联), and a real Unified Memory (统一内存) implementation with hardware page-fault handling — letting GPU and CPU share a single virtual address space (虚拟地址空间) without explicit copies.

**Representative GPU Models**:

- GeForce GTX 1060 / 1070 / 1080 / 1080 Ti / TITAN X (GP104 / GP102) — 20–30 SM, 1920–3584 CUDA Cores
- Tesla P100 (GP100) — 56 SM, 3584 CUDA Cores, HBM2, NVLink (data center)
- Tesla P40 / P4 (inference)
- Quadro P6000
- Jetson TX2 (embedded)

**Architecture Diagram** — GP100 Example (56 SM):

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GP100 GPU (Pascal)                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │       Host Interface (PCIe 3.0) + 4× NVLink (160 GB/s)         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │     GigaThread Engine + Page Migration Engine (页迁移引擎)     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  GPC 0          GPC 1         ...                       GPC 5        │
│  ┌────────┐    ┌────────┐                              ┌────────┐    │
│  │ 10 SM  │    │ 10 SM  │                              │ 10 SM  │    │
│  │        │    │        │                              │        │    │
│  │ Each   │    │ Each   │            ...               │ Each   │    │
│  │ SM:    │    │ SM:    │                              │ SM:    │    │
│  │ 64 SP  │    │ 64 SP  │                              │ 64 SP  │    │
│  │ 32 DP  │    │ 32 DP  │                              │ 32 DP  │    │
│  │ 64 KB  │    │ 64 KB  │                              │ 64 KB  │    │
│  │ Shmem  │    │ Shmem  │                              │ Shmem  │    │
│  └────────┘    └────────┘                              └────────┘    │
│       │             │                                       │        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  Unified L2 Cache (4 MB)                       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │       HBM2 Memory (16 GB, 4096-bit, 720 GB/s, ECC)             │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

Total: 56 SM × 64 SP = 3584 CUDA Cores (GP100, full chip = 60 SM)
```

**Solved**: Maxwell required explicit `cudaMemcpy` for every host-device transfer and was bandwidth-limited by GDDR5 — Pascal added HBM2 (3× bandwidth), NVLink (5× faster than PCIe for GPU-GPU), and true Unified Memory with on-demand page migration (按需页迁移).

**Best Suited For**: Deep learning training (深度学习训练), HPC simulations exceeding GPU memory (超出显存的 HPC 仿真), and multi-GPU workloads (多 GPU 工作负载) connected via NVLink.

<br>

## 1. Unified Memory (统一内存)

Unified Memory (统一内存) is a single managed memory pool addressable by both CPU and GPU, where the runtime automatically migrates pages on access — eliminating the need for explicit copies.

**Solved**: Manual `cudaMalloc` + `cudaMemcpy` is error-prone (易错) and forces over-copying (过度拷贝) when only part of the data is touched.

### 1) Managed Memory (托管内存)

Managed Memory (托管内存) is allocated via `cudaMallocManaged` and is accessible by both host and device using the same pointer.

**Old**: Separate host and device allocations with explicit copies in both directions.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= k;
}

int main() {
    const int N = 8;
    float h[N] = {1,2,3,4,5,6,7,8};

    float* d;
    cudaMalloc(&d, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);  // explicit copy

    scale<<<1, N>>>(d, 2.0f, N);

    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);  // explicit copy
    for (int i = 0; i < N; ++i) printf("%g ", h[i]);
    printf("\n");

    cudaFree(d);
    return 0;
}

/* Expected Output:
2 4 6 8 10 12 14 16
*/
```

**New**: Single `cudaMallocManaged` pointer used by both sides.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= k;
}

int main() {
    const int N = 8;
    float* x;
    cudaMallocManaged(&x, N*sizeof(float));   // one allocation, both sides
    for (int i = 0; i < N; ++i) x[i] = i + 1; // host writes directly

    scale<<<1, N>>>(x, 2.0f, N);
    cudaDeviceSynchronize();                  // required before host read

    for (int i = 0; i < N; ++i) printf("%g ", x[i]);
    printf("\n");

    cudaFree(x);
    return 0;
}

/* Expected Output:
2 4 6 8 10 12 14 16
*/
```

### 2) Page Migration (页迁移)

Page Migration (页迁移) is the hardware-driven mechanism that moves a memory page (默认 4 KB / 64 KB) from one processor's physical memory to another's on first access — triggered by a Page Fault (缺页中断).

**Solved**: Pre-Pascal Unified Memory had to migrate the entire allocation eagerly (主动迁移) on kernel launch — wasteful when only part of the data was used.

**Old**: Allocate, migrate everything to GPU at launch, do a tiny computation — most bytes wasted.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void touchFirst(float* x) {
    if (threadIdx.x == 0) x[0] = 99.0f;   // touch only one element
}

int main() {
    const int N = 1 << 20;   // 4 MB
    float* x;
    cudaMallocManaged(&x, N*sizeof(float));
    for (int i = 0; i < N; ++i) x[i] = 0.0f;

    // Pre-Pascal: even though kernel touches only x[0], the runtime
    // had to migrate the whole 4 MB to the GPU.
    touchFirst<<<1, 32>>>(x);
    cudaDeviceSynchronize();

    printf("x[0]=%g, x[N-1]=%g (whole array migrated)\n", x[0], x[N-1]);

    cudaFree(x);
    return 0;
}

/* Expected Output:
x[0]=99, x[N-1]=0 (whole array migrated)
*/
```

**New**: Pascal migrates only the page containing `x[0]` on demand — cheap, lazy.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void touchFirst(float* x) {
    if (threadIdx.x == 0) x[0] = 99.0f;
}

int main() {
    const int N = 1 << 20;
    float* x;
    cudaMallocManaged(&x, N*sizeof(float));
    for (int i = 0; i < N; ++i) x[i] = 0.0f;

    // Pascal: only the page holding x[0] migrates on the first GPU fault.
    // The rest stays on the host until accessed.
    touchFirst<<<1, 32>>>(x);
    cudaDeviceSynchronize();

    printf("x[0]=%g, x[N-1]=%g (only one page migrated)\n", x[0], x[N-1]);

    cudaFree(x);
    return 0;
}

/* Expected Output:
x[0]=99, x[N-1]=0 (only one page migrated)
*/
```

### 3) Fine-grained Memory Migration (细粒度内存迁移)

Fine-grained Memory Migration (细粒度内存迁移) uses `cudaMemPrefetchAsync` and `cudaMemAdvise` to give the runtime hints about which pages should live where — letting the programmer drive migration intentionally without giving up the unified-pointer model.

**Solved**: Pure on-demand migration causes page-fault stalls (缺页停顿) on first access; prefetching warms the GPU's memory before the kernel runs.

**Old**: Rely on lazy faulting only — first kernel pays the migration cost mid-execution.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void sumAll(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(out, x[i]);
}

int main() {
    const int N = 1 << 20;
    float *x, *sum;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&sum, sizeof(float));
    for (int i = 0; i < N; ++i) x[i] = 1.0f;
    *sum = 0.0f;

    // No prefetch — kernel pays page-fault cost while running
    sumAll<<<(N+255)/256, 256>>>(x, sum, N);
    cudaDeviceSynchronize();

    printf("Sum = %g\n", *sum);

    cudaFree(x); cudaFree(sum);
    return 0;
}

/* Expected Output:
Sum = 1.04858e+06
*/
```

**New**: `cudaMemPrefetchAsync` warms the GPU memory before the kernel — no fault stalls.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void sumAll(const float* x, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(out, x[i]);
}

int main() {
    const int N = 1 << 20;
    float *x, *sum;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&sum, sizeof(float));
    for (int i = 0; i < N; ++i) x[i] = 1.0f;
    *sum = 0.0f;

    int dev;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(x, N*sizeof(float), dev);   // prefetch to GPU
    cudaMemAdvise(x, N*sizeof(float), cudaMemAdviseSetReadMostly, dev);

    sumAll<<<(N+255)/256, 256>>>(x, sum, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(sum, sizeof(float), cudaCpuDeviceId);  // back to CPU
    cudaDeviceSynchronize();

    printf("Sum = %g\n", *sum);

    cudaFree(x); cudaFree(sum);
    return 0;
}

/* Expected Output:
Sum = 1.04858e+06
*/
```

<br>
<br>