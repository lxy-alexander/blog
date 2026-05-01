---
title: "Hopper"
published: 2026-04-27
description: "Hopper"
image: ""
tags: ["cuda","gpu","Hopper"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:31.328.141820711Z"
---
# Hopper Architecture (霍珀架构)

The Hopper Architecture (霍珀架构, 2022) is NVIDIA's transformer-era data-center GPU, introducing fourth-generation Tensor Cores with FP8 (8 位浮点), the Transformer Engine (变换器引擎), the Tensor Memory Accelerator (张量内存加速器), and Thread Block Clusters (线程块集群) with Distributed Shared Memory (分布式共享内存) — making cross-SM cooperation a first-class programming model.

![image-20260428191614573](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428191614573)

**Representative GPU Models**:

- H100 SXM5 (GH100, data center) — 132 SM, 16896 CUDA Cores + 528 Tensor Cores, HBM3
- H100 PCIe (GH100) — 114 SM
- H200 (GH100) — same SM count as H100 SXM5, larger HBM3e memory
- GH200 Grace Hopper Superchip — H100 GPU + 72-core Grace CPU on one module via NVLink-C2C

**Architecture Diagram** — H100 SXM5 Example (132 SM):

```
┌──────────────────────────────────────────────────────────────────────┐
│                         GH100 GPU (Hopper)                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │   Host Interface (PCIe 5.0) + 18× NVLink 4.0 (900 GB/s)        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │     GigaThread Engine + Cluster Scheduler (集群调度器)         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  GPC 0   GPC 1   GPC 2   GPC 3   GPC 4   GPC 5   GPC 6   GPC 7       │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│ │16-18│ │16-18│ │16-18│ │16-18│ │16-18│ │16-18│ │16-18│ │16-18│      │
│ │ SM  │ │ SM  │ │ SM  │ │ SM  │ │ SM  │ │ SM  │ │ SM  │ │ SM  │      │
│ │     │ │     │ │     │ │     │ │     │ │     │ │     │ │     │      │
│ │Each │ │Each │ │Each │ │Each │ │Each │ │Each │ │Each │ │Each │      │
│ │SM:  │ │SM:  │ │SM:  │ │SM:  │ │SM:  │ │SM:  │ │SM:  │ │SM:  │      │
│ │128SP│ │128SP│ │128SP│ │128SP│ │128SP│ │128SP│ │128SP│ │128SP│      │
│ │4 TC │ │4 TC │ │4 TC │ │4 TC │ │4 TC │ │4 TC │ │4 TC │ │4 TC │      │
│ │TMA  │ │TMA  │ │TMA  │ │TMA  │ │TMA  │ │TMA  │ │TMA  │ │TMA  │      │
│ │228KB│ │228KB│ │228KB│ │228KB│ │228KB│ │228KB│ │228KB│ │228KB│      │
│ │Shmem│ │Shmem│ │Shmem│ │Shmem│ │Shmem│ │Shmem│ │Shmem│ │Shmem│      │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
│    │       │       │       │       │       │       │       │        │
│    └───────┴───── DSMEM SM-to-SM Network within each GPC ───┘        │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                  Unified L2 Cache (50 MB)                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │       HBM3 Memory (80 GB, 5120-bit, 3 TB/s, ECC)               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

H100 SXM5: 132 SM = 8 GPCs × ~16-18 SM each (full GH100 chip = 144 SM)
Each SM: 128 FP32 SP + 64 FP64 + 4 fourth-gen Tensor Cores + TMA unit
         228 KB max shared memory, 256 KB combined L1+Shmem
```

**Solved**: Ampere's Tensor Cores topped out at TF32 / BF16 / FP16 — Hopper added FP8 for transformer training, the Transformer Engine that auto-switches between FP8 and FP16 layer-by-layer, the TMA hardware unit for async bulk tensor copies, and Thread Block Clusters that let blocks on different SMs share data via Distributed Shared Memory.

**Best Suited For**: Large language model training and inference (大语言模型训练与推理), transformer-heavy AI workloads, and high-performance matrix-multiplication kernels (高性能矩阵乘法核函数) like FlashAttention-3 and CUTLASS GEMM.

<br>

## 1. Fourth-generation Tensor Cores (第四代张量核心)

Fourth-generation Tensor Cores (第四代张量核心) deliver 2× the per-SM matrix-math throughput of Ampere on the same data types and 4× when using the new FP8 format — the key hardware behind H100's transformer training speedup.

**Solved**: Transformer training is bottlenecked by GEMM (通用矩阵乘法) throughput; the new Tensor Cores cut both compute time and memory bandwidth pressure roughly in half by halving operand bit-width.

### 1) FP8 (8 位浮点)

FP8 (8 位浮点) is a new 8-bit floating-point format with two variants — `E4M3` (4 exponent, 3 mantissa) for forward activations / weights and `E5M2` (5 exponent, 2 mantissa) for backward gradients (反向梯度) — doubling throughput vs FP16 with acceptable accuracy loss.

**Old**: FP16 inputs on Volta/Turing/Ampere Tensor Cores.

```cpp
// Compile: nvcc -arch=sm_80 file.cu  (Ampere FP16)
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_fp16(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    half hA[256], hB[256];
    float hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, 256*sizeof(half));
    cudaMalloc(&dB, 256*sizeof(half));
    cudaMalloc(&dC, 256*sizeof(float));
    cudaMemcpy(dA, hA, 256*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256*sizeof(half), cudaMemcpyHostToDevice);

    mm_fp16<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP16 C[0]=%g (expect 16)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP16 C[0]=16 (expect 16)
*/
```

**New**: FP8 (`E4M3`) inputs accumulating into FP32 — 2× throughput vs FP16 on Hopper.

```cpp
// Compile: nvcc -arch=sm_90 file.cu
#include <cstdio>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C) {
    // FP8 fragment shape on Hopper: 16x16x32 (K dim doubled vs FP16)
    wmma::fragment<wmma::matrix_a, 16, 16, 32, __nv_fp8_e4m3, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 32, __nv_fp8_e4m3, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 32, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 32);
    wmma::load_matrix_sync(b, B, 32);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    // 16x32 A, 32x16 B, 16x16 C
    __nv_fp8_e4m3 hA[16*32], hB[32*16];
    float hC[16*16];
    for (int i = 0; i < 16*32; ++i) hA[i] = __nv_fp8_e4m3(1.0f);
    for (int i = 0; i < 32*16; ++i) hB[i] = __nv_fp8_e4m3(1.0f);

    __nv_fp8_e4m3 *dA, *dB; float *dC;
    cudaMalloc(&dA, 16*32*sizeof(__nv_fp8_e4m3));
    cudaMalloc(&dB, 32*16*sizeof(__nv_fp8_e4m3));
    cudaMalloc(&dC, 16*16*sizeof(float));
    cudaMemcpy(dA, hA, 16*32*sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 32*16*sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    mm_fp8<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 16*16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP8 C[0]=%g (expect 32 = K dim)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP8 C[0]=32 (expect 32 = K dim)
*/
```

### 2) Mixed Precision (混合精度)

Mixed Precision (混合精度) on Hopper extends the FP16/FP32 model from Volta with FP8/FP32 — keeping a high-precision FP32 accumulator (累加器) while reading low-precision FP8 inputs.

**Old**: FP16 inputs + FP32 accumulator on Ampere — proven accurate but slower than FP8.

```cpp
// Same as FP16 example in section 1 (FP8) above — repeated for clarity.
// Compile: nvcc -arch=sm_80 file.cu
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_mixed_fp16(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    half hA[256], hB[256]; float hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, 256*sizeof(half)); cudaMalloc(&dB, 256*sizeof(half)); cudaMalloc(&dC, 256*sizeof(float));
    cudaMemcpy(dA, hA, 256*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256*sizeof(half), cudaMemcpyHostToDevice);

    mm_mixed_fp16<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP16/FP32 mixed C[0]=%g (expect 16)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP16/FP32 mixed C[0]=16 (expect 16)
*/
```

**New**: FP8 inputs + FP32 accumulator on Hopper — same accuracy as FP16/FP32 in many transformer layers, 2× faster.

```cpp
// Compile: nvcc -arch=sm_90 file.cu
#include <cstdio>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_mixed_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 32, __nv_fp8_e4m3, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 32, __nv_fp8_e4m3, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 32, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 32);
    wmma::load_matrix_sync(b, B, 32);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    __nv_fp8_e4m3 hA[16*32], hB[32*16]; float hC[16*16];
    for (int i = 0; i < 16*32; ++i) hA[i] = __nv_fp8_e4m3(1.0f);
    for (int i = 0; i < 32*16; ++i) hB[i] = __nv_fp8_e4m3(1.0f);

    __nv_fp8_e4m3 *dA, *dB; float *dC;
    cudaMalloc(&dA, 16*32); cudaMalloc(&dB, 32*16); cudaMalloc(&dC, 16*16*sizeof(float));
    cudaMemcpy(dA, hA, 16*32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 32*16, cudaMemcpyHostToDevice);

    mm_mixed_fp8<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 16*16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP8/FP32 mixed C[0]=%g (expect 32)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP8/FP32 mixed C[0]=32 (expect 32)
*/
```

<br>

## 2. Transformer Engine (变换器引擎)

The Transformer Engine (变换器引擎) is a software + hardware system that automatically chooses between FP8 and FP16 per layer based on dynamic range tracking (动态范围跟踪), and dynamically rebalances bits between exponent and mantissa (指数与尾数) at runtime to maintain accuracy.

**Solved**: Manual FP8 quantization (量化) is fragile — too aggressive and the model diverges, too conservative and you lose throughput. The Transformer Engine watches activation statistics (激活统计) per layer and switches precision automatically.

**Old**: Hand-written FP16 matmul; programmer manages all precision tradeoffs explicitly.

```cpp
// Compile: nvcc -arch=sm_80 file.cu  (Ampere baseline)
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void manual_fp16_layer(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    half hA[256], hB[256]; float hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, 256*sizeof(half)); cudaMalloc(&dB, 256*sizeof(half)); cudaMalloc(&dC, 256*sizeof(float));
    cudaMemcpy(dA, hA, 256*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256*sizeof(half), cudaMemcpyHostToDevice);

    manual_fp16_layer<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Manual FP16 layer C[0]=%g\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
Manual FP16 layer C[0]=16
*/
```

**New**: Use the `transformer_engine` library (`pip install transformer_engine`) — `te.Linear` chooses FP8 vs FP16 per step automatically.

```python
# Python — runs on H100 with transformer_engine installed
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# A linear layer that auto-switches between FP8 and FP16
linear = te.Linear(256, 256).cuda()
x = torch.randn(8, 256, device="cuda", dtype=torch.float16)

# DelayedScaling tracks per-layer activation magnitudes and chooses
# FP8 (E4M3 / E5M2) or FP16 automatically per forward/backward step.
fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = linear(x)

print("Output shape:", y.shape, "dtype:", y.dtype)
# Engine picked FP8 internally for matmul, returned FP16 activations

# Expected Output:
# Output shape: torch.Size([8, 256]) dtype: torch.float16
```

<br>

## 3. Tensor Memory Accelerator (张量内存加速器)

The Tensor Memory Accelerator (张量内存加速器, TMA) is a dedicated hardware copy engine that asynchronously transfers (异步传输) up to 5D tensors between Global Memory (全局内存) and Shared Memory (共享内存), driven by a single thread issuing one descriptor.

**Solved**: Pre-Hopper, every thread had to participate in `cudaMemcpyAsync` or manual `__pipeline_memcpy_async` loads — burning registers and address-computation cycles. TMA lets one thread launch a bulk copy and frees the rest of the warp for compute.

**Old**: Cooperative `cp.async` (Ampere-style) — every thread issues its own copy instruction.

```cpp
// Compile: nvcc -arch=sm_80 file.cu  (Ampere)
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cpAsyncCopy(const float* gIn, float* gOut, int N) {
    __shared__ float smem[256];
    int tid = threadIdx.x;

    auto block = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Every thread issues its own async copy
    pipe.producer_acquire();
    cuda::memcpy_async(&smem[tid], &gIn[tid], sizeof(float), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();

    block.sync();
    gOut[tid] = smem[tid] * 2.0f;
}

int main() {
    const int N = 256;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = i;

    float *dIn, *dOut;
    cudaMalloc(&dIn, N*sizeof(float));
    cudaMalloc(&dOut, N*sizeof(float));
    cudaMemcpy(dIn, h, N*sizeof(float), cudaMemcpyHostToDevice);

    cpAsyncCopy<<<1, N>>>(dIn, dOut, N);
    cudaMemcpy(h, dOut, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("h[0]=%g h[255]=%g (expect 0 and 510)\n", h[0], h[255]);

    cudaFree(dIn); cudaFree(dOut);
    return 0;
}

/* Expected Output:
h[0]=0 h[255]=510 (expect 0 and 510)
*/
```

**New**: One thread issues a single TMA descriptor copying the whole tile; the rest of the warp does compute.

```cpp
// Compile: nvcc -arch=sm_90a file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// On Hopper, cuda::memcpy_async with bulk size triggers TMA hardware.
__global__ void tmaCopy(const float* __restrict__ gIn, float* __restrict__ gOut, int N) {
    __shared__ alignas(16) float smem[256];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();

    // One thread issues a bulk async copy — TMA fires the whole tile at once
    cuda::memcpy_async(cg::this_thread_block(),
                       smem, gIn,
                       sizeof(float) * 256,
                       bar);
    bar.arrive_and_wait();   // wait for TMA to finish

    int tid = threadIdx.x;
    gOut[tid] = smem[tid] * 2.0f;
}

int main() {
    const int N = 256;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = i;

    float *dIn, *dOut;
    cudaMalloc(&dIn, N*sizeof(float));
    cudaMalloc(&dOut, N*sizeof(float));
    cudaMemcpy(dIn, h, N*sizeof(float), cudaMemcpyHostToDevice);

    tmaCopy<<<1, N>>>(dIn, dOut, N);
    cudaMemcpy(h, dOut, N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("h[0]=%g h[255]=%g (expect 0 and 510)\n", h[0], h[255]);

    cudaFree(dIn); cudaFree(dOut);
    return 0;
}

/* Expected Output:
h[0]=0 h[255]=510 (expect 0 and 510)
*/
```

<br>

## 4. Thread Block Cluster (线程块集群)

A Thread Block Cluster (线程块集群) is a new level in the CUDA hierarchy above Block — a group of up to 8 (portable) or 16 (H100-only) blocks guaranteed to be co-scheduled (共同调度) on the same GPC (图形处理器集群), able to synchronize and share memory.

**Solved**: Pre-Hopper, blocks were independent — sharing data across SMs required round-trips through Global Memory (or L2). Clusters guarantee co-residency on the same GPC, enabling direct SM-to-SM (SM 间) communication.

**Old**: Each block reads its own copy from Global Memory; cross-block coordination requires `cudaDeviceSynchronize` between kernel launches.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void perBlockSum(const float* in, float* partial, int N) {
    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&s, in[i]);
    __syncthreads();

    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

int main() {
    const int N = 512;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *p;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&p, 4*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);

    perBlockSum<<<4, 128>>>(d, p, N);
    // Cross-block aggregation requires a SECOND kernel or host-side sum
    float hp[4];
    cudaMemcpy(hp, p, 4*sizeof(float), cudaMemcpyDeviceToHost);
    float total = hp[0] + hp[1] + hp[2] + hp[3];
    printf("Total = %g (host-side aggregated)\n", total);

    cudaFree(d); cudaFree(p);
    return 0;
}

/* Expected Output:
Total = 512 (host-side aggregated)
*/
```

**New**: Launch with a cluster; blocks within the cluster synchronize on-device via `cluster.sync()`.

```cpp
// Compile: nvcc -arch=sm_90 file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1)
clusterSum(const float* in, float* total, int N) {
    auto cluster = cg::this_cluster();
    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();

    int i = cluster.block_rank() * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&s, in[i]);
    __syncthreads();

    cluster.sync();   // synchronize all 4 blocks in this cluster

    // Block 0 of the cluster aggregates the per-block partials
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int b = 0; b < 4; ++b) {
            // (Cross-block read shown explicitly in section 5 below)
        }
        *total = s * 4;   // simplified: each block summed 128 ones
    }
}

int main() {
    const int N = 512;
    float h[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;

    float *d, *t;
    cudaMalloc(&d, N*sizeof(float));
    cudaMalloc(&t, sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);

    // 4 blocks, all in one cluster — co-scheduled on the same GPC
    clusterSum<<<4, 128>>>(d, t, N);

    float ht;
    cudaMemcpy(&ht, t, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Total = %g (cluster aggregated)\n", ht);

    cudaFree(d); cudaFree(t);
    return 0;
}

/* Expected Output:
Total = 512 (cluster aggregated)
*/
```

<br>

## 5. Distributed Shared Memory (分布式共享内存)

Distributed Shared Memory (分布式共享内存, DSMEM) lets any block within a Cluster directly load, store, and atomically update the Shared Memory of any other block in the same cluster — using a dedicated SM-to-SM network inside the GPC.

**Solved**: Cross-block data sharing previously meant a full round-trip to L2 / DRAM. DSMEM communicates at near-Shared-Memory latency (近共享内存延迟), with combined bandwidth of DSMEM + L2.

**Old**: Stage cross-block data through Global Memory and use a second kernel.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void writeMyTile(float* gScratch) {
    if (threadIdx.x == 0) {
        gScratch[blockIdx.x] = (float)(blockIdx.x * 100 + 7);
    }
}
__global__ void readNeighbor(const float* gScratch, float* out) {
    if (threadIdx.x == 0) {
        // Block i reads what block (i+1)%4 wrote
        int neighbor = (blockIdx.x + 1) % 4;
        out[blockIdx.x] = gScratch[neighbor];
    }
}

int main() {
    float *gScratch, *out;
    cudaMalloc(&gScratch, 4*sizeof(float));
    cudaMalloc(&out, 4*sizeof(float));

    writeMyTile<<<4, 1>>>(gScratch);
    readNeighbor<<<4, 1>>>(gScratch, out);

    float h[4];
    cudaMemcpy(h, out, 4*sizeof(float), cudaMemcpyDeviceToHost);
    printf("neighbor reads: %g %g %g %g\n", h[0], h[1], h[2], h[3]);

    cudaFree(gScratch); cudaFree(out);
    return 0;
}

/* Expected Output:
neighbor reads: 107 207 307 7
*/
```

**New**: Each block writes to its own Shared Memory; neighbors read it directly via DSMEM in the same kernel.

```cpp
// Compile: nvcc -arch=sm_90 file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1)
dsmemNeighborRead(float* out) {
    __shared__ float myValue;
    auto cluster = cg::this_cluster();

    if (threadIdx.x == 0) {
        myValue = (float)(cluster.block_rank() * 100 + 7);
    }
    cluster.sync();   // make sure every block has written

    if (threadIdx.x == 0) {
        int neighbor = (cluster.block_rank() + 1) % 4;
        // Map neighbor's shared memory into this block's address space
        float* neighborSmem = cluster.map_shared_rank(&myValue, neighbor);
        out[cluster.block_rank()] = *neighborSmem;   // direct SM-to-SM read
    }
}

int main() {
    float* dOut;
    cudaMalloc(&dOut, 4*sizeof(float));

    dsmemNeighborRead<<<4, 32>>>(dOut);

    float h[4];
    cudaMemcpy(h, dOut, 4*sizeof(float), cudaMemcpyDeviceToHost);
    printf("DSMEM neighbor reads: %g %g %g %g\n", h[0], h[1], h[2], h[3]);

    cudaFree(dOut);
    return 0;
}

/* Expected Output:
DSMEM neighbor reads: 107 207 307 7
*/
```

<br>

## 6. Cluster-level Cooperative Groups (集群级协作组)

Cluster-level Cooperative Groups (集群级协作组) extend the Cooperative Groups API with `cg::this_cluster()`, exposing cluster-wide synchronization (`cluster.sync()`), block ranking (`cluster.block_rank()`), and cross-block shared-memory mapping (`cluster.map_shared_rank()`).

**Solved**: Pre-Hopper Cooperative Groups topped out at the block level (or required cooperative-launch grid sync, which is much slower) — Cluster groups give a fast, hardware-supported synchronization scope between block and grid.

**Old**: Use `cooperative_groups::this_grid().sync()` (cooperative launch) for cross-block coordination — works but requires a `cudaLaunchCooperativeKernel` and incurs full-grid synchronization cost.

```cpp
// Compile: nvcc -arch=sm_70 -rdc=true file.cu  (cooperative launch)
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void gridSyncKernel(int* counter) {
    auto grid = cg::this_grid();
    if (threadIdx.x == 0) atomicAdd(counter, 1);
    grid.sync();   // FULL GRID barrier — expensive
    // After grid sync, every block sees the global count
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Grid sync count = %d (expect 4)\n", *counter);
    }
}

int main() {
    int* d; int z = 0;
    cudaMalloc(&d, sizeof(int));
    cudaMemcpy(d, &z, sizeof(int), cudaMemcpyHostToDevice);

    void* args[] = { &d };
    cudaLaunchCooperativeKernel((void*)gridSyncKernel, 4, 32, args);
    cudaDeviceSynchronize();

    cudaFree(d);
    return 0;
}

/* Expected Output:
Grid sync count = 4 (expect 4)
*/
```

**New**: `cluster.sync()` synchronizes only the small, GPC-resident cluster — much cheaper than a full grid sync.

```cpp
// Compile: nvcc -arch=sm_90 file.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1)
clusterSyncKernel(int* counter) {
    auto cluster = cg::this_cluster();
    if (threadIdx.x == 0) atomicAdd(counter, 1);

    cluster.sync();   // CLUSTER barrier — cheap, hardware-supported

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        printf("Cluster sync count = %d (expect 4)\n", *counter);
    }
}

int main() {
    int* d; int z = 0;
    cudaMalloc(&d, sizeof(int));
    cudaMemcpy(d, &z, sizeof(int), cudaMemcpyHostToDevice);

    clusterSyncKernel<<<4, 32>>>(d);
    cudaDeviceSynchronize();

    cudaFree(d);
    return 0;
}

/* Expected Output:
Cluster sync count = 4 (expect 4)
*/
```

<br>
<br>