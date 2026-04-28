---
title: "Turing"
published: 2026-04-27
description: "Turing"
image: ""
tags: ["cuda","gpu","Turing"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:07.736.447460054Z"
---
# Turing Architecture (图灵架构)

The Turing Architecture (图灵架构, 2018) introduced second-generation Tensor Cores (第二代张量核心) with INT8 / INT4 support, dedicated RT Cores (光线追踪核心) for ray tracing (光线追踪), and CUDA Graphs (CUDA 图) — letting the runtime capture and replay entire kernel sequences with minimal launch overhead.

**Representative GPU Models**:

- GeForce RTX 2060 / 2070 / 2080 / 2080 Ti (TU102 / TU104 / TU106) — 30–68 SM
- GeForce GTX 1660 / 1660 Ti (TU116, no RT/Tensor Cores)
- TITAN RTX (TU102) — 72 SM, 4608 CUDA Cores + 576 Tensor Cores
- Quadro RTX 6000 / 8000
- Tesla T4 (TU104, inference, 推理) — 40 SM, 2560 CUDA Cores + 320 Tensor Cores

**Architecture Diagram** — TU102 Example (72 SM):

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TU102 GPU (Turing)                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │     Host Interface (PCIe 3.0) + NVLink 2.0 (TU102/TU104)       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                       GigaThread Engine                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   GPC 0          GPC 1         ...                       GPC 5       │
│  ┌─────────┐    ┌─────────┐                            ┌─────────┐   │
│  │ 12 SM   │    │ 12 SM   │                            │ 12 SM   │   │
│  │         │    │         │                            │         │   │
│  │ Each SM:│    │ Each SM:│           ...              │ Each SM:│   │
│  │ 64 SP   │    │ 64 SP   │                            │ 64 SP   │   │
│  │ 8 Tensor│    │ 8 Tensor│                            │ 8 Tensor│   │
│  │ 1 RT    │    │ 1 RT    │                            │ 1 RT    │   │
│  │ Core    │    │ Core    │                            │ Core    │   │
│  │ 96 KB   │    │ 96 KB   │                            │ 96 KB   │   │
│  │L1+Shmem │    │L1+Shmem │                            │L1+Shmem │   │
│  └─────────┘    └─────────┘                            └─────────┘   │
│       │              │                                       │       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                   Unified L2 Cache (6 MB)                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │           GDDR6 Memory (11–24 GB, 384-bit, 616 GB/s)           │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

Total: 72 SM × 64 SP = 4608 CUDA Cores + 72 SM × 8 = 576 Tensor Cores
       + 72 RT Cores
Each SM: 96 KB unified L1 + Shared Memory, concurrent FP + INT execution
```

**Solved**: Volta's Tensor Cores supported only FP16 inputs — Turing extended them to INT8 and INT4 for inference (推理) workloads, added dedicated RT Cores for hardware-accelerated ray tracing, and introduced CUDA Graphs to slash CPU launch overhead (CPU 启动开销) for repetitive kernel sequences.

**Best Suited For**: Real-time inference (实时推理), ray-traced rendering (光追渲染), and small-kernel iterative workloads (小核函数迭代负载) like deep-learning inference batches and physics solvers.

<br>

## 1. Second-generation Tensor Cores (第二代张量核心)

Second-generation Tensor Cores (第二代张量核心) extend Volta's FP16 matrix-multiply hardware with INT8 (8-bit integer) and INT4 (4-bit integer) modes, doubling and quadrupling throughput respectively for inference (推理).

**Solved**: Volta only supported FP16 — but inference often tolerates lower precision; Turing's integer modes deliver much higher OPS (operations-per-second) at lower power.

### 1) INT8 (8 位整数)

INT8 (8 位整数) Tensor Core operations multiply 8-bit integer matrices and accumulate into 32-bit integers — typically 2× throughput vs FP16 for the same hardware.

**Old**: FP16 inputs on Volta-style Tensor Cores — accurate but slower for inference.

```cpp
// Compile: nvcc -arch=sm_70 file.cu  (Volta-style FP16)
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
    printf("FP16 C[0]=%g C[255]=%g (expect 16)\n", hC[0], hC[255]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP16 C[0]=16 C[255]=16 (expect 16)
*/
```

**New**: INT8 inputs accumulating into INT32 — 2× throughput on Turing Tensor Cores.

```cpp
// Compile: nvcc -arch=sm_75 file.cu
#include <cstdio>
#include <mma.h>
using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__global__ void mm_int8(const signed char* A, const signed char* B, int* C) {
    // Turing INT8 fragment shape: 16x16x16, INT8 inputs, INT32 accumulator
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
    wmma::fill_fragment(c, 0);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    signed char hA[256], hB[256];
    int hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = 1; hB[i] = 1; }

    signed char *dA, *dB; int *dC;
    cudaMalloc(&dA, 256);
    cudaMalloc(&dB, 256);
    cudaMalloc(&dC, 256*sizeof(int));
    cudaMemcpy(dA, hA, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);

    mm_int8<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(int), cudaMemcpyDeviceToHost);
    printf("INT8 C[0]=%d C[255]=%d (expect 16)\n", hC[0], hC[255]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
INT8 C[0]=16 C[255]=16 (expect 16)
*/
```

### 2) INT4 (4 位整数)

INT4 (4 位整数) Tensor Core operations pack two 4-bit integers per byte and deliver 4× the throughput of FP16 — used for highly quantized inference (量化推理) where accuracy loss is acceptable.

**Old**: INT8 — 2× throughput vs FP16 but still 8 bits per operand.

```cpp
// (See INT8 example above — Old here is identical INT8 code.)
// Same kernel as the INT8 example in section 1) above; INT8 is the
// "old" baseline that INT4 doubles in throughput.
#include <cstdio>
int main() { printf("INT8 baseline (see section 1)\n"); return 0; }

/* Expected Output:
INT8 baseline (see section 1)
*/
```

**New**: INT4 inputs via the `experimental::precision::s4` (signed 4-bit) WMMA type — 4× FP16 throughput.

```cpp
// Compile: nvcc -arch=sm_75 file.cu
// Note: INT4 WMMA uses an opaque 32-bit container holding 8 packed s4 values.
#include <cstdio>
#include <mma.h>
using namespace nvcuda;
using namespace nvcuda::wmma::experimental;

__global__ void mm_int4(const int* A, const int* B, int* C) {
    // 8x8x32 fragment shape for INT4 on Turing
    wmma::fragment<wmma::matrix_a, 8, 8, 32, precision::s4, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, precision::s4, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c;

    wmma::fill_fragment(c, 0);
    wmma::load_matrix_sync(a, (const precision::s4*)A, 32);
    wmma::load_matrix_sync(b, (const precision::s4*)B, 32);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 8, wmma::mem_row_major);
}

int main() {
    // 8x32 INT4 matrix = 256 s4 values = 32 ints (each holds 8 packed s4)
    // Pack each s4 = 1 → byte 0x11; int = 0x11111111 = all-ones
    int hA[32], hB[32], hC[64];
    for (int i = 0; i < 32; ++i) { hA[i] = 0x11111111; hB[i] = 0x11111111; }

    int *dA, *dB, *dC;
    cudaMalloc(&dA, 32*sizeof(int));
    cudaMalloc(&dB, 32*sizeof(int));
    cudaMalloc(&dC, 64*sizeof(int));
    cudaMemcpy(dA, hA, 32*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 32*sizeof(int), cudaMemcpyHostToDevice);

    mm_int4<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 64*sizeof(int), cudaMemcpyDeviceToHost);
    printf("INT4 C[0]=%d C[63]=%d (expect 32 = K dim)\n", hC[0], hC[63]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
INT4 C[0]=32 C[63]=32 (expect 32 = K dim)
*/
```

### 3) Mixed Precision (混合精度)

Mixed Precision (混合精度) on Turing keeps the same FP16-input / FP32-accumulate model from Volta — preserved for training (训练) workloads where INT8/INT4 cannot match FP16 accuracy.

**Old**: Pure FP32 matrix multiply on CUDA Cores — slow.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void mm_fp32(const float* A, const float* B, float* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    float sum = 0.f;
    for (int k = 0; k < 16; ++k) sum += A[row*16+k] * B[k*16+col];
    C[row*16+col] = sum;
}

int main() {
    float hA[256], hB[256], hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = 1.0f; hB[i] = 1.0f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, 256*sizeof(float));
    cudaMalloc(&dB, 256*sizeof(float));
    cudaMalloc(&dC, 256*sizeof(float));
    cudaMemcpy(dA, hA, 256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256*sizeof(float), cudaMemcpyHostToDevice);

    mm_fp32<<<1, dim3(16,16)>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP32 C[0]=%g (expect 16)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP32 C[0]=16 (expect 16)
*/
```

**New**: FP16 inputs + FP32 accumulator on Tensor Cores — much faster, same accuracy as FP32.

```cpp
// Compile: nvcc -arch=sm_75 file.cu
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_mixed(const half* A, const half* B, float* C) {
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

    mm_mixed<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Mixed C[0]=%g (expect 16)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
Mixed C[0]=16 (expect 16)
*/
```

<br>

## 2. CUDA Graphs (CUDA 图)

CUDA Graphs (CUDA 图) capture an entire sequence of kernel launches and memory operations into a graph object that can be replayed (重放) with a single call — eliminating per-kernel CPU launch overhead (CPU 启动开销).

**Solved**: For workloads with many small kernels (e.g. transformer inference), CPU launch overhead can dominate runtime — replaying a pre-built graph submits all work to the GPU with one API call.

**Old**: Launch each kernel via `<<<>>>` every iteration — host launches every time.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void inc(int* x) { (*x)++; }
__global__ void dbl(int* x) { (*x) *= 2; }

int main() {
    int* d;
    int h = 1;
    cudaMalloc(&d, sizeof(int));
    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);

    // 5 iterations of (inc, dbl) — each iteration costs 2 host launches
    for (int i = 0; i < 5; ++i) {
        inc<<<1,1>>>(d);
        dbl<<<1,1>>>(d);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    // (((((1+1)*2 +1)*2 +1)*2 +1)*2 +1)*2 = 94
    printf("Iterative launches: %d (expect 94)\n", h);

    cudaFree(d);
    return 0;
}

/* Expected Output:
Iterative launches: 94 (expect 94)
*/
```

**New**: Capture the (inc, dbl) sequence once into a graph; replay it 5 times with a single launch each.

```cpp
// Compile: nvcc -arch=sm_75 file.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void inc(int* x) { (*x)++; }
__global__ void dbl(int* x) { (*x) *= 2; }

int main() {
    int* d;
    int h = 1;
    cudaMalloc(&d, sizeof(int));
    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Capture the (inc, dbl) sequence into a graph
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    inc<<<1, 1, 0, stream>>>(d);
    dbl<<<1, 1, 0, stream>>>(d);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Replay 5 times — almost no host overhead per replay
    for (int i = 0; i < 5; ++i) {
        cudaGraphLaunch(exec, stream);
    }
    cudaStreamSynchronize(stream);

    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Graph replays: %d (expect 94)\n", h);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d);
    return 0;
}

/* Expected Output:
Graph replays: 94 (expect 94)
*/
```

<br>
<br>