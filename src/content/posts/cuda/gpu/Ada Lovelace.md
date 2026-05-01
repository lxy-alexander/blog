---
title: "Ada Lovelace"
published: 2026-04-27
description: "Ada Lovelace"
image: ""
tags: ["cuda","gpu","Ada Lovelace"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:42.076.925532275Z"
---
# Ada Lovelace Architecture (爱达 · 洛芙莱斯架构)

The Ada Lovelace Architecture (爱达 · 洛芙莱斯架构, 2022) is NVIDIA's gaming-and-pro-visual generation built on TSMC 4N, introducing fourth-generation Tensor Cores with FP8 (8 位浮点), third-generation RT Cores (光线追踪核心), a dedicated Optical Flow Accelerator (光流加速器) for DLSS 3 (深度学习超级采样 3) Frame Generation (帧生成), and Shader Execution Reordering (着色器执行重排).

![image-20260428191736608](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428191736608)

**Representative GPU Models**:

- GeForce RTX 4090 (AD102) — 128 SM, 16384 CUDA Cores, 512 Tensor Cores
- GeForce RTX 4080 / 4080 Super / 4070 Ti / 4070 / 4060 Ti / 4060 (AD103 / AD104 / AD106 / AD107)
- RTX 6000 Ada Generation (AD102, professional) — 142 SM, 18176 CUDA Cores
- L40 / L40S (AD102, data center / generative AI) — 142 SM
- L4 (AD104, low-power inference) — 60 SM, 72 W

**Architecture Diagram** — AD102 Example (full chip = 144 SM):

```
┌──────────────────────────────────────────────────────────────────────┐
│                       AD102 GPU (Ada Lovelace)                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │     Host Interface (PCIe 4.0) + Optical Flow Accelerator       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │    GigaThread Engine + Shader Execution Reordering (SER)       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  GPC 0    GPC 1    GPC 2    GPC 3   ...   GPC 11                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      ┌──────┐                   │
│  │ 12SM │ │ 12SM │ │ 12SM │ │ 12SM │      │ 12SM │                   │
│  │      │ │      │ │      │ │      │      │      │                   │
│  │Each  │ │Each  │ │Each  │ │Each  │      │Each  │                   │
│  │SM:   │ │SM:   │ │SM:   │ │SM:   │ ...  │SM:   │                   │
│  │128 SP│ │128 SP│ │128 SP│ │128 SP│      │128 SP│                   │
│  │4 TC  │ │4 TC  │ │4 TC  │ │4 TC  │      │4 TC  │                   │
│  │1 RT  │ │1 RT  │ │1 RT  │ │1 RT  │      │1 RT  │                   │
│  │128KB │ │128KB │ │128KB │ │128KB │      │128KB │                   │
│  │L1+Sh │ │L1+Sh │ │L1+Sh │ │L1+Sh │      │L1+Sh │                   │
│  └──────┘ └──────┘ └──────┘ └──────┘      └──────┘                   │
│       │       │       │       │                │                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              Unified L2 Cache (96 MB, 16× Ampere)              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │           GDDR6X Memory (24 GB, 384-bit, 1008 GB/s)            │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

Full AD102: 12 GPCs × 12 SMs = 144 SMs (RTX 4090 uses 128, RTX 6000 Ada uses 142)
Each SM: 128 CUDA Cores + 4 fourth-gen Tensor Cores + 1 third-gen RT Core
```

**Solved**: Ampere's L2 was small (6 MB) and Tensor Cores lacked FP8 — Ada Lovelace inherited Hopper's FP8 Tensor Core capability for client / pro-viz, increased L2 to 96 MB to absorb ray-tracing memory traffic, and added the Optical Flow Accelerator + DLSS 3 Frame Generation to multiply gaming framerates.

**Best Suited For**: Real-time ray-traced gaming (实时光线追踪游戏), generative-AI inference on workstation cards (L40S, RTX 6000 Ada), low-power inference at the edge (L4), and 8K AV1 video encoding (AV1 视频编码).

<br>

## 1. Fourth-generation Tensor Cores (第四代张量核心)

Fourth-generation Tensor Cores (第四代张量核心) on Ada bring Hopper's FP8 capability to client / pro-viz GPUs while keeping FP16 / BF16 / TF32 / INT8 support and 2:4 structured sparsity (结构化稀疏).

**Solved**: Ampere Tensor Cores topped out at FP16 / BF16 / TF32 / INT8 — Ada adds FP8 (E4M3 / E5M2) for inference / generative-AI workloads on RTX-class hardware.

### 1) FP8 (8 位浮点)

FP8 (8 位浮点) on Ada uses the same `E4M3` / `E5M2` formats as Hopper, doubling Tensor Core throughput vs FP16 — RTX 4090 reaches ~1.32 PFLOPS in FP8.

**Old**: FP16 inputs on Ampere-style Tensor Cores.

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
    half hA[256], hB[256]; float hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, 256*sizeof(half)); cudaMalloc(&dB, 256*sizeof(half)); cudaMalloc(&dC, 256*sizeof(float));
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

**New**: FP8 inputs (E4M3) accumulating into FP32 — 2× throughput on Ada's Tensor Cores.

```cpp
// Compile: nvcc -arch=sm_89 file.cu  (Ada Lovelace)
#include <cstdio>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C) {
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

### 2) INT8 (8 位整数)

INT8 (8 位整数) Tensor Core operations are inherited from Turing / Ampere — Ada keeps the same throughput as FP8 (both at 2× FP16) and adds 2:4 structured sparsity (4× theoretical speedup).

**Old**: FP16 baseline (above) — slower than INT8 / FP8.

```cpp
// See FP16 example in section 1 above — INT8 on Ada matches FP8 throughput
// at 2× FP16 baseline.
#include <cstdio>
int main() { printf("FP16 baseline (see section 1)\n"); return 0; }

/* Expected Output:
FP16 baseline (see section 1)
*/
```

**New**: INT8 inputs accumulating into INT32 — same 2× throughput as FP8 on Ada.

```cpp
// Compile: nvcc -arch=sm_89 file.cu
#include <cstdio>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_int8(const signed char* A, const signed char* B, int* C) {
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
    signed char hA[256], hB[256]; int hC[256];
    for (int i = 0; i < 256; ++i) { hA[i] = 1; hB[i] = 1; }

    signed char *dA, *dB; int *dC;
    cudaMalloc(&dA, 256); cudaMalloc(&dB, 256); cudaMalloc(&dC, 256*sizeof(int));
    cudaMemcpy(dA, hA, 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256, cudaMemcpyHostToDevice);

    mm_int8<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 256*sizeof(int), cudaMemcpyDeviceToHost);
    printf("INT8 C[0]=%d (expect 16)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
INT8 C[0]=16 (expect 16)
*/
```

### 3) Mixed Precision (混合精度)

Mixed Precision (混合精度) on Ada keeps the FP8/FP32 model from Hopper plus FP16/FP32 and BF16/FP32 — letting frameworks pick precision per layer based on numerical sensitivity (数值敏感性).

**Old**: FP16 inputs + FP32 accumulator on Ampere — proven for training but slower than FP8.

```cpp
// Same as the FP16 example in section 1.
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
    printf("FP16/FP32 C[0]=%g\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP16/FP32 C[0]=16
*/
```

**New**: FP8 inputs + FP32 accumulator on Ada — same accuracy as FP16/FP32 for many inference layers, 2× faster.

```cpp
// Compile: nvcc -arch=sm_89 file.cu
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
    printf("FP8/FP32 C[0]=%g\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP8/FP32 C[0]=32
*/
```

<br>

## 2. DLSS 3 / Optical Flow Accelerator (DLSS 3 / 光流加速器)

DLSS 3 (深度学习超级采样 3) extends DLSS 2 super resolution (超分辨率) with Frame Generation (帧生成) — synthesizing entirely new in-between frames using the dedicated Optical Flow Accelerator (光流加速器, OFA) and motion vectors (运动向量) — bypassing the CPU and rendering pipeline.

**Solved**: DLSS 2 only upscaled rendered frames (1× framerate); DLSS 3 inserts AI-generated frames between rendered ones, roughly doubling perceived framerate — useful when CPU-bound (CPU 受限) or rendering at very high resolution.

**Old**: DLSS 2 — upscale only. Pipeline: render at low res → AI upscale → display. Frame rate equals render rate.

```cpp
// Conceptual pseudo-code: a game's render loop on Ampere with DLSS 2.
// (Real DLSS access is via the NGX SDK; this shows the conceptual flow.)
#include <cstdio>

void renderFrame(int frameId) { /* GPU rasterization + ray tracing */ }
void dlssSuperResolution(int frameId) { /* Upscale 1080p -> 4K via Tensor Cores */ }
void presentFrame(int frameId) { /* Display */ }

int main() {
    // Render rate: 60 fps. DLSS 2 upscales each frame. Output rate: 60 fps.
    for (int f = 0; f < 4; ++f) {
        renderFrame(f);
        dlssSuperResolution(f);
        presentFrame(f);
        printf("DLSS2: presented frame %d (rendered)\n", f);
    }
    return 0;
}

/* Expected Output:
DLSS2: presented frame 0 (rendered)
DLSS2: presented frame 1 (rendered)
DLSS2: presented frame 2 (rendered)
DLSS2: presented frame 3 (rendered)
*/
```

**New**: DLSS 3 — upscale + Frame Generation. The OFA computes optical flow (光流) between two rendered frames, the Tensor Cores synthesize an interpolated frame, and both are presented — roughly doubling output framerate.

```cpp
// Conceptual pseudo-code: a game's render loop on Ada Lovelace with DLSS 3.
#include <cstdio>

void renderFrame(int frameId) { /* GPU rasterization + ray tracing */ }
void dlssSuperResolution(int frameId) { /* Tensor Core upscale */ }
void opticalFlowAccelerator(int prev, int curr, int generatedId) {
    /* OFA hardware computes per-pixel motion between prev and curr */
}
void aiFrameInterpolate(int prev, int curr, int generatedId) {
    /* Tensor Cores synthesize an in-between frame using OFA + motion vectors */
}
void presentFrame(int frameId) { /* Display */ }

int main() {
    // Render rate: 60 fps. DLSS 3 inserts a generated frame between
    // each pair of rendered frames. Output rate: ~120 fps.
    for (int f = 0; f < 4; ++f) {
        renderFrame(f);
        dlssSuperResolution(f);
        if (f > 0) {
            int g = 100 + f;          // generated frame id
            opticalFlowAccelerator(f - 1, f, g);
            aiFrameInterpolate(f - 1, f, g);
            presentFrame(g);
            printf("DLSS3: presented frame %d (AI-generated)\n", g);
        }
        presentFrame(f);
        printf("DLSS3: presented frame %d (rendered)\n", f);
    }
    return 0;
}

/* Expected Output:
DLSS3: presented frame 0 (rendered)
DLSS3: presented frame 101 (AI-generated)
DLSS3: presented frame 1 (rendered)
DLSS3: presented frame 102 (AI-generated)
DLSS3: presented frame 2 (rendered)
DLSS3: presented frame 103 (AI-generated)
DLSS3: presented frame 3 (rendered)
*/
```

<br>

## 3. CUDA Graphs (CUDA 图)

CUDA Graphs (CUDA 图) on Ada inherit Hopper-era improvements — graphs can now be updated cheaply (`cudaGraphExecUpdate`) so the same `cudaGraphExec_t` can be reused across iterations even when kernel parameters change, eliminating per-iteration capture overhead (捕获开销).

**Solved**: On Turing, every iteration that changed kernel arguments either re-captured the graph (expensive) or fell back to per-kernel launches — Ada-era graph updates let you swap parameters without rebuilding.

**Old**: Re-capture the graph from scratch every iteration when arguments change.

```cpp
// Compile: nvcc -arch=sm_75 file.cu  (Turing baseline)
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k) { x[0] *= k; }

int main() {
    float* d;
    float h = 1.0f;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Iterate with different k each time — re-capture every iteration
    float ks[5] = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f};
    for (int i = 0; i < 5; ++i) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        scale<<<1, 1, 0, stream>>>(d, ks[i]);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }

    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %g (expect 2*3*5*7*11 = 2310)\n", h);

    cudaStreamDestroy(stream);
    cudaFree(d);
    return 0;
}

/* Expected Output:
Result: 2310 (expect 2*3*5*7*11 = 2310)
*/
```

**New**: Capture once, then use `cudaGraphExecKernelNodeSetParams` to swap arguments cheaply.

```cpp
// Compile: nvcc -arch=sm_89 file.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void scale(float* x, float k) { x[0] *= k; }

int main() {
    float* d;
    float h = 1.0f;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Capture once with placeholder argument
    float k = 1.0f;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    scale<<<1, 1, 0, stream>>>(d, k);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Find the kernel node we want to update
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    cudaGraphNode_t* nodes = new cudaGraphNode_t[numNodes];
    cudaGraphGetNodes(graph, nodes, &numNodes);

    // Iterate — only update parameters, no re-capture / re-instantiate
    float ks[5] = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f};
    for (int i = 0; i < 5; ++i) {
        cudaKernelNodeParams params = {};
        void* args[] = { &d, &ks[i] };
        params.func = (void*)scale;
        params.gridDim = dim3(1);
        params.blockDim = dim3(1);
        params.kernelParams = args;
        cudaGraphExecKernelNodeSetParams(exec, nodes[0], &params);

        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
    }

    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %g (expect 2*3*5*7*11 = 2310)\n", h);

    delete[] nodes;
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d);
    return 0;
}

/* Expected Output:
Result: 2310 (expect 2*3*5*7*11 = 2310)
*/
```

<br>
<br>