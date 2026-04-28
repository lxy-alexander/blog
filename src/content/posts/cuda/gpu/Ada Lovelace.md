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
# Ada Lovelace (图形与推理增强)

Ada Lovelace (阿达·洛芙莱斯) leans toward graphics (图形), ray tracing (光线追踪), and inference (推理), continuing the Tensor Core (张量核心) and Mixed Precision (混合精度) path with smaller changes (更小变化) to the general CUDA programming model (通用 CUDA 编程模型).

## 1. Architecture Diagram (架构图)

<br>

## 2. Tensor Cores 4th Gen (张量核心第四代)

Ada (阿达) uses 4th-gen Tensor Cores (第四代张量核心) similar to Hopper (霍珀), supporting FP8 (8 位浮点) for accelerated inference (加速推理) on consumer/workstation GPUs.

```cpp
// Compile with: nvcc -arch=sm_89 ada_wmma.cu -o app
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
using namespace nvcuda::wmma;

__global__ void fp16_wmma(const __half* a, const __half* b, float* c) {
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    mma_sync(c_frag, a_frag, b_frag, c_frag);  // 4th gen tensor core (第四代)
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

int main() {
    const int N = 16;
    __half h_a[N * N], h_b[N * N];
    float h_c[N * N];
    for (int i = 0; i < N * N; i++) {
        h_a[i] = __float2half(1.0f);
        h_b[i] = __float2half(2.0f);
    }

    __half *d_a, *d_b;
    float* d_c;
    cudaMalloc(&d_a, N * N * sizeof(__half));
    cudaMalloc(&d_b, N * N * sizeof(__half));
    cudaMalloc(&d_c, N * N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(__half), cudaMemcpyHostToDevice);

    fp16_wmma<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("c[0]=%.1f c[255]=%.1f\n", h_c[0], h_c[255]);
    // Output: c[0]=32.0 c[255]=32.0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 3. Mixed Precision Inference (混合精度推理)

Ada (阿达) excels at low-precision inference (低精度推理) — the workflow combines FP16 (半精度) for compute and FP32 (单精度) for accumulation (累加) to balance speed and accuracy.

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void mixed_inference(const __half* x, const __half* w, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // FP16 multiply, FP32 accumulate (FP16 乘,FP32 累加)
        float prod = __half2float(x[idx]) * __half2float(w[idx]);
        atomicAdd(y, prod);
    }
}

int main() {
    const int n = 1024;
    __half h_x[1024], h_w[1024];
    for (int i = 0; i < n; i++) {
        h_x[i] = __float2half(0.5f);
        h_w[i] = __float2half(2.0f);
    }
    // Expected = 1024 * 0.5 * 2 = 1024

    __half *d_x, *d_w;
    float *d_y, h_y = 0;
    cudaMalloc(&d_x, n * sizeof(__half));
    cudaMalloc(&d_w, n * sizeof(__half));
    cudaMalloc(&d_y, sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(float));

    mixed_inference<<<(n + 255) / 256, 256>>>(d_x, d_w, d_y, n);
    cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    printf("inference y = %.1f\n", h_y);
    // Output: inference y = 1024.0

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
}
```

<br>

## 4. INT8 Inference Acceleration (INT8 推理加速)

Ada (阿达) keeps strong INT8 (8 位整数) support for inference (推理), which quantizes (量化) FP32 models into INT8 for ~4x throughput (~4 倍吞吐).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void int8_matvec(const int8_t* W, const int8_t* x, int* y, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int sum = 0;
        for (int k = 0; k < K; k++)
            sum += (int)W[row * K + k] * (int)x[k];   // INT8 mul, INT32 accum (INT32 累加)
        y[row] = sum;
    }
}

int main() {
    const int M = 4, K = 8;
    int8_t h_W[M * K], h_x[K];
    int h_y[M];
    for (int i = 0; i < M * K; i++) h_W[i] = 1;
    for (int i = 0; i < K; i++) h_x[i] = 2;
    // Expected each y = 8 * 1 * 2 = 16

    int8_t *d_W, *d_x;
    int* d_y;
    cudaMalloc(&d_W, M * K); cudaMalloc(&d_x, K);
    cudaMalloc(&d_y, M * sizeof(int));
    cudaMemcpy(d_W, h_W, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, K, cudaMemcpyHostToDevice);

    int8_matvec<<<1, M>>>(d_W, d_x, d_y, M, K);
    cudaMemcpy(h_y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) printf("%d ", h_y[i]);
    // Output: 16 16 16 16

    cudaFree(d_W); cudaFree(d_x); cudaFree(d_y);
}
```

<br>

## 5. CUDA Graphs Continued (CUDA 图延续)

CUDA Graphs (CUDA 图) on Ada (阿达) continue to reduce launch overhead (启动开销) for repetitive inference (重复推理) loops — essential for serving (服务) low-latency models (低延迟模型).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void layer1(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = x[idx] * 2.0f;     // Scale
}
__global__ void layer2(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = x[idx] + 1.0f;     // Bias
}
__global__ void layer3(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = x[idx] > 0 ? x[idx] : 0;  // ReLU
}

int main() {
    const int n = 256;
    float h_x[256];
    for (int i = 0; i < n; i++) h_x[i] = 1.0f;

    float* d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    layer1<<<1, n, 0, stream>>>(d_x, n);  // 1 -> 2
    layer2<<<1, n, 0, stream>>>(d_x, n);  // 2 -> 3
    layer3<<<1, n, 0, stream>>>(d_x, n);  // 3 -> 3
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("x[0] = %.1f\n", h_x[0]);
    // Output: x[0] = 3.0

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream); cudaFree(d_x);
}
```

<br>