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
# Turing (推理张量核心增强)

Turing (图灵) continued Volta's Independent Thread Scheduling (独立线程调度) and improved Tensor Cores (张量核心), especially adding INT8 (8 位整数) and INT4 (4 位整数) modes for inference (推理).

## 1. Architecture Diagram (架构图)


<br>

## 2. Tensor Cores 2nd Gen (张量核心第二代)

The 2nd-gen Tensor Cores (第二代张量核心) added INT8 (8 位整数) and INT4 (4 位整数) precision (精度) on top of FP16 (半精度) for fast inference (快速推理).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the inference throughput bottleneck (推理吞吐瓶颈) where FP16 (半精度) was overkill for many inference workloads (推理负载), wasting silicon area (硅片面积) and power (功耗).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for quantized inference (量化推理) of CNNs (卷积神经网络), Transformers (Transformer), and recommendation models (推荐模型) where INT8 accuracy is sufficient (INT8 精度足够).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without INT8 Tensor Cores (INT8 张量核心), inference latency (推理延迟) is 2-4x higher (高 2-4 倍), driving up serving cost (服务成本) and limiting batch size (批量大小).

```cpp
// Compile with: nvcc -arch=sm_75 turing_wmma.cu -o app
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda::wmma;

__global__ void int8_wmma(const int8_t* a, const int8_t* b, int* c) {
    fragment<matrix_a, 16, 16, 16, signed char, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int> c_frag;

    fill_fragment(c_frag, 0);
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

int main() {
    const int N = 16;
    int8_t h_a[N * N], h_b[N * N];
    int h_c[N * N];
    for (int i = 0; i < N * N; i++) { h_a[i] = 1; h_b[i] = 2; }

    int8_t *d_a, *d_b;
    int* d_c;
    cudaMalloc(&d_a, N * N); cudaMalloc(&d_b, N * N);
    cudaMalloc(&d_c, N * N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N, cudaMemcpyHostToDevice);

    int8_wmma<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("c[0]=%d c[255]=%d\n", h_c[0], h_c[255]);
    // Output: c[0]=32 c[255]=32

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 3. INT8 Inference (INT8 推理)

INT8 (8 位整数) inference quantizes (量化) FP32 weights and activations into 8-bit integers (8 位整数), trading a small accuracy loss (精度损失) for ~4x throughput (吞吐).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the model-size and bandwidth bottleneck (模型大小与带宽瓶颈) — INT8 weights are 4x smaller (4 倍小) than FP32, fitting in cache (放入缓存) and saturating less DRAM (减少 DRAM 占用).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for production inference (生产推理) of ResNet (残差网络), BERT, and YOLO models, where post-training quantization (训练后量化) maintains accuracy (保持精度).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without INT8 (8 位整数), large models (大模型) cannot fit in fast caches (快速缓存), causing frequent DRAM round-trips (频繁 DRAM 往返) and 4x lower throughput (4 倍低吞吐).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void int8_dot(const int8_t* a, const int8_t* b, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int prod = (int)a[idx] * (int)b[idx];
        atomicAdd(result, prod);
    }
}

int main() {
    const int n = 1024;
    int8_t h_a[1024], h_b[1024];
    for (int i = 0; i < n; i++) { h_a[i] = 1; h_b[i] = 2; }

    int8_t *d_a, *d_b;
    int *d_result, h_result = 0;
    cudaMalloc(&d_a, n); cudaMalloc(&d_b, n);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_a, h_a, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    int8_dot<<<(n + 255) / 256, 256>>>(d_a, d_b, d_result, n);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("INT8 dot = %d\n", h_result);
    // Output: INT8 dot = 2048

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
}
```

<br>

## 4. CUDA Graphs (CUDA 图)

CUDA Graphs (CUDA 图) capture a sequence of operations (操作序列) once and replay (回放) them with a single `cudaGraphLaunch`, drastically reducing CPU launch overhead (启动开销).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the per-launch CPU overhead bottleneck (单次启动 CPU 开销瓶颈) — each `<<<>>>` launch costs ~5-10 us (微秒) of CPU time (CPU 时间), dominating short kernels (短内核).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for inference loops (推理循环), training iterations (训练迭代), and any kernel sequence (内核序列) executed many times (多次执行) with identical structure (结构相同).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without CUDA Graphs (CUDA 图), an inference pipeline (推理流水线) of 100 small kernels (100 个小内核) wastes ~1 ms (毫秒) per iteration (每次迭代) on launch overhead alone (仅启动开销).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_one(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += 1.0f;
}

int main() {
    const int n = 1024;
    size_t bytes = n * sizeof(float);
    float *d_x;
    cudaMalloc(&d_x, bytes);
    cudaMemset(d_x, 0, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 10; i++)
        add_one<<<(n + 255) / 256, 256, 0, stream>>>(d_x, n);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);

    float h_x;
    cudaMemcpy(&h_x, d_x, sizeof(float), cudaMemcpyDeviceToHost);
    printf("x[0] = %.1f\n", h_x);
    // Output: x[0] = 10.0

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
}
```

<br>

## 5. Mixed Precision Continue to Enhance (混合精度延续)

Turing's mixed precision (混合精度) extends to multiple precisions (多种精度): FP16 (半精度), INT8 (8 位整数), and INT4 (4 位整数) all share the Tensor Core path (张量核心通路).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the precision-flexibility bottleneck (精度灵活性瓶颈) — different layers (不同层) and different models (不同模型) need different precisions (不同精度) for optimal accuracy/speed tradeoff (精度速度平衡).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for heterogeneous-precision pipelines (异构精度流水线) where some layers (某些层) use FP16 for accuracy and others (其他) use INT8 for speed.

### 3) What goes wrong without it? (不用它时有什么问题?)

Without flexible mixed precision (灵活混合精度), the entire model (整个模型) must use one precision (一种精度), forcing either accuracy loss (精度损失) or speed loss (速度损失) globally.

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void mixed_compute(const __half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __half2float(in[idx]);
        out[idx] = v * v + v;
    }
}

int main() {
    const int n = 8;
    __half h_in[8];
    float h_out[8];
    for (int i = 0; i < n; i++) h_in[i] = __float2half((float)i);

    __half* d_in;
    float* d_out;
    cudaMalloc(&d_in, n * sizeof(__half));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(__half), cudaMemcpyHostToDevice);

    mixed_compute<<<1, n>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_out[i]);
    // Output: 0.0 2.0 6.0 12.0 20.0 30.0 42.0 56.0

    cudaFree(d_in); cudaFree(d_out);
}
```

<br>