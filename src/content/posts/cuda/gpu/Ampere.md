---
title: "Ampere"
published: 2026-04-27
description: "Ampere"
image: ""
tags: ["cuda","gpu","Ampere"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:21.608.672741896Z"
---
# Ampere (异步拷贝与 TF32)

Ampere (安培) is a critical generation for kernel optimization (内核优化) because it added hardware-accelerated Asynchronous Copy (异步拷贝) from Global Memory (全局内存) to Shared Memory (共享内存), known as `cp.async` (异步拷贝指令).

## 1. Architecture Diagram (架构图)

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428003922215" alt="image-20260428003922215" style="zoom:50%;" /> 

-   This file illustrates the architecture of NVIDIA Ampere（英伟达 Ampere）around 2020.
-   It uses the A100 80GB as a representative GPU model.
-   It shows the PCIe / NVLink Interface（PCIe / NVLink 接口）, Streaming Multiprocessors / SMs（流式多处理器）, Tensor Cores（张量核心）, L2 Cache（二级缓存）, HBM2e Stacks（HBM2e 堆叠显存）, and HBM2e graphics memory（HBM2e 显存）.
-   It highlights key Ampere-era features, including MIG（多实例 GPU）, sparse Tensor Cores（稀疏张量核心）, and improved AI / HPC acceleration（AI / 高性能计算加速）.
-   It summarizes key specifications, including 6912 CUDA cores（CUDA 核心）, 432 Tensor Cores（张量核心）, 7 nm process technology（7 纳米制程）, 1.41 GHz boost clock（加速频率）, 80 GB HBM2e memory（80GB HBM2e 显存）, 5120-bit memory bus（5120 位显存总线）, 2.0 TB/s memory bandwidth（显存带宽）, and 19.5 TFLOPS single-precision performance（单精度浮点性能）.

**Ampere architecture（Ampere 架构）GPU models**

-   **NVIDIA A100 40GB / A100 80GB**: Representative Ampere data-center GPU shown in the diagram.
-   **NVIDIA A30 / A40 / A10 / A16 / A2**: Ampere data-center GPUs for AI inference（AI 推理）, virtualization（虚拟化）, and cloud workloads（云工作负载）.
-   **GeForce RTX 3090 / RTX 3080 / RTX 3070 / RTX 3060**: Consumer Ampere GPUs with RT Cores（光线追踪核心）and Tensor Cores（张量核心）.
-   **RTX A6000 / RTX A5000 / RTX A4000 / RTX A2000**: Professional workstation GPUs（专业工作站 GPU）based on Ampere.
-   **NVIDIA CMP series**: Mining-focused GPUs（挖矿专用 GPU）based on Ampere variants.

<br>

## 2. Asynchronous Copy `cp.async` (异步拷贝)

`cp.async` (异步拷贝) lets a thread issue(发起...) a global-to-shared (全局到共享) copy that runs in the background (后台运行) while the thread continues computing — overlapping data movement (数据搬运) with compute (计算).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the load-then-compute serialization bottleneck (加载-计算串行化瓶颈) — pre-Ampere kernels (Ampere 之前内核) must finish loading shared memory (共享内存加载) before computing on it.

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for tiled GEMM (分块矩阵乘法), convolution (卷积), and stencils (模板计算) where you can prefetch the next tile (预取下一块) while computing on the current tile (当前块).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without `cp.async` (异步拷贝), kernels stall (停顿) on every shared-memory load (共享内存加载), wasting Tensor Core cycles (Tensor Core 周期) waiting for data (等数据).

```cpp
// Compile with: nvcc -arch=sm_80 cp_async.cu -o app
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void async_copy(const float* in, float* out, int n) {
    __shared__ float smem[256];
    auto block = cg::this_thread_block();
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    cg::memcpy_async(block, smem, in + blockIdx.x * blockDim.x,
                     sizeof(float) * blockDim.x);
    cg::wait(block);

    if (idx < n) out[idx] = smem[tid] * 2.0f;
}

int main() {
    const int n = 256;
    float h_in[256], h_out[256];
    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    async_copy<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("out[0]=%.1f out[255]=%.1f\n", h_out[0], h_out[255]);
    // Output: out[0]=0.0 out[255]=510.0

    cudaFree(d_in); cudaFree(d_out);
}
```

<br>

## 3. Tensor Cores 3rd Gen (张量核心第三代)

The 3rd-gen Tensor Cores (第三代张量核心) added FP64 (双精度), TF32 (TensorFloat-32), and BF16 (Brain Floating Point 16) support, broadening the workloads (工作负载) that can use Tensor Cores.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the workload-coverage bottleneck (工作负载覆盖瓶颈) where HPC (高性能计算) needed FP64 and ML (机器学习) needed BF16, but Tensor Cores only supported FP16 (仅支持 FP16).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for HPC simulations (HPC 仿真) using FP64 GEMM (双精度矩阵乘法), and for training (训练) using BF16 (BF16) or TF32 (TF32) without code changes.

### 3) What goes wrong without it? (不用它时有什么问题?)

Without 3rd-gen Tensor Cores (第三代张量核心), HPC and BF16 workloads (HPC 与 BF16 负载) must fall back to CUDA cores (CUDA 核心), missing 5-10x speedup (5-10 倍加速).

```cpp
// Compile with: nvcc -arch=sm_80 ampere_wmma.cu -o app
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cstdio>
using namespace nvcuda::wmma;

__global__ void bf16_wmma(const __nv_bfloat16* a, const __nv_bfloat16* b, float* c) {
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

int main() {
    const int N = 16;
    __nv_bfloat16 h_a[N * N], h_b[N * N];
    float h_c[N * N];
    for (int i = 0; i < N * N; i++) {
        h_a[i] = __float2bfloat16(1.0f);
        h_b[i] = __float2bfloat16(1.0f);
    }

    __nv_bfloat16 *d_a, *d_b;
    float* d_c;
    cudaMalloc(&d_a, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_b, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_c, N * N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    bf16_wmma<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("c[0]=%.1f c[255]=%.1f\n", h_c[0], h_c[255]);
    // Output: c[0]=16.0 c[255]=16.0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 4. TF32 (TensorFloat-32)

TF32 (TensorFloat-32) is a 19-bit format (19 位格式) with FP32's range (FP32 范围) and FP16's precision (FP16 精度), enabled automatically (自动启用) by cuBLAS/cuDNN on Ampere — no code changes (无需代码修改) required.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the FP32-Tensor-Core bottleneck (FP32 与 Tensor Core 不兼容瓶颈) — pre-Ampere FP32 kernels (FP32 内核) couldn't use Tensor Cores at all (完全不能用).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for legacy FP32 ML code (旧版 FP32 ML 代码) that wants Tensor Core acceleration (Tensor Core 加速) without rewriting to FP16/BF16.

### 3) What goes wrong without it? (不用它时有什么问题?)

Without TF32 (TF32), FP32 GEMM (FP32 矩阵乘法) runs on CUDA cores (CUDA 核心) at FP32 throughput (FP32 吞吐), missing the ~8x Tensor Core speedup (约 8 倍加速).

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

int main() {
    const int N = 16;
    float h_A[N * N], h_B[N * N], h_C[N * N];
    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("C[0]=%.1f C[255]=%.1f\n", h_C[0], h_C[255]);
    // Output: C[0]=16.0 C[255]=16.0

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

<br>

## 5. BF16 (Brain Floating Point 16)

BF16 (脑浮点 16) keeps FP32's 8-bit exponent (8 位指数) but only 7-bit mantissa (7 位尾数), giving the same dynamic range (动态范围) as FP32 — preferred for training (训练) over FP16.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the FP16-overflow bottleneck (FP16 溢出瓶颈) where FP16's small exponent range (小指数范围) causes gradient overflow (梯度溢出) during training (训练).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for large-model training (大模型训练) where gradients (梯度) span wide magnitudes (大范围量级), and as a drop-in replacement for FP32 (FP32 替代品).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without BF16 (BF16), FP16 training (FP16 训练) requires loss scaling (损失缩放) and careful tuning (细致调优), making the training pipeline (训练流水线) fragile.

```cpp
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void bf16_add(const __nv_bfloat16* a, const __nv_bfloat16* b,
                         __nv_bfloat16* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = __hadd(a[idx], b[idx]);
}

int main() {
    const int n = 8;
    __nv_bfloat16 h_a[8], h_b[8], h_c[8];
    for (int i = 0; i < n; i++) {
        h_a[i] = __float2bfloat16((float)i);
        h_b[i] = __float2bfloat16(10.0f);
    }

    __nv_bfloat16 *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(__nv_bfloat16);
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    bf16_add<<<1, n>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", __bfloat162float(h_c[i]));
    // Output: 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 6. Mature CUDA Graphs (成熟的 CUDA 图)

CUDA Graphs (CUDA 图) on Ampere (安培) became the standard pattern (标准模式) for repeating kernel sequences (重复内核序列) in deep learning training (深度学习训练) loops.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the launch-overhead bottleneck (启动开销瓶颈) for training iterations (训练迭代) with hundreds of small kernels (上百个小内核) per step (每步).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for fixed-shape training loops (固定形状训练循环) and inference servers (推理服务) where the same kernel sequence (相同内核序列) repeats every iteration (每次迭代).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without CUDA Graphs (CUDA 图), the CPU becomes the bottleneck (CPU 成为瓶颈) for fast GPUs (快速 GPU), since the host (主机) cannot launch kernels fast enough (启动够快) to keep the GPU busy.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale(float* x, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= a;
}

__global__ void shift(float* x, float b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += b;
}

int main() {
    const int n = 1024;
    float h_x = 1.0f, *d_x;
    cudaMalloc(&d_x, n * sizeof(float));
    for (int i = 0; i < n; i++)
        cudaMemcpy(d_x + i, &h_x, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    scale<<<(n + 255) / 256, 256, 0, stream>>>(d_x, 2.0f, n);
    shift<<<(n + 255) / 256, 256, 0, stream>>>(d_x, 3.0f, n);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(&h_x, d_x, sizeof(float), cudaMemcpyDeviceToHost);
    printf("x[0] = %.1f\n", h_x);
    // Output: x[0] = 5.0

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream); cudaFree(d_x);
}
```

<br>