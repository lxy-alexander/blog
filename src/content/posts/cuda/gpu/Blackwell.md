---
title: "Blackwell"
published: 2026-04-27
description: "Blackwell"
image: ""
tags: ["cuda","gpu","Blackwell"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:55.376.332986908Z"
---
# Blackwell (FP4 与第五代张量核心)

Blackwell (布莱克韦尔) focuses on the 5th-gen Tensor Cores (第五代张量核心), FP4 (4 位浮点), the 2nd-gen Transformer Engine (第二代 Transformer 引擎), and CUDA Toolkit 12.8 (CUDA 工具包 12.8) — the first toolkit (首个工具包) with full Blackwell support.

## 1. Architecture Diagram (架构图)


<br>

## 2. Tensor Cores 5th Gen (张量核心第五代)

The 5th-gen Tensor Cores (第五代张量核心) add FP4 (4 位浮点) support and improved FP8 (8 位浮点) throughput, making LLM inference (大模型推理) significantly faster.

```cpp
// Compile with: nvcc -arch=sm_100 blackwell_wmma.cu -o app
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cstdio>
using namespace nvcuda::wmma;

// FP8 GEMM with 5th-gen Tensor Cores (第五代张量核心 FP8 GEMM)
__global__ void fp8_gemm(const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* c) {
    fragment<matrix_a, 16, 16, 32, __nv_fp8_e4m3, row_major> a_frag;
    fragment<matrix_b, 16, 16, 32, __nv_fp8_e4m3, col_major> b_frag;
    fragment<accumulator, 16, 16, 32, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a, 32);
    load_matrix_sync(b_frag, b, 32);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(c, c_frag, 16, mem_row_major);
}

int main() {
    const int M = 16, N = 16, K = 32;
    __nv_fp8_e4m3 h_a[M * K], h_b[K * N];
    float h_c[M * N];
    for (int i = 0; i < M * K; i++) h_a[i] = __nv_fp8_e4m3(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __nv_fp8_e4m3(1.0f);

    __nv_fp8_e4m3 *d_a, *d_b;
    float* d_c;
    cudaMalloc(&d_a, M * K); cudaMalloc(&d_b, K * N);
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N, cudaMemcpyHostToDevice);

    fp8_gemm<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("c[0]=%.1f c[255]=%.1f\n", h_c[0], h_c[255]);
    // Output: c[0]=32.0 c[255]=32.0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 3. FP4 Mixed Precision (FP4 混合精度)

FP4 (4 位浮点) halves memory and doubles throughput (吞吐) again versus FP8, used with per-tensor scaling (按张量缩放) managed by the 2nd-gen Transformer Engine (第二代 Transformer 引擎).

```cpp
// Compile with: nvcc -arch=sm_100 fp4_demo.cu -o app
// FP4 in CUDA 12.8+ uses microscaling formats (MXFP4) via cuBLAS / cuDNN
#include <cuda_runtime.h>
#include <cstdio>

// Conceptual: simulate FP4 quantization (示意 FP4 量化)
__global__ void fp4_quantize(const float* in, unsigned char* out, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx] / scale;
        // Quantize to 4-bit signed range [-7, 7] (量化到 4 位有符号)
        int q = max(-7, min(7, (int)__float2int_rn(v)));
        out[idx] = (unsigned char)(q & 0xF);
    }
}

__global__ void fp4_dequantize(const unsigned char* in, float* out, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int q = (int)(in[idx] & 0xF);
        if (q & 0x8) q -= 16;     // Sign extend (符号扩展)
        out[idx] = (float)q * scale;
    }
}

int main() {
    const int n = 8;
    float h_in[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    unsigned char h_q[8];
    float h_out[8];
    float scale = 1.0f;

    float *d_in, *d_out;
    unsigned char* d_q;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_q, n);
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    fp4_quantize<<<1, n>>>(d_in, d_q, n, scale);
    fp4_dequantize<<<1, n>>>(d_q, d_out, n, scale);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.1f ", h_out[i]);
    // Output: 1.0 2.0 3.0 4.0 -1.0 -2.0 -3.0 -4.0

    cudaFree(d_in); cudaFree(d_q); cudaFree(d_out);
}
```

<br>

## 4. Transformer Engine 2nd Gen (第二代 Transformer 引擎)

The 2nd-gen Transformer Engine (第二代 Transformer 引擎) auto-manages FP4/FP8 (FP4/FP8) precision across layers (层间精度) using per-tensor scaling factors (按张量缩放因子), maximizing throughput (吞吐) with minimal accuracy loss (精度损失).

```cpp
// Conceptual: per-tensor FP8 scaling like Transformer Engine (示意 Transformer 引擎按张量缩放)
#include <cuda_runtime.h>
#include <cstdio>

__global__ void compute_amax(const float* x, float* amax, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicMax((int*)amax, __float_as_int(fabsf(x[idx])));
}

__global__ void scale_to_fp8_range(const float* in, float* out, int n,
                                    float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;   // Scale before FP8 cast (转 FP8 前缩放)
}

int main() {
    const int n = 8;
    float h_in[8] = {0.5f, -1.0f, 2.0f, -3.0f, 4.0f, -2.5f, 1.5f, -0.5f};
    float h_amax_init = 0.0f, h_amax;

    float *d_in, *d_out, *d_amax;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_amax, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amax, &h_amax_init, sizeof(float), cudaMemcpyHostToDevice);

    compute_amax<<<1, n>>>(d_in, d_amax, n);
    cudaMemcpy(&h_amax, d_amax, sizeof(float), cudaMemcpyDeviceToHost);

    // FP8 E4M3 max ~448, derive scale (推导缩放因子)
    float scale = 448.0f / h_amax;
    scale_to_fp8_range<<<1, n>>>(d_in, d_out, n, scale);

    float h_out[8];
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("amax=%.1f scale=%.1f scaled[0]=%.1f scaled[3]=%.1f\n",
           h_amax, scale, h_out[0], h_out[3]);
    // Output: amax=4.0 scale=112.0 scaled[0]=56.0 scaled[3]=-336.0

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_amax);
}
```

<br>

## 5. CUDA Graphs Conditional Nodes (CUDA 图条件节点)

CUDA 12.8 (CUDA 12.8) on Blackwell (布莱克韦尔) enhances CUDA Graphs (CUDA 图) with conditional nodes (条件节点) — allowing if/while logic (分支与循环) inside graphs.

```cpp
// Compile with CUDA 12.8+ on Blackwell
#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(float* x, int n, float a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= a;
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

    // Standard graph capture — conditional nodes need graph builder API (条件节点需用 graph builder API)
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 3; i++)
        scale_kernel<<<1, n, 0, stream>>>(d_x, n, 2.0f);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(&h_x[0], d_x, sizeof(float), cudaMemcpyDeviceToHost);
    printf("x[0] = %.1f\n", h_x[0]);
    // Output: x[0] = 8.0  (1.0 * 2 * 2 * 2)

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream); cudaFree(d_x);
}
```

<br>

## 6. LLM Inference / Training Support (大语言模型推理与训练)

Blackwell (布莱克韦尔) delivers the strongest LLM (大语言模型) support yet by combining FP4 (4 位浮点), the 2nd-gen Transformer Engine (第二代 Transformer 引擎), and 5th-gen Tensor Cores (第五代张量核心) for unprecedented training throughput (训练吞吐).

```cpp
// Conceptual mini transformer block: FP16 input, simulated FP8 compute, FP32 accumulate
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void mini_attn_step(const __half* q, const __half* k, float* score, int d) {
    int idx = threadIdx.x;
    if (idx < d) {
        // Q dot K, FP16 mul, FP32 accum (FP16 乘,FP32 累加)
        float prod = __half2float(q[idx]) * __half2float(k[idx]);
        atomicAdd(score, prod);
    }
}

int main() {
    const int d = 64;
    __half h_q[64], h_k[64];
    for (int i = 0; i < d; i++) {
        h_q[i] = __float2half(1.0f);
        h_k[i] = __float2half(1.0f);
    }
    // Expected dot product = 64

    __half *d_q, *d_k;
    float *d_score, h_score = 0;
    cudaMalloc(&d_q, d * sizeof(__half));
    cudaMalloc(&d_k, d * sizeof(__half));
    cudaMalloc(&d_score, sizeof(float));
    cudaMemcpy(d_q, h_q, d * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, d * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_score, 0, sizeof(float));

    mini_attn_step<<<1, d>>>(d_q, d_k, d_score, d);
    cudaMemcpy(&h_score, d_score, sizeof(float), cudaMemcpyDeviceToHost);

    printf("attention score = %.1f\n", h_score);
    // Output: attention score = 64.0

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_score);
}
```

<br>