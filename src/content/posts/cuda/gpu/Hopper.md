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
# Hopper (FP8、TMA、线程块集群)

Hopper (霍珀) is critical for large model training (大模型训练) because it introduced the Transformer Engine (Transformer 引擎), FP8 (8 位浮点), Thread Block Cluster (线程块集群), and Tensor Memory Accelerator / TMA (张量内存加速器).

## 1. Architecture Diagram (架构图)



<br>

## 2. Tensor Cores 4th Gen (张量核心第四代)

The 4th-gen Tensor Cores (第四代张量核心) added FP8 (8 位浮点) support with two formats — E4M3 (4 位指数 3 位尾数) and E5M2 (5 位指数 2 位尾数) — for ultra-fast (极速) LLM training and inference.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the LLM throughput bottleneck (大模型吞吐瓶颈) where FP16/BF16 (FP16/BF16) was no longer fast enough for trillion-parameter models (万亿参数模型).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for LLM training (大模型训练) and inference (推理), particularly large GEMMs (大矩阵乘法) in attention (注意力) and FFN layers (前馈层).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without FP8 Tensor Cores (FP8 张量核心), training a 100B+ parameter model (千亿参数模型) takes 2x more time (2 倍时间) and memory (内存), driving up training cost (训练成本).

```cpp
// Compile with: nvcc -arch=sm_90 hopper_wmma.cu -o app
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cstdio>
using namespace nvcuda::wmma;

__global__ void fp8_wmma(const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, float* c) {
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

    fp8_wmma<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("c[0]=%.1f c[255]=%.1f\n", h_c[0], h_c[255]);
    // Output: c[0]=32.0 c[255]=32.0

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```

<br>

## 3. FP8 Mixed Precision (FP8 混合精度)

FP8 (8 位浮点) halves memory footprint (内存占用) and doubles arithmetic throughput (算术吞吐) versus FP16 (半精度), while Transformer Engine (Transformer 引擎) auto-manages numeric stability (数值稳定性).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the model-memory-and-bandwidth bottleneck (模型内存与带宽瓶颈) — FP8 weights are half the size of FP16 (FP16 一半大小), letting larger models (更大模型) fit on the same GPU.

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for Transformer GEMMs (Transformer 矩阵乘法), attention QKV projections (QKV 投影), and any FP16/BF16 kernel (FP16/BF16 内核) with stable numerical behavior (数值稳定).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without FP8 (FP8), model size (模型大小) is bounded by GPU memory (受 GPU 内存限制) at half the capacity (一半容量), forcing model parallelism (模型并行) earlier than necessary (不必要地提前).

```cpp
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>

__global__ void fp8_to_fp32(const __nv_fp8_e4m3* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = float(in[idx]);
}

int main() {
    const int n = 8;
    __nv_fp8_e4m3 h_in[8];
    float h_out[8];
    for (int i = 0; i < n; i++) h_in[i] = __nv_fp8_e4m3((float)i * 0.5f);

    __nv_fp8_e4m3* d_in;
    float* d_out;
    cudaMalloc(&d_in, n); cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n, cudaMemcpyHostToDevice);

    fp8_to_fp32<<<1, n>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%.2f ", h_out[i]);
    // Output: 0.00 0.50 1.00 1.50 2.00 2.50 3.00 3.50

    cudaFree(d_in); cudaFree(d_out);
}
```

<br>

## 4. Cluster-Level Cooperative Groups (集群级协作组)

Cooperative Groups (协作组) on Hopper (霍珀) extends to cluster-level programming (集群级编程) — a new tier (新层级) above the thread block (线程块).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the inter-block coordination bottleneck (块间协调瓶颈) — pre-Hopper (Hopper 之前), blocks (块) couldn't synchronize directly (直接同步), only via global memory (全局内存).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for kernels (内核) needing cross-block cooperation (跨块协作) like large reductions (大规模归约), persistent threads (常驻线程), and producer-consumer pipelines (生产者-消费者流水线).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without cluster cooperation (集群协作), inter-block sync (块间同步) requires kernel relaunch (内核重启) or global atomic spin-locks (全局原子自旋锁), both adding latency (延迟).

```cpp
// Compile with: nvcc -arch=sm_90 cluster_cg.cu -o app
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(int* out) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    int cluster_rank = cluster.block_rank();
    int tid = block.thread_rank();

    if (tid == 0) out[cluster_rank] = blockIdx.x;
}

int main() {
    int *d_out, h_out[2];
    cudaMalloc(&d_out, 2 * sizeof(int));

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(32, 1, 1);
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    config.attrs = &attr; config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluster_kernel, d_out);
    cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("out[0]=%d out[1]=%d\n", h_out[0], h_out[1]);
    // Output: out[0]=0 out[1]=1

    cudaFree(d_out);
}
```

<br>

## 5. TMA — Tensor Memory Accelerator (张量内存加速器)

TMA (张量内存加速器) is a dedicated hardware engine (专用硬件引擎) for asynchronous bulk tensor copies (异步批量张量拷贝) between global and shared memory — a major upgrade over Ampere's `cp.async`.

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the per-thread-load bottleneck (单线程加载瓶颈) of `cp.async` — every thread (线程) had to issue its own copy instruction (自己的拷贝指令), wasting register and instruction bandwidth (寄存器与指令带宽).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for tiled GEMM (分块矩阵乘法), large convolutions (大卷积), and any kernel (内核) that loads multi-dimensional tiles (多维分块) from global memory (全局内存).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without TMA (张量内存加速器), tile-loading code (分块加载代码) is verbose (冗长), uses many registers (大量寄存器), and competes for instruction issue slots (指令发射槽位) with compute (计算).

```cpp
// Compile with: nvcc -arch=sm_90a tma_demo.cu -o app
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void tma_like_copy(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    auto block = cg::this_thread_block();
    int tid = threadIdx.x;

    cg::memcpy_async(block, smem, in + blockIdx.x * blockDim.x,
                     sizeof(float) * blockDim.x);
    cg::wait(block);

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) out[idx] = smem[tid] + 100.0f;
}

int main() {
    const int n = 256;
    float h_in[256], h_out[256];
    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    tma_like_copy<<<1, 256, 256 * sizeof(float)>>>(d_in, d_out, n);
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("out[0]=%.1f out[255]=%.1f\n", h_out[0], h_out[255]);
    // Output: out[0]=100.0 out[255]=355.0

    cudaFree(d_in); cudaFree(d_out);
}
```

<br>

## 6. Thread Block Cluster (线程块集群)

Thread Block Cluster (线程块集群) groups multiple thread blocks (线程块) so they can synchronize (同步) and share data (共享数据) via Distributed Shared Memory (分布式共享内存).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the block-size limit bottleneck (线程块大小上限瓶颈) — blocks (线程块) cap at 1024 threads (1024 线程) and ~228 KB shared memory (~228 KB 共享内存), but some workloads need more (需要更多).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for large tile GEMM (大分块矩阵乘法), wide attention windows (宽注意力窗口), and any kernel (内核) where the tile size (分块大小) exceeds a single block's resources (单块资源).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without clusters (集群), large tiles (大分块) must be split across multiple blocks (多块), losing shared memory locality (共享内存局部性) and adding global memory traffic (全局内存流量).

```cpp
// Compile with: nvcc -arch=sm_90 cluster.cu -o app
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) cluster_sync_kernel(int* out) {
    cg::cluster_group cluster = cg::this_cluster();
    int tid = threadIdx.x;
    int block_rank = cluster.block_rank();

    cluster.sync();

    if (tid == 0) out[block_rank] = block_rank * 100;
}

int main() {
    int *d_out, h_out[2];
    cudaMalloc(&d_out, 2 * sizeof(int));

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(32, 1, 1);
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    config.attrs = &attr; config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluster_sync_kernel, d_out);
    cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("out[0]=%d out[1]=%d\n", h_out[0], h_out[1]);
    // Output: out[0]=0 out[1]=100

    cudaFree(d_out);
}
```

<br>

## 7. Distributed Shared Memory (分布式共享内存)

Distributed Shared Memory (分布式共享内存) lets blocks (块) within a cluster (集群) read and write each other's shared memory (共享内存) directly via cluster pointers (集群指针).

### 1) What bottleneck does it solve? (解决了什么瓶颈?)

It solves the cross-block sharing bottleneck (跨块共享瓶颈) where data sharing (数据共享) between blocks (块) had to go through slow global memory (慢全局内存).

### 2) What kernels is it good for? (适合什么 kernel?)

It is good for split-K GEMM (Split-K 矩阵乘法), large stencils (大模板计算), and producer-consumer kernels (生产者-消费者内核) within a cluster (集群).

### 3) What goes wrong without it? (不用它时有什么问题?)

Without distributed shared memory (分布式共享内存), inter-block data exchange (块间数据交换) goes through DRAM (DRAM), adding ~10x latency (~10 倍延迟) versus shared memory (共享内存).

```cpp
// Compile with: nvcc -arch=sm_90 dsmem.cu -o app
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) dsmem_kernel(int* out) {
    __shared__ int smem[1];
    cg::cluster_group cluster = cg::this_cluster();

    int rank = cluster.block_rank();
    smem[0] = rank * 10;
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
        int* remote = cluster.map_shared_rank(smem, 1);
        out[0] = smem[0];
        out[1] = *remote;
    }
}

int main() {
    int *d_out, h_out[2];
    cudaMalloc(&d_out, 2 * sizeof(int));

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(32, 1, 1);
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    config.attrs = &attr; config.numAttrs = 1;

    cudaLaunchKernelEx(&config, dsmem_kernel, d_out);
    cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("local=%d remote=%d\n", h_out[0], h_out[1]);
    // Output: local=0 remote=10

    cudaFree(d_out);
}
```

<br>