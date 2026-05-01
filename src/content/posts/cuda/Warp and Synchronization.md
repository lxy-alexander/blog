---
title: "Warp and Synchronization"
published: 2026-04-27
description: "Warp and Synchronization"
image: ""
tags: ["cuda","Warp and Synchronization"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-28T03:20:52.591.560148077Z"
---

# Warp and Synchronization (线程束与同步)

Goal: understand that the GPU's actual execution unit (执行单位) is the Warp (线程束), not a single Thread (线程).

## 1. Warp Execution Architecture (Warp 执行架构)

A Warp (线程束) is a group of 32 threads (线程) executing the same instruction (指令) in lockstep — divergence (分支发散) forces the hardware to serialize the branches.<br>

## 2. Concepts to Learn

### 1) Warp (线程束)

A Warp (线程束) is the GPU's hardware scheduling unit (硬件调度单位) — 32 threads (线程) issued together as one instruction.
The warp scheduler switches among ready warps, so while one warp is waiting, the SM can keep executing other warps. This is not a heavy CPU-style context switch. It is a lightweight hardware scheduling decision.


### 2) Warp Size (线程束大小)

The Warp Size (线程束大小) is **32** on all current NVIDIA GPUs, which is why block sizes (线程块大小) should be multiples of 32.

### 3) SIMT (单指令多线程)

SIMT (Single Instruction Multiple Thread, 单指令多线程) means all 32 threads (线程) in a warp (线程束) execute the same instruction (指令) on different data each cycle.

### 4) Warp Divergence (线程束分支发散)

Warp Divergence (线程束分支发散) occurs when threads (线程) in the same warp (线程束) take different control-flow paths (控制流路径), forcing the hardware to **serialize** (串行化) each path.

### 5) `__syncthreads()` (线程块同步)

`__syncthreads()` is a barrier (屏障) that synchronizes **all threads in a block** (整个线程块), commonly used When threads read and write to each other through shared memory.

### 6) Warp-level Primitives (Warp 级原语)

Warp-level primitives (Warp 级原语) are intrinsics (内建函数) like `__shfl_sync` and `__ballot_sync` that let threads (线程) in the same warp (线程束) exchange data **without using shared memory** (无需共享内存).

### 7) `__shfl_sync` (Warp 内数据交换)

`__shfl_sync` lets a thread (线程) directly read a register (寄存器) value from another thread (线程) in the same warp (线程束) in **one cycle** (单周期).

### 8) `__ballot_sync` (Warp 投票)

`__ballot_sync` returns a 32-bit mask (掩码) where bit `i` is set if lane `i` evaluates the predicate (谓词) as true — useful for warp-wide voting (投票).

### 9) `__activemask()` (活跃线程掩码)

`__activemask()` returns the mask (掩码) of currently active lanes (活跃通道) in the warp (线程束), needed when divergence (分支发散) is unavoidable and you must pass the correct mask to `_sync` primitives.

<br>

## 3. Code to Write

### 1) warp_id_demo.cu (Warp ID 演示)

Each thread (线程) belongs to a warp (线程束) determined by `threadIdx.x / 32`, and `lane id` is `threadIdx.x % 32`.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void warp_id_demo() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;        // Which warp (哪个 warp)
    int lane_id = tid % 32;        // Which lane (哪个通道)
    if (lane_id == 0)              // Print one line per warp
        printf("warp %d, first lane tid=%d\n", warp_id, tid);
}

int main() {
    warp_id_demo<<<1, 64>>>();
    cudaDeviceSynchronize();
    // Output: warp 0, first lane tid=0
    //         warp 1, first lane tid=32
}
```

### 2) warp_divergence_demo.cu (分支发散演示)

When threads (线程) in the same warp (线程束) take different branches, the warp executes **both paths sequentially** (顺序执行两条路径).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void divergent(int* out) {
    int tid = threadIdx.x;
    int val = 0;
    if (tid % 2 == 0) {         // Even lanes (偶数通道)
        val = tid * 10;          // Path A
    } else {                     // Odd lanes (奇数通道)
        val = tid * 100;         // Path B — runs after A in same warp
    }
    out[tid] = val;
}

int main() {
    int *d_out, h_out[8];
    cudaMalloc(&d_out, 8 * sizeof(int));
    divergent<<<1, 8>>>(d_out);
    cudaMemcpy(h_out, d_out, 8 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; i++) printf("%d ", h_out[i]);
    // Output: 0 100 20 300 40 500 60 700
    cudaFree(d_out);
}
```

### 3) warp_reduce_sum.cu (Warp 级归约)

A warp reduction (Warp 级归约) sums 32 values inside one warp (线程束) using `__shfl_down_sync` — **no shared memory** (无共享内存) needed.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__device__ float warp_reduce_sum(float val) {
    // Butterfly reduction (蝶式归约) within warp
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // Lane 0 holds the final sum (最终和)
}

__global__ void warp_sum(const float* in, float* out) {
    int tid = threadIdx.x;
    float v = in[tid];
    v = warp_reduce_sum(v);
    if (tid == 0) *out = v;
}

int main() {
    float h_in[32], h_out;
    for (int i = 0; i < 32; i++) h_in[i] = 1.0f;  // Sum = 32
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32 * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice);
    warp_sum<<<1, 32>>>(d_in, d_out);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("warp sum = %.1f\n", h_out);
    // Output: warp sum = 32.0
    cudaFree(d_in); cudaFree(d_out);
}
```

### 4) block_reduce_sum.cu (Block 级归约)

A block reduction (Block 级归约) combines warp-level reductions (Warp 级归约) via shared memory (共享内存) to sum **a whole block** (整个线程块).

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void block_reduce(const float* in, float* out, int n) {
    __shared__ float sdata[32];                  // One slot per warp (每 warp 一格)
    int tid = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    float v = (tid < n) ? in[tid] : 0.0f;
    v = warp_reduce_sum(v);                       // Step 1: warp reduce
    if (lane == 0) sdata[wid] = v;                // Lane 0 writes warp result
    __syncthreads();                              // Block sync (块同步)
    v = (tid < blockDim.x / 32) ? sdata[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);         // Step 2: reduce warp results
    if (tid == 0) *out = v;
}

int main() {
    const int n = 256;
    float *h_in = new float[n], h_out;
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;   // Sum = 256
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    block_reduce<<<1, 256>>>(d_in, d_out, n);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("block sum = %.1f\n", h_out);
    // Output: block sum = 256.0
    delete[] h_in; cudaFree(d_in); cudaFree(d_out);
}
```

### 5) shuffle_reduce.cu (Shuffle 归约对比)

This compares shared-memory reduction (共享内存归约) against shuffle-based reduction (Shuffle 归约) — shuffle wins by saving shared memory and avoiding `__syncthreads()`.

```cpp
#include <cuda_runtime.h>
#include <cstdio>

// Shuffle-based: no shared memory needed (无需共享内存)
__global__ void reduce_shuffle(const float* in, float* out) {
    int tid = threadIdx.x;
    float v = in[tid];
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if (tid == 0) *out = v;
}

int main() {
    float h_in[32], h_out;
    for (int i = 0; i < 32; i++) h_in[i] = (float)(i + 1);  // Sum = 528
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32 * sizeof(float)); cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice);
    reduce_shuffle<<<1, 32>>>(d_in, d_out);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("shuffle sum = %.1f\n", h_out);
    // Output: shuffle sum = 528.0
    cudaFree(d_in); cudaFree(d_out);
}
```

<br>

## 4. Must-Know Questions

### 1) Why does Warp Divergence (线程束分支发散) reduce efficiency?

Because the warp (线程束) must execute each branch path sequentially with the inactive lanes (非活跃通道) masked off, so an N-way divergence costs N times the latency (延迟).

### 2) What does `__syncthreads()` synchronize?

It synchronizes **all threads (线程) in the same block (线程块)** — not the whole grid (网格), and not just one warp (线程束).

### 3) Why does `__shfl_sync` reduce Shared Memory (共享内存) usage?

Because data is exchanged directly between registers (寄存器) within a warp (线程束) in one cycle, eliminating the round-trip through shared memory (共享内存) and the associated `__syncthreads()`.

### 4) How do you write a Warp-level Reduction (Warp 级归约)?

Use `__shfl_down_sync` in a halving loop with offsets 16 → 8 → 4 → 2 → 1, accumulating the value, so lane 0 (通道 0) ends up with the sum of all 32 lanes.

### 5) Why can't you put `__syncthreads()` inside a divergent branch?

Because `__syncthreads()` requires **every thread (线程) in the block** to reach it; if some threads skip the branch, the barrier (屏障) waits forever and the kernel deadlocks (死锁).

<br>

## 5. Stage Outcome (阶段成果)

You should now be able to write a warp reduction (Warp 级归约) using `__shfl_down_sync`:

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

The key insight (核心洞察) is that the GPU schedules work **per warp (线程束), not per thread (线程)** — writing kernels with this in mind unlocks divergence-free (无分支发散) and shared-memory-free (无共享内存) reductions.

<br>
