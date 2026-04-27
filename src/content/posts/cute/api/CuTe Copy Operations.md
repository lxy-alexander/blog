---
title: "CuTe Copy Operations"
published: 2026-04-27
description: "CuTe Copy Operations"
image: ""
tags: ["cute","api","CuTe Copy Operations"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T20:32:47.131.746054484Z"
---

# CuTe Copy Operations (拷贝操作)

`cute::copy` is the unified async/sync (统一异步/同步) data movement (数据移动) API, with optional `Atom` (原子操作) controlling the underlying hardware instruction (底层硬件指令).

## 1. `copy(src, dst)` — Generic Copy

`copy(src, dst)` is the generic copy (通用拷贝) operation — it picks a default copy atom (默认拷贝原子) based on memory spaces (内存空间).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    // each thread copies one element via local_partition
    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg = local_partition(gA, thr_layout, threadIdx.x);
    auto ts = local_partition(sA, thr_layout, threadIdx.x);
    copy(tg, ts);                                     // gmem -> smem
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("smem[0]=%.0f smem[5]=%.0f smem[15]=%.0f\n",
               smem[0], smem[5], smem[15]);
    }
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // smem[0]=0 smem[5]=5 smem[15]=15
    cudaFree(d);
}
```

<br>

## 2. `copy(atom, src, dst)` — Atom-Based Copy

Passing a `Copy_Atom` (拷贝原子) explicitly selects a hardware instruction (硬件指令) like `cp.async` or `LDGSTS`.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    // explicit copy atom: 4-byte universal copy
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    CopyAtom atom;

    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg = local_partition(gA, thr_layout, threadIdx.x);
    auto ts = local_partition(sA, thr_layout, threadIdx.x);
    copy(atom, tg, ts);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("sA(2,2) = %.0f\n", float(sA(2, 2)));
    }
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i * 2);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA(2,2) = 20    (index 10, value 20)
    cudaFree(d);
}
```

<br>

## 3. `copy_if(pred, src, dst)` — Predicated Copy

`copy_if(pred, src, dst)` performs a predicated copy (谓词拷贝), only copying elements where the predicate (谓词) is true — used for boundary handling (边界处理).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem, int N) {
    __shared__ float smem[16];
    for (int i = threadIdx.x; i < 16; i += blockDim.x) smem[i] = -1.0f;
    __syncthreads();

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg = local_partition(gA, thr_layout, threadIdx.x);
    auto ts = local_partition(sA, thr_layout, threadIdx.x);

    // predicate tensor: only copy if linear index < N
    auto pred = make_tensor<bool>(shape(tg));
    for (int i = 0; i < size(pred); ++i)
        pred(i) = (int(threadIdx.x) < N);
    copy_if(pred, tg, ts);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("smem[0]=%.0f smem[5]=%.0f smem[10]=%.0f\n",
               smem[0], smem[5], smem[10]);   // 10 unchanged (-1)
    }
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 16>>>(d, 8);   // only first 8 threads copy
    cudaDeviceSynchronize();
    // Output:
    // smem[0]=0 smem[5]=5 smem[10]=-1
    cudaFree(d);
}
```

<br>

## 4. `cp_async_fence`

`cp_async_fence()` inserts a fence (屏障) marking a group of in-flight (在飞) async copies, used as a checkpoint (检查点) before waiting.

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg = local_partition(gA, thr_layout, threadIdx.x);
    auto ts = local_partition(sA, thr_layout, threadIdx.x);

    copy(tg, ts);
    cp_async_fence();                                 // mark copy group
    cp_async_wait<0>();                               // wait all
    __syncthreads();

    if (threadIdx.x == 0) printf("sA(1,1)=%.0f\n", float(sA(1, 1)));
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA(1,1)=5
    cudaFree(d);
}
```

<br>

## 5. `cp_async_wait<N>`

`cp_async_wait<N>()` waits until all but `N` async copy groups (异步拷贝组) complete, enabling software pipelining (软件流水线) with multiple stages (多阶段).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void pipeline_kernel(float* gmem) {
    __shared__ float smem0[16], smem1[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA  = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA0 = make_tensor(make_smem_ptr(smem0), layout);
    auto sA1 = make_tensor(make_smem_ptr(smem1), layout);

    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg  = local_partition(gA,  thr_layout, threadIdx.x);
    auto ts0 = local_partition(sA0, thr_layout, threadIdx.x);
    auto ts1 = local_partition(sA1, thr_layout, threadIdx.x);

    copy(tg, ts0); cp_async_fence();   // group 0
    copy(tg, ts1); cp_async_fence();   // group 1

    cp_async_wait<1>();                // keep at most 1 in flight
    __syncthreads();
    if (threadIdx.x == 0) printf("sA0(2,2)=%.0f\n", float(sA0(2, 2)));

    cp_async_wait<0>();                // wait all
    __syncthreads();
    if (threadIdx.x == 0) printf("sA1(2,2)=%.0f\n", float(sA1(2, 2)));
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    pipeline_kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA0(2,2)=10
    // sA1(2,2)=10
    cudaFree(d);
}
```

<br>

## 6. `cp_async_wait_all`

`cp_async_wait_all()` waits for all (等待全部) outstanding async copies (未完成异步拷贝) to finish — equivalent to `cp_async_wait<0>()`.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    auto thr_layout = make_layout(make_shape(_4{}, _4{}));
    auto tg = local_partition(gA, thr_layout, threadIdx.x);
    auto ts = local_partition(sA, thr_layout, threadIdx.x);

    copy(tg, ts);
    cp_async_fence();
    cp_async_wait_all();                              // wait everything
    __syncthreads();

    if (threadIdx.x == 0) printf("sA(3,3)=%.0f\n", float(sA(3, 3)));
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA(3,3)=15
    cudaFree(d);
}
```

<br> <br>
