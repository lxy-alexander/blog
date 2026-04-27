---
title: "CuTe Tensor Construction"
published: 2026-04-27
description: "CuTe Tensor Construction"
image: ""
tags: ["cute","api","CuTe Tensor Construction"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T17:36:44.672.369269503Z"
---
# CuTe Tensor Construction (张量构造)

A `Tensor` (张量) in CuTe is a pointer (指针) plus a Layout (布局) — it is a logical view (逻辑视图) of memory and does not own data (不拥有数据).

## 1. `make_tensor(ptr, layout)`

`make_tensor(ptr, layout)` wraps a raw pointer (原始指针) with a layout (布局) to form a Tensor (张量) view.

```cpp
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
    float data[16];
    for (int i = 0; i < 16; ++i) data[i] = float(i);

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));   // row-major 4x4
    auto tensor = make_tensor(data, layout);

    print("tensor(0,0) = %.0f\n", float(tensor(0, 0)));
    print("tensor(1,2) = %.0f\n", float(tensor(1, 2)));
    print("tensor(3,3) = %.0f\n", float(tensor(3, 3)));
    // Output:
    // tensor(0,0) = 0
    // tensor(1,2) = 6     (row 1, col 2 -> offset 6)
    // tensor(3,3) = 15
}
```

<br>

## 2. `make_tensor(layout)` — Register Tensor

`make_tensor(layout)` without a pointer (无指针) allocates a register tensor (寄存器张量) on the stack (栈), used for thread-local accumulators (线程本地累加器).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // pure register tensor (no external memory)
    auto rC = make_tensor<float>(make_shape(_2{}, _4{}));
    clear(rC);                                           // zero-fill

    rC(0, 0) = 1.0f;
    rC(1, 2) = 7.5f;

    print("rC(0,0) = %.1f\n", rC(0, 0));
    print("rC(1,2) = %.1f\n", rC(1, 2));
    print("rC(0,3) = %.1f\n", rC(0, 3));   // still zero
    // Output:
    // rC(0,0) = 1.0
    // rC(1,2) = 7.5
    // rC(0,3) = 0.0
}
```

<br>

## 3. `make_gmem_ptr`

`make_gmem_ptr(ptr)` tags a pointer as global memory (全局内存), so CuTe can pick correct instructions (正确指令) like `cp.async` for async copy.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    auto gptr   = make_gmem_ptr(gmem);                   // tagged as gmem
    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA     = make_tensor(gptr, layout);

    if (threadIdx.x == 0) {
        printf("gA(0,0) = %.0f, gA(1,1) = %.0f\n",
               float(gA(0, 0)), float(gA(1, 1)));
    }
}

int main() {
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // gA(0,0) = 0, gA(1,1) = 5
    cudaFree(d);
}
```

<br>

## 4. `make_smem_ptr`

`make_smem_ptr(ptr)` tags a pointer as shared memory (共享内存), enabling shared-memory-specific (共享内存特有) optimizations like swizzling (混洗).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel() {
    __shared__ float smem[16];
    if (threadIdx.x < 16) smem[threadIdx.x] = float(threadIdx.x);
    __syncthreads();

    auto sptr   = make_smem_ptr(smem);                   // tagged as smem
    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto sA     = make_tensor(sptr, layout);

    if (threadIdx.x == 0) {
        printf("sA(2,1) = %.0f\n", float(sA(2, 1)));
    }
}

int main() {
    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    // Output:
    // sA(2,1) = 9   (2*4 + 1)
}
```

<br>

## 5. `make_fragment_like`

`make_fragment_like(tensor)` allocates a register fragment (寄存器片段) with the same shape (相同形状) and dtype (数据类型) as the source — common pattern for accumulators (累加器).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[8];
    auto gA = make_tensor(data, make_layout(make_shape(_2{}, _4{})));

    // allocate a register fragment with same shape as gA
    auto rA = make_fragment_like(gA);
    clear(rA);
    rA(0, 0) = 99.0f;

    print("rA shape: "); print(shape(rA)); print("\n");
    print("rA(0,0) = %.0f\n", rA(0, 0));
    // Output:
    // rA shape: (_2,_4)
    // rA(0,0) = 99
}
```

<br>

## 6. `recast` — Reinterpret Element Type

`recast<T>(tensor)` reinterprets (重解释) a tensor's element type (元素类型), e.g. viewing 4 fp16 values as one uint64 — used in vectorized loads (向量化加载).

```cpp
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
using namespace cute;

int main() {
    half data[8];
    for (int i = 0; i < 8; ++i) data[i] = __float2half(float(i));

    auto t_half = make_tensor(data, make_layout(make_shape(_8{})));
    auto t_int  = recast<uint32_t>(t_half);              // 8 halves -> 4 uint32

    print("t_half size: %d\n", int(size(t_half)));
    print("t_int  size: %d\n", int(size(t_int)));
    // Output:
    // t_half size: 8
    // t_int  size: 4
}
```

<br>
<br>