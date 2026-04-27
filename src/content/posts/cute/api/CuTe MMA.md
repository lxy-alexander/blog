---
title: "CuTe MMA"
published: 2026-04-27
description: "CuTe MMA"
image: ""
tags: ["cute","api","CuTe MMA"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T20:33:09.662.055627643Z"
---

# CuTe MMA (Matrix Multiply Accumulate, 矩阵乘累加)

MMA (矩阵乘累加) maps to Tensor Cores (张量核心), the source (源头) of FLOPs (浮点运算) on Ampere/Hopper GPUs — these APIs are the **highest difficulty (最高难度)** in CuTe.

## 1. `gemm(mma, A, B, C)` — Basic GEMM

`gemm(mma, A, B, C)` computes $$C = A \cdot B + C$$ on register tensors (寄存器张量) using a TiledMMA (分块 MMA) instance.

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

__global__ void mma_kernel(float* C_out) {
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;        // Ampere fp16 MMA
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op>,
                              Layout<Shape<_1, _1, _1>>>;
    TiledMMA tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // allocate fragments per thread
    auto rA = thr_mma.partition_fragment_A(make_layout(Shape<_16, _16>{}));
    auto rB = thr_mma.partition_fragment_B(make_layout(Shape<_8,  _16>{}));
    auto rC = thr_mma.partition_fragment_C(make_layout(Shape<_16, _8 >{}));

    // initialize: A = 1.0, B = 1.0, C = 0
    for (int i = 0; i < size(rA); ++i) rA(i) = __float2half(1.0f);
    for (int i = 0; i < size(rB); ++i) rB(i) = __float2half(1.0f);
    clear(rC);

    gemm(tiled_mma, rA, rB, rC);                        // C = A*B + 0

    if (threadIdx.x == 0)
        printf("rC(0) = %.1f (expect 16.0)\n", __half2float(rC(0)));
}

int main() {
    float* d; cudaMalloc(&d, sizeof(float));
    mma_kernel<<<1, 32>>>(d);                            // 1 warp = 32 threads
    cudaDeviceSynchronize();
    // Output:
    // rC(0) = 16.0 (expect 16.0)   — 16 = K dimension sum of 1*1
    cudaFree(d);
}
```

<br>

## 2. `gemm(mma, alpha, A, B, beta, C)` — Scaled GEMM

The scaled overload (缩放重载) computes $$C = \alpha A B + \beta C$$, matching cuBLAS GEMM semantics (语义).

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

__global__ void scaled_gemm_kernel() {
    using MMA_Op   = SM80_16x8x16_F16F16F16F16_TN;
    using TiledMMA = TiledMMA<MMA_Atom<MMA_Op>, Layout<Shape<_1,_1,_1>>>;
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto rA = thr_mma.partition_fragment_A(make_layout(Shape<_16,_16>{}));
    auto rB = thr_mma.partition_fragment_B(make_layout(Shape<_8, _16>{}));
    auto rC = thr_mma.partition_fragment_C(make_layout(Shape<_16,_8 >{}));

    for (int i = 0; i < size(rA); ++i) rA(i) = __float2half(1.0f);
    for (int i = 0; i < size(rB); ++i) rB(i) = __float2half(1.0f);
    for (int i = 0; i < size(rC); ++i) rC(i) = __float2half(2.0f);   // C init

    half alpha = __float2half(2.0f), beta = __float2half(3.0f);
    gemm(tiled_mma, alpha, rA, rB, beta, rC);            // 2*16 + 3*2 = 38

    if (threadIdx.x == 0)
        printf("rC(0) = %.1f (expect 38.0)\n", __half2float(rC(0)));
}

int main() {
    scaled_gemm_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    // Output:
    // rC(0) = 38.0 (expect 38.0)
}
```

<br>

## 3. `MMA_Atom<MMA_Op>`

`MMA_Atom<MMA_Op>` wraps a single hardware MMA instruction (单条硬件 MMA 指令); the `MMA_Op` selects the architecture/dtype variant (架构/数据类型变体).

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    using Op   = SM80_16x8x16_F16F16F16F16_TN;
    using Atom = MMA_Atom<Op>;

    print("MMA tile: ");
    print(typename Atom::Shape_MNK{});                   // (16, 8, 16)
    print("\n");
    // Output:
    // MMA tile: (_16,_8,_16)
}
```

<br>

## 4. `SM80_16x8x16_F16F16F16F16_TN` — Ampere fp16 MMA

This is the canonical Ampere fp16 MMA op (典型安培 fp16 MMA), with tile shape (分块形状) $$M{=}16, N{=}8, K{=}16$$ and TN layout (TN 布局: A row-major, B col-major).

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    using Op = SM80_16x8x16_F16F16F16F16_TN;
    using T  = MMA_Traits<Op>;

    print("Shape  M,N,K: "); print(typename T::Shape_MNK{}); print("\n");
    print("ThrID warp  : "); print(typename T::ThrID{});    print("\n");
    // Output:
    // Shape  M,N,K: (_16,_8,_16)
    // ThrID warp  : _32:_1     (32 threads = 1 warp)
}
```

<br>

## 5. `make_tiled_mma`

`make_tiled_mma(atom, atom_layout)` tiles MMA atoms (分块 MMA 原子) across multiple warps (多个 warp), scaling up MMA throughput (吞吐量).

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    using Op   = SM80_16x8x16_F16F16F16F16_TN;
    using Atom = MMA_Atom<Op>;

    // tile MMA atoms in a 2x2 warp grid -> uses 4 warps (128 threads)
    auto tiled_mma = make_tiled_mma(Atom{}, Layout<Shape<_2, _2, _1>>{});

    print("tiled_mma tile MNK: ");
    print(tile_shape(tiled_mma)); print("\n");          // (32, 16, 16)
    print("threads needed: %d\n", int(size(tiled_mma)));
    // Output:
    // tiled_mma tile MNK: (_32,_16,_16)
    // threads needed: 128
}
```

<br>

## 6. `tiled_mma.get_thread_slice`

`tiled_mma.get_thread_slice(tid)` returns a per-thread (每线程) view (视图) of the MMA, used to allocate (分配) the right fragments (片段) for that thread.

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

__global__ void slice_kernel() {
    using Op       = SM80_16x8x16_F16F16F16F16_TN;
    using TiledMMA = TiledMMA<MMA_Atom<Op>, Layout<Shape<_1,_1,_1>>>;
    TiledMMA tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rC = thr_mma.partition_fragment_C(make_layout(Shape<_16,_8>{}));

    if (threadIdx.x == 0) {
        printf("thread 0 holds %d C elements\n", int(size(rC)));
    }
}

int main() {
    slice_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    // Output:
    // thread 0 holds 4 C elements   (16*8 = 128, distributed across 32 threads)
}
```

<br>

## 7. `partition_fragment_A / _B / _C`

These methods (方法) allocate (分配) per-thread A/B/C register fragments (寄存器片段) sized exactly for the MMA instruction's needs (MMA 指令需求).

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

__global__ void frag_kernel() {
    using Op       = SM80_16x8x16_F16F16F16F16_TN;
    using TiledMMA = TiledMMA<MMA_Atom<Op>, Layout<Shape<_1,_1,_1>>>;
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto rA = thr_mma.partition_fragment_A(make_layout(Shape<_16,_16>{}));
    auto rB = thr_mma.partition_fragment_B(make_layout(Shape<_8, _16>{}));
    auto rC = thr_mma.partition_fragment_C(make_layout(Shape<_16,_8 >{}));

    if (threadIdx.x == 0) {
        printf("|rA|=%d |rB|=%d |rC|=%d\n",
               int(size(rA)), int(size(rB)), int(size(rC)));
    }
}

int main() {
    frag_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    // Output:
    // |rA|=8 |rB|=4 |rC|=4   (per-thread fragment sizes)
}
```

<br>

## 8. `SM90_64x128x16_F32F16F16_SS` — Hopper WGMMA

On Hopper (H100), WGMMA (warpgroup MMA, warp 组 MMA) is async (异步) and operates on 128 threads (4 warps), with much larger tiles (更大分块) than Ampere.

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    // Hopper WGMMA — note: requires sm_90 to actually launch
    using Op = SM90_64x128x16_F16F16F32_SS<GMMA::Major::K, GMMA::Major::K>;
    using T  = MMA_Traits<Op>;

    print("WGMMA Shape_MNK: ");
    print(typename T::Shape_MNK{});
    print("\n");
    print("threads needed (warpgroup): ");
    print(typename T::ThrID{});
    print("\n");
    // Output:
    // WGMMA Shape_MNK: (_64,_128,_16)
    // threads needed (warpgroup): _128:_1
}
```

<br> <br>
