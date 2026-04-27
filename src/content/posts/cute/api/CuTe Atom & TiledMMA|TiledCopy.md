---
title: "CuTe Atom & TiledMMA|TiledCopy"
published: 2026-04-27
description: "CuTe Atom & TiledMMA|TiledCopy"
image: ""
tags: ["cute","api","CuTe Atom & TiledMMA|TiledCopy"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T20:34:29.683.170473931Z"
---

# CuTe Atom & TiledMMA/TiledCopy (原子与分块操作)

Atoms (原子) describe a single hardware instruction (单条硬件指令); TiledMMA / TiledCopy compose multiple atoms (多个原子) into a thread-block-level (线程块级) operation.

## 1. `Copy_Atom<Op, T>`

`Copy_Atom<Op, T>` wraps a copy instruction (拷贝指令) with the element type (元素类型), defining the unit (单位) of data movement (数据移动).

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
using namespace cute;

int main() {
    using Atom = Copy_Atom<UniversalCopy<float>, float>;

    print("Copy_Atom value layout: ");
    print(typename Atom::ValLayoutSrc{});
    print("\n");
    print("threads needed: ");
    print(typename Atom::ThrID{});
    print("\n");
    // Output:
    // Copy_Atom value layout: (_1,_1):(_0,_0)
    // threads needed: _1:_0
}
```

<br>

## 2. `make_tiled_copy(atom, thr_layout, val_layout)`

`make_tiled_copy` builds a tiled copy (构造分块拷贝) by tiling atoms (分块原子) across threads (线程) and values (值) — the standard pattern for gmem→smem loads.

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
using namespace cute;

int main() {
    using Atom = Copy_Atom<UniversalCopy<float>, float>;
    Atom atom;

    // 16 threads in 4x4 grid, each loads 1 float -> 4x4 tile
    auto tiled_copy = make_tiled_copy(
        atom,
        Layout<Shape<_4, _4>>{},                         // thread layout
        Layout<Shape<_1, _1>>{}                          // values per thread
    );

    print("tiled_copy tile shape: ");
    print(tile_shape(tiled_copy));
    print("\n");
    // Output:
    // tiled_copy tile shape: (_4,_4)
}
```

<br>

## 3. `tiled_copy.get_thread_slice(tid)`

This returns a per-thread (每线程) copy slice (拷贝切片), then `partition_S` (source) and `partition_D` (destination) split the tensors.

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
using namespace cute;

__global__ void slice_copy_kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    using Atom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(Atom{},
                          Layout<Shape<_4,_4>>{},
                          Layout<Shape<_1,_1>>{});

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto tg = thr_copy.partition_S(gA);                  // source slice
    auto ts = thr_copy.partition_D(sA);                  // destination slice
    copy(tiled_copy, tg, ts);
    __syncthreads();

    if (threadIdx.x == 0) printf("sA(2,3)=%.0f\n", float(sA(2, 3)));
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    slice_copy_kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA(2,3)=11
    cudaFree(d);
}
```

<br>

## 4. `SM80_CP_ASYNC_CACHEALWAYS<T>`

This is the Ampere `cp.async` instruction (Ampere `cp.async` 指令), which copies (拷贝) gmem→smem asynchronously (异步), bypassing registers (绕过寄存器) for higher throughput (更高吞吐量).

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm80.hpp>
using namespace cute;

__global__ void cp_async_kernel(float* gmem) {
    __shared__ float smem[16];

    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    auto gA = make_tensor(make_gmem_ptr(gmem), layout);
    auto sA = make_tensor(make_smem_ptr(smem), layout);

    using Atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>;
    auto tiled_copy = make_tiled_copy(Atom{},
                          Layout<Shape<_4,_4>>{},
                          Layout<Shape<_1,_1>>{});

    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    copy(tiled_copy,
         thr_copy.partition_S(gA),
         thr_copy.partition_D(sA));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    if (threadIdx.x == 0) printf("sA(1,3)=%.0f\n", float(sA(1, 3)));
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    cp_async_kernel<<<1, 16>>>(d);
    cudaDeviceSynchronize();
    // Output:
    // sA(1,3)=7
    cudaFree(d);
}
```

<br>

## 5. `UniversalCopy<T>`

`UniversalCopy<T>` is the generic copy atom (通用拷贝原子) that works on all architectures (所有架构), used as a portable fallback (可移植回退).

```cpp
#include <cute/atom/copy_atom.hpp>
using namespace cute;

int main() {
    using Atom32 = Copy_Atom<UniversalCopy<float>,    float>;
    using Atom64 = Copy_Atom<UniversalCopy<uint64_t>, uint64_t>;
    using Atom128 = Copy_Atom<UniversalCopy<uint4>,   uint4>;     // 16-byte

    print("Atom32 value type: float (4B)\n");
    print("Atom64 value type: uint64 (8B)\n");
    print("Atom128 value type: uint4 (16B)\n");
    // Output:
    // Atom32 value type: float (4B)
    // Atom64 value type: uint64 (8B)
    // Atom128 value type: uint4 (16B)
}
```

<br>

## 6. `AutoVectorizingCopy`

`AutoVectorizingCopyWithAssumedAlignment` automatically picks vectorized loads (向量化加载) like 128-bit `LDG.128` based on alignment (对齐), maximizing memory bandwidth (内存带宽).

```cpp
#include <cute/atom/copy_atom.hpp>
using namespace cute;

int main() {
    // 128-bit aligned loads
    using AlignedAtom = Copy_Atom<
        AutoVectorizingCopyWithAssumedAlignment<128>,
        float>;

    print("AutoVectorizingCopy with 128-bit alignment\n");
    print("- emits LDG.128 / STG.128 when shape allows\n");
    print("- falls back to smaller widths when not aligned\n");
    // Output:
    // AutoVectorizingCopy with 128-bit alignment
    // - emits LDG.128 / STG.128 when shape allows
    // - falls back to smaller widths when not aligned
}
```

<br>

## 7. `MMA_Traits<Op>`

`MMA_Traits<Op>` exposes metadata (元数据) about an MMA op (MMA 操作): shape, thread count, layouts of A/B/C — used by `MMA_Atom` internally.

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    using Op = SM80_16x8x16_F16F16F16F16_TN;
    using T  = MMA_Traits<Op>;

    print("Shape_MNK: "); print(typename T::Shape_MNK{}); print("\n");
    print("ThrID    : "); print(typename T::ThrID{});    print("\n");
    print("ALayout  : "); print(typename T::ALayout{});  print("\n");
    print("BLayout  : "); print(typename T::BLayout{});  print("\n");
    print("CLayout  : "); print(typename T::CLayout{});  print("\n");
    // Output (Shape_MNK = (16,8,16), ThrID = 32 threads):
    // Shape_MNK: (_16,_8,_16)
    // ThrID    : _32:_1
    // ALayout  : ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
    // BLayout  : ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
    // CLayout  : ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
}
```

<br>

## 8. `MMA_Atom<Op>`

`MMA_Atom<Op>` is the wrapper (包装器) used by `make_tiled_mma` — it inherits from `MMA_Traits<Op>` and exposes a uniform interface (统一接口).

```cpp
#include <cute/atom/mma_atom.hpp>
using namespace cute;

int main() {
    using Op   = SM80_16x8x16_F16F16F16F16_TN;
    using Atom = MMA_Atom<Op>;

    print("Atom Shape_MNK: "); print(typename Atom::Shape_MNK{}); print("\n");
    print("Atom ThrID:     "); print(typename Atom::ThrID{});    print("\n");
    print("Note: MMA_Atom = thin wrapper over MMA_Traits<Op>\n");
    // Output:
    // Atom Shape_MNK: (_16,_8,_16)
    // Atom ThrID:     _32:_1
    // Note: MMA_Atom = thin wrapper over MMA_Traits<Op>
}
```

<br> <br>
