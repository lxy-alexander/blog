---
title: "CuTe Tiling & Partitioning"
published: 2026-04-27
description: "CuTe Tiling & Partitioning"
image: ""
tags: ["cute","api","CuTe Tiling & Partitioning"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T17:37:10.063.781054692Z"
---

# CuTe Tiling & Partitioning (分块与划分)

Tiling (分块) divides a large tensor (大张量) into smaller tiles (小分块), and partitioning (划分) assigns tiles to threads (线程) — the foundation (基础) of GPU parallelism (GPU 并行).

## 1. `local_tile`

`local_tile(tensor, tile_shape, coord)` extracts one tile (提取一个分块) from a larger tensor at a given block coordinate (块坐标), used to give each thread block (线程块) its slice (切片).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // 8x8 source tensor (row-major)
    float data[64];
    for (int i = 0; i < 64; ++i) data[i] = float(i);
    auto A = make_tensor(data, make_layout(make_shape(_8{}, _8{}),
                                            make_stride(_8{}, _1{})));

    // tile shape 4x4, take block (1, 0) -> rows 4..7, cols 0..3
    auto tile = local_tile(A, make_shape(_4{}, _4{}), make_coord(1, 0));

    print("tile(0,0) = %.0f\n", tile(0, 0));     // should be 32
    print("tile(0,3) = %.0f\n", tile(0, 3));     // 35
    print("tile(3,0) = %.0f\n", tile(3, 0));     // 56
    // Output:
    // tile(0,0) = 32
    // tile(0,3) = 35
    // tile(3,0) = 56
}
```

<br>

## 2. `local_partition`

`local_partition(tensor, thr_layout, tid)` partitions (划分) a tile among threads (线程), giving each thread its element slice (元素切片).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

__global__ void kernel(float* gmem) {
    auto A = make_tensor(make_gmem_ptr(gmem),
                         make_layout(make_shape(_4{}, _4{}),
                                     make_stride(_4{}, _1{})));   // 4x4 tile

    // 4 threads laid out as 2x2; each thread handles a 2x2 sub-tile
    auto thr_layout = make_layout(make_shape(_2{}, _2{}));
    auto thr_A      = local_partition(A, thr_layout, threadIdx.x);

    if (threadIdx.x == 0) {
        printf("tid=0: thr_A(0,0)=%.0f thr_A(1,1)=%.0f\n",
               float(thr_A(0, 0)), float(thr_A(1, 1)));
    }
    if (threadIdx.x == 3) {
        printf("tid=3: thr_A(0,0)=%.0f thr_A(1,1)=%.0f\n",
               float(thr_A(0, 0)), float(thr_A(1, 1)));
    }
}

int main() {
    float h[16]; for (int i = 0; i < 16; ++i) h[i] = float(i);
    float* d; cudaMalloc(&d, 16 * sizeof(float));
    cudaMemcpy(d, h, 16 * sizeof(float), cudaMemcpyHostToDevice);

    kernel<<<1, 4>>>(d);
    cudaDeviceSynchronize();
    // Output (order may vary):
    // tid=0: thr_A(0,0)=0  thr_A(1,1)=5
    // tid=3: thr_A(0,0)=10 thr_A(1,1)=15
    cudaFree(d);
}
```

<br>

## 3. `tiled_divide`

`tiled_divide(tensor, tile)` divides a tensor (划分张量) into a 3D structure: `(TileShape, RestShape)`, exposing both intra-tile (块内) and inter-tile (块间) coordinates.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[64];
    for (int i = 0; i < 64; ++i) data[i] = float(i);
    auto A = make_tensor(data, make_layout(make_shape(_8{}, _8{}),
                                            make_stride(_8{}, _1{})));

    // divide 8x8 into 4x4 tiles -> 2x2 grid of tiles
    auto td = tiled_divide(A, make_shape(_4{}, _4{}));
    print("tiled_divide layout: "); print(layout(td)); print("\n");
    print("rank = %d\n", int(rank(td)));
    // td shape is ((4,4), 2, 2): inner tile + outer grid
    // Output:
    // tiled_divide layout: ((_4,_4),_2,_2):((_8,_1),_32,_4)
    // rank = 3
}
```

<br>

## 4. `flat_divide`

`flat_divide(tensor, tile)` flattens (展平) tile-shape and rest-shape modes into a single flat structure (单层平坦结构) — easier to index when modes don't need to stay nested.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[64];
    auto A = make_tensor(data, make_layout(make_shape(_8{}, _8{}),
                                            make_stride(_8{}, _1{})));

    auto fd = flat_divide(A, make_shape(_4{}, _4{}));
    print("flat_divide layout: "); print(layout(fd)); print("\n");
    print("rank = %d\n", int(rank(fd)));
    // 4 modes: tileM, tileN, restM, restN
    // Output:
    // flat_divide layout: (_4,_4,_2,_2):(_8,_1,_32,_4)
    // rank = 4
}
```

<br>

## 5. `zipped_divide`

`zipped_divide(tensor, tile)` zips tile dims together and rest dims together — produces 2 modes (`(TileDims..., RestDims...)`), commonly seen in matmul kernels.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[64];
    auto A = make_tensor(data, make_layout(make_shape(_8{}, _8{}),
                                            make_stride(_8{}, _1{})));

    auto zd = zipped_divide(A, make_shape(_4{}, _4{}));
    print("zipped_divide layout: "); print(layout(zd)); print("\n");
    print("rank = %d\n", int(rank(zd)));
    // shape: ((4,4),(2,2)) — tile group, rest group
    // Output:
    // zipped_divide layout: ((_4,_4),(_2,_2)):((_8,_1),(_32,_4))
    // rank = 2
}
```

<br>

## 6. `logical_divide`

`logical_divide(layout, tile)` performs the layout-level (布局级) division without touching data, returning a new layout (新布局) describing the tiled view (分块视图).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_8{}, _8{}),
                              make_stride(_8{}, _1{}));
    auto ld = logical_divide(layout, make_shape(_4{}, _4{}));
    print("logical_divide: "); print(ld); print("\n");
    // structure: ((TileM, RestM), (TileN, RestN))
    // Output:
    // logical_divide: ((_4,_2),(_4,_2)):((_8,_32),(_1,_4))
}
```

<br>

## 7. `composition`

`composition(A, B)` composes two layouts (布局复合): `A ∘ B` means apply `B` first, then `A` — used to remap (重映射) coordinates.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto A = make_layout(make_shape(_4{}, _4{}),     // 4x4 row-major
                         make_stride(_4{}, _1{}));
    auto B = make_layout(make_shape(_2{}, _2{}),     // pick a 2x2 sub-pattern
                         make_stride(_2{}, _1{}));
    auto C = composition(A, B);
    print("composition: "); print(C); print("\n");
    print("C(0,0)=%d C(1,0)=%d C(0,1)=%d C(1,1)=%d\n",
          int(C(0, 0)), int(C(1, 0)), int(C(0, 1)), int(C(1, 1)));
    // Output:
    // composition: (_2,_2):(_8,_1)
    // C(0,0)=0 C(1,0)=8 C(0,1)=1 C(1,1)=9
}
```

<br>

## 8. `tile_to_shape`

`tile_to_shape(tile, shape)` repeats a tile pattern (重复分块模式) until it covers a target shape (目标形状), used to build TiledMMA / TiledCopy patterns.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto tile  = make_layout(make_shape(_2{}, _2{}),
                             make_stride(_2{}, _1{}));
    auto big   = tile_to_shape(tile, make_shape(_4{}, _4{}));
    print("tiled layout: "); print(big); print("\n");
    print("size = %d\n", int(size(big)));
    // Output:
    // tiled layout: ((_2,_2),(_2,_2)):((_2,_1),(_8,_4))
    // size = 16
}
```

<br> <br>
