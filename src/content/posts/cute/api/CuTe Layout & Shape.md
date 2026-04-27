---
title: "CuTe Layout & Shape"
published: 2026-04-27
description: "CuTe Layout & Shape"
image: ""
tags: ["cute","api","CuTe Layout & Shape"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T17:36:16.490.992888430Z"
---

# CuTe Layout & Shape (布局与形状)

Layout (布局) is CuTe's core abstraction (核心抽象) — a function (函数) mapping logical coordinates (逻辑坐标) to memory offsets (内存偏移), defined by `Shape` (形状) and `Stride` (步长).

## 1. `Int<N>` and `_N` Constants

`Int<N>{}` and shortcuts like `_1`, `_2`, `_4` are compile-time integers (编译期整数), enabling the compiler to specialize (特化) and unroll loops (展开循环).

```cpp
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
    auto a = Int<8>{};                              // compile-time 8
    auto b = _16{};                                  // compile-time 16
    auto c = a * b;                                  // compile-time 128
    printf("a=%d b=%d c=%d\n", int(a), int(b), int(c));
    // Output:
    // a=8 b=16 c=128
}
```

<br>

## 2. `make_shape`

`make_shape(...)` creates a shape tuple (形状元组), which can mix compile-time (编译期) and runtime (运行时) values.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto s1 = make_shape(_4{}, _8{});                // static (4, 8)
    auto s2 = make_shape(4, 8);                      // dynamic (4, 8)
    auto s3 = make_shape(_4{}, 8);                   // mixed
    print("s1: "); print(s1); print("\n");
    print("s2: "); print(s2); print("\n");
    print("s3: "); print(s3); print("\n");
    // Output:
    // s1: (_4,_8)
    // s2: (4,8)
    // s3: (_4,8)
}
```

<br>

## 3. `make_stride`

`make_stride(...)` creates a stride tuple (步长元组) — the step size in memory between consecutive elements (相邻元素) along each mode (模态).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // row-major (行主序) for 4x8 matrix: stride (8, 1)
    auto row_major = make_stride(_8{}, _1{});
    // col-major (列主序) for 4x8 matrix: stride (1, 4)
    auto col_major = make_stride(_1{}, _4{});
    print("row_major: "); print(row_major); print("\n");
    print("col_major: "); print(col_major); print("\n");
    // Output:
    // row_major: (_8,_1)
    // col_major: (_1,_4)
}
```

<br>

## 4. `make_layout`

`make_layout(shape, stride)` builds a layout (布局) — the function (函数) `coord -> offset` that defines how a tensor maps to memory.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_4{}, _8{}),
                              make_stride(_8{}, _1{}));   // row-major 4x8
    print("layout: "); print(layout); print("\n");
    print("layout(0,0) = %d\n", int(layout(0, 0)));
    print("layout(1,0) = %d\n", int(layout(1, 0)));
    print("layout(0,1) = %d\n", int(layout(0, 1)));
    print("layout(2,3) = %d\n", int(layout(2, 3)));
    // Output:
    // layout: (_4,_8):(_8,_1)
    // layout(0,0) = 0
    // layout(1,0) = 8
    // layout(0,1) = 1
    // layout(2,3) = 19   (2*8 + 3*1)
}
```

<br>

## 5. `Layout<Shape, Stride>` Static Type

`Layout<Shape, Stride>` is a static layout type (静态布局类型), known fully at compile time (编译期完全已知) — fastest path with zero runtime cost (零运行时开销).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    using MyShape  = Shape<_4, _8>;
    using MyStride = Stride<_8, _1>;
    using MyLayout = Layout<MyShape, MyStride>;

    MyLayout layout{};
    print("static layout: "); print(layout); print("\n");
    print("offset(3,5) = %d\n", int(layout(3, 5)));
    // Output:
    // static layout: (_4,_8):(_8,_1)
    // offset(3,5) = 29   (3*8 + 5*1)
}
```

<br>

## 6. `size`

`size(layout)` returns the total number of elements (总元素数) — the product of all shape dimensions (所有形状维度).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_4{}, _8{}, _2{}));
    auto total = size(layout);                       // 4 * 8 * 2 = 64
    auto dim0  = size<0>(layout);                    // 4
    auto dim1  = size<1>(layout);                    // 8
    print("total = %d, dim0 = %d, dim1 = %d\n",
          int(total), int(dim0), int(dim1));
    // Output:
    // total = 64, dim0 = 4, dim1 = 8
}
```

<br>

## 7. `rank`

`rank(layout)` returns the number of modes (模态数), equivalent to the number of axes (轴数) in NumPy.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto l1 = make_layout(make_shape(_8{}));                // rank 1
    auto l2 = make_layout(make_shape(_4{}, _8{}));          // rank 2
    auto l3 = make_layout(make_shape(_2{}, _4{}, _8{}));    // rank 3
    print("rank(l1) = %d\n", int(rank(l1)));
    print("rank(l2) = %d\n", int(rank(l2)));
    print("rank(l3) = %d\n", int(rank(l3)));
    // Output:
    // rank(l1) = 1
    // rank(l2) = 2
    // rank(l3) = 3
}
```

<br>

## 8. `shape` and `stride` Accessors

`shape(layout)` and `stride(layout)` extract (提取) the shape and stride components, with `<I>` selecting a specific mode (特定模态).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_4{}, _8{}),
                              make_stride(_8{}, _1{}));
    print("shape:  "); print(shape(layout));  print("\n");
    print("stride: "); print(stride(layout)); print("\n");
    print("shape<0>:  %d\n", int(shape<0>(layout)));
    print("stride<1>: %d\n", int(stride<1>(layout)));
    // Output:
    // shape:  (_4,_8)
    // stride: (_8,_1)
    // shape<0>:  4
    // stride<1>: 1
}
```

<br>

## 9. Hierarchical Layout (Nested)

Layouts can be nested (嵌套) — a mode (模态) can itself be a sub-shape (子形状), enabling complex tile patterns (复杂分块模式) like (M, (BlockM, ThreadM)).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // hierarchical 2-level shape: outer 2x2, inner 4x4 -> total 8x8
    auto layout = make_layout(make_shape(make_shape(_2{}, _4{}),
                                         make_shape(_2{}, _4{})),
                              make_stride(make_stride(_4{},  _32{}),
                                          make_stride(_1{},  _8{})));
    print("nested layout: "); print(layout); print("\n");
    print("size = %d\n", int(size(layout)));
    print("rank = %d\n", int(rank(layout)));
    // Output:
    // nested layout: ((_2,_4),(_2,_4)):((_4,_32),(_1,_8))
    // size = 64
    // rank = 2
}
```

<br>

## 10. `make_layout` with Single Shape (Default Stride)

When only a shape is provided, `make_layout` defaults to col-major (列主序) compact stride (紧凑步长), useful for quick prototyping (快速原型).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_4{}, _8{}));   // default: col-major
    print("default layout: "); print(layout); print("\n");
    // strides: (_1, _4) -> col-major
    print("layout(1,0) = %d\n", int(layout(1, 0)));      // 1 (col-major)
    print("layout(0,1) = %d\n", int(layout(0, 1)));      // 4
    // Output:
    // default layout: (_4,_8):(_1,_4)
    // layout(1,0) = 1
    // layout(0,1) = 4
}
```

<br> <br>
