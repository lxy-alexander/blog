---
title: "CuTe Algorithm Helpers"
published: 2026-04-27
description: "CuTe Algorithm Helpers"
image: ""
tags: ["cute","api","CuTe Algorithm Helpers"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T20:35:00.322.501239671Z"
---

# CuTe Algorithm Helpers (算法辅助)

These helpers (辅助函数) handle common patterns (常见模式) like fill (填充), clear (清零), and axpby (axpby 运算) — convenient one-liners (一行代码) over `for_each`.

## 1. `clear(tensor)`

`clear(tensor)` zero-fills (零填充) every element of a tensor — most commonly used for accumulators (累加器) before MMA.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto t = make_tensor(data, make_layout(make_shape(_2{}, _4{})));

    clear(t);

    print("after clear: ");
    for (int i = 0; i < 8; ++i) print("%.0f ", data[i]);
    print("\n");
    // Output:
    // after clear: 0 0 0 0 0 0 0 0
}
```

<br>

## 2. `fill(tensor, value)`

`fill(tensor, value)` fills every element with a given constant (常量), used for biases (偏置) or initial values (初始值).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[8] = {0};
    auto t = make_tensor(data, make_layout(make_shape(_2{}, _4{})));

    fill(t, 3.14f);

    print("after fill: ");
    for (int i = 0; i < 8; ++i) print("%.2f ", data[i]);
    print("\n");
    // Output:
    // after fill: 3.14 3.14 3.14 3.14 3.14 3.14 3.14 3.14
}
```

<br>

## 3. `axpby(a, X, b, Y)`

`axpby(a, X, b, Y)` computes $$Y = aX + bY$$ element-wise (逐元素), a building block (基础块) of GEMM epilogues (GEMM 尾部) and BLAS-style ops.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float Xd[4] = {1, 2, 3, 4};
    float Yd[4] = {10, 20, 30, 40};
    auto X = make_tensor(Xd, make_layout(make_shape(_4{})));
    auto Y = make_tensor(Yd, make_layout(make_shape(_4{})));

    axpby(2.0f, X, 0.5f, Y);                // Y = 2*X + 0.5*Y

    print("Y: ");
    for (int i = 0; i < 4; ++i) print("%.1f ", Yd[i]);
    print("\n");
    // Output:
    // Y: 7.0 14.0 21.0 28.0
    // 2*1 + 0.5*10 = 7, 2*2 + 0.5*20 = 14, ...
}
```

<br>

## 4. `transform(src, dst, fn)`

`transform(src, dst, fn)` applies an element-wise function (逐元素函数) `dst[i] = fn(src[i])`, used for activations (激活) like ReLU or scaling (缩放).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float Xd[4] = {-1, 2, -3, 4};
    float Yd[4] = {0};
    auto X = make_tensor(Xd, make_layout(make_shape(_4{})));
    auto Y = make_tensor(Yd, make_layout(make_shape(_4{})));

    // ReLU: max(x, 0)
    transform(X, Y, [] (float x) { return x > 0.f ? x : 0.f; });

    print("Y after ReLU: ");
    for (int i = 0; i < 4; ++i) print("%.1f ", Yd[i]);
    print("\n");
    // Output:
    // Y after ReLU: 0.0 2.0 0.0 4.0
}
```

<br>

## 5. `for_each(tensor, fn)`

`for_each(tensor, fn)` iterates over every element (遍历每个元素), useful for custom side effects (副作用) like printing or accumulating (累加).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float Xd[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    auto X = make_tensor(Xd, make_layout(make_shape(_4{})));

    float total = 0.0f;
    for_each(X, [&] (float v) { total += v; });

    print("sum = %.1f\n", total);
    // Output:
    // sum = 12.0
}
```

<br>

## 6. `cosize(layout)`

`cosize(layout)` returns the codomain size (值域大小) — the smallest memory range (最小内存范围) the layout actually addresses, often `> size` for strided layouts (跨步布局).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    // strided layout: 4 elements with stride 8 -> last offset is 24
    auto layout = make_layout(make_shape(_4{}), make_stride(_8{}));

    print("size   = %d\n", int(size(layout)));         // 4
    print("cosize = %d\n", int(cosize(layout)));       // 25 (offsets: 0,8,16,24)
    // Output:
    // size   = 4
    // cosize = 25
}
```

<br> <br>
