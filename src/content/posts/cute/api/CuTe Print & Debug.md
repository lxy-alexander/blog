---
title: "CuTe Print & Debug"
published: 2026-04-27
description: "CuTe Print & Debug"
image: ""
tags: ["cute","api","CuTe Print & Debug"]
category: cute / api
draft: false
lang: ""
createdAt: "2026-04-27T20:36:09.850.495028341Z"
---

# CuTe Print & Debug (打印与调试)

CuTe provides rich printing (丰富的打印) for layouts (布局), tensors (张量), and tiled operations (分块操作) — essential for debugging (调试) layout bugs (布局错误).

## 1. `print(layout)`

`print(layout)` outputs the layout in text form (文本形式) `(Shape):(Stride)`, the most basic debugging tool (最基础的调试工具).

```cpp
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
    auto l1 = make_layout(make_shape(_4{}, _8{}),
                          make_stride(_8{}, _1{}));
    auto l2 = make_layout(make_shape(make_shape(_2{}, _4{}), _8{}));

    print("simple : "); print(l1); print("\n");
    print("nested : "); print(l2); print("\n");
    // Output:
    // simple : (_4,_8):(_8,_1)
    // nested : ((_2,_4),_8):((_1,_2),_8)
}
```

<br>

## 2. `print_tensor(tensor)`

`print_tensor(tensor)` prints the actual values (实际值) of a tensor in a 2D-formatted table (二维表格), used to inspect data (检查数据).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    float data[12];
    for (int i = 0; i < 12; ++i) data[i] = float(i);

    auto t = make_tensor(data, make_layout(make_shape(_3{}, _4{}),
                                            make_stride(_4{}, _1{})));
    print_tensor(t);
    // Output:
    // ptr[32b](0x...) o (_3,_4):(_4,_1):
    //     0     1     2     3
    //     4     5     6     7
    //     8     9    10    11
}
```

<br>

## 3. `print_layout(layout)`

`print_layout(layout)` visualizes the layout (可视化布局) as a 2D grid of offsets (偏移网格), making strides (步长) and tiling (分块) intuitive to read.

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_4{}, _4{}),
                              make_stride(_4{}, _1{}));
    print_layout(layout);
    // Output (offsets in 4x4 grid):
    // (_4,_4):(_4,_1)
    //       0    1    2    3
    //     +----+----+----+----+
    //  0  |  0 |  1 |  2 |  3 |
    //     +----+----+----+----+
    //  1  |  4 |  5 |  6 |  7 |
    //     +----+----+----+----+
    //  2  |  8 |  9 | 10 | 11 |
    //     +----+----+----+----+
    //  3  | 12 | 13 | 14 | 15 |
    //     +----+----+----+----+
}
```

<br>

## 4. `print_latex(layout)`

`print_latex(layout)` outputs a LaTeX/TikZ visualization (LaTeX 可视化) of the layout, ideal for documentation (文档) and academic papers (学术论文).

```cpp
#include <cute/tensor.hpp>
using namespace cute;

int main() {
    auto layout = make_layout(make_shape(_2{}, _4{}),
                              make_stride(_4{}, _1{}));
    print_latex(layout);
    // Output (snippet):
    // \documentclass[convert]{standalone}
    // \usepackage{tikz}
    // \begin{document}
    // \begin{tikzpicture}[...]
    //   \node at (0,0) {0}; \node at (1,0) {1}; ...
    // \end{tikzpicture}
    // \end{document}
    //
    // -> Compile with pdflatex to get a publication-quality figure
}
```

<br> <br>
