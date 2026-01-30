---
title: "OpenMP"
published: 2026-01-29
description: "OpenMP"
image: ""
tags: ["c++","OpenMP"]
category: c++
draft: false
lang: ""
---

# 多线程并行编程（parallel programming）

在 C/C++ / Fortran 里，**很方便地把循环变成多线程运行**，从而利用 CPU 的多个核心（multi-core CPU）。

------

## OpenMP

-   让 `for` 循环并行执行（parallelize loops）
-   提高 CPU 密集型任务速度（speed up CPU-heavy work）
-   常用于：
    -   科学计算（scientific computing）
    -   图像处理（image processing）
    -   大数据计算（numerical computing）

------



## 经典例子：并行 for 循环

```cpp
#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    #pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        cout << "i=" << i << ", thread=" << omp_get_thread_num() << "\n";
    }
}
```

`#pragma omp parallel for` 的意思就是：

>   让这个 for 循环由多个线程同时跑。

------



## OpenMP 需要什么？

编译时要加一个选项：

GCC：

```bash
g++-15 main.cpp -fopenmp
```

Clang（macOS 默认）：

通常需要额外安装 OpenMP 库，配置更麻烦一点。

------



# OpenMP vs Thread Libraries（多线程工具）

**OpenMP = easy parallelism for loops (简单并行循环)**

**std::thread/pthread = low-level full-control threading (底层可控多线程)**

## 1) Programming Style（编程方式）

### **OpenMP**

**Directive-based**（指令式）
You add `#pragma` and the compiler handles threads automatically（编译器自动帮你开线程）.

Example:

```cpp
#pragma omp parallel for
for (...) { ... }
```

### **std::thread / pthread**

**Manual thread management**（手动管理线程）
You explicitly create/join threads（创建/回收线程） and split work yourself（自己分任务）.

Example:

```cpp
std::thread t(worker);
t.join();
```

------

## 2) Work Sharing（任务分配）

### **OpenMP**

Built-in work sharing（内置任务划分）
It can automatically split loops across threads（自动把循环分给多个线程）.

### **std::thread / pthread**

You must design the workload splitting（你要自己分配任务）
More flexible but more code（更灵活但更麻烦）.

------

## 3) Level of Control（控制粒度）

### **OpenMP**

Less control（控制少）
Good for data-parallel tasks（数据并行） like big loops.

### **std::thread / pthread**

Full control（完全控制）
Better for complex concurrent logic（复杂并发逻辑）.

------

## 4) Portability & Setup（可移植性/配置）

### **OpenMP**

Needs compiler support（依赖编译器支持）
Compile flag required（需要编译参数）:

-   GCC: `-fopenmp`
-   Clang(macOS): needs extra setup（需要额外安装库）

### **std::thread**

Standard C++ library（C++ 标准库）
Just compile normally:

```bash
g++ main.cpp
```

------

## 5) Typical Use Cases（典型场景）

### ✅ OpenMP is best for:

-   parallel loops（并行 for 循环）
-   matrix/vector operations（矩阵/向量计算）
-   scientific computing（科学计算）
-   quick speed-up（快速提速）

### ✅ std::thread / pthread is best for:

-   servers（服务器）
-   task pipelines（任务流水线）
-   producer-consumer（生产者消费者）
-   custom scheduling（自定义调度）



