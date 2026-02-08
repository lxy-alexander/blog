---
title: "CUDA Learning Plan"
published: 2026-02-07
description: "CUDA Learning Plan"
image: ""
tags: ["cuda","CUDA Learning Plan"]
category: cuda
draft: false
lang: ""

---



# 写高性能 CUDA kernel

## ① CUDA Driver API（最底层）

### 本质

直接和 **GPU驱动** 交互的接口。

### 能做什么

-   创建 GPU 上下文
-   加载 PTX / cubin
-   启动 kernel
-   管理显存

### 特点

-   **最灵活**
-   **最复杂**
-   框架底层才用

### 谁在用

-   PyTorch / TensorFlow 底层
-   CUDA 编译器 / JIT
-   GPU 虚拟化系统

👉 **普通 CUDA 开发几乎不用。**

------

## ② CUDA Runtime API（最重要）

### 本质

对 Driver API 的 **高级封装**。

### 典型函数

```
cudaMalloc
cudaMemcpy
kernel<<<>>>
cudaStreamCreate
```

### 特点

-   90% CUDA 程序员只用这个
-   自动管理上下文
-   写 kernel 的标准方式

👉 **这是你必须重点学习的核心。**

------

## ③ CUDA Math API

### 本质

GPU 端的 **数学函数集合**。

比如：

-   sin / cos / exp
-   double / float 运算
-   向量数学

### 用途

在 **kernel 里面** 调用数学函数：

```
__global__ void f(float* x){
    x[0] = sinf(x[0]);
}
```

👉 **只是工具函数，不是性能核心。**

------

## ④ CUDA C++ Standard Library（libcu++）

### 本质

类似 C++ STL 的 **GPU 版标准库**。

提供：

-   并行算法
-   原子操作
-   线程同步
-   device 容器

有点像：

```
C++ STL + 并行 + GPU
```

### 使用场景

-   写现代 C++ CUDA 代码
-   高级模板库
-   Thrust / 并行算法

👉 **进阶才重要。**
