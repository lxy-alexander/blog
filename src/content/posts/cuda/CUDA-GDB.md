---
title: "CUDA-GDB"
published: 2026-02-08
description: "CUDA-GDB"
image: ""
tags: ["cuda","CUDA-GDB"]
category: cuda
draft: false
lang: ""
---



# 编译应用程序

## 1）调试编译

`-g`、`-G`、`-O0` 在 NVCC 中的含义

`-g` 管 CPU，`-G` 管 GPU，`-O0` 保证断点真实。

在使用 **cuda-gdb** 调试 CUDA 程序时，通常需要同时启用三个编译选项：

-   `-g`：生成 **CPU 端调试符号**，用于查看源码行号、变量名和调用栈。
-   `-G`：生成 **GPU kernel 调试信息**，使断点能够命中 kernel、支持单步执行线程并查看设备端变量；这是 CUDA 调试最关键的选项。官方语义可理解为 “`-G` 隐含 `-O0`”，但在工程实践中仍建议显式写出 `-O0`，以避免工具链差异带来的不确定行为。
-   `-O0`：关闭优化，防止变量被消除、指令重排或控制流改变，从而避免断点漂移或变量显示异常。
-    `-lineinfo` 提供更准确的源码行号映射，在性能分析工具中尤为有用；但**真正的设备端单步调试仍依赖 `-G`**。

```
nvcc -g -G -O0 -lineinfo -arch=sm_86 foo.cu -o foo
```





## 2）显式写 `-gencode`

兼容（compatible）其他GPU

例如：

```
既支持 A40 (sm_86)
又支持 H100 (sm_90)
```

才需要：

```
多个 -gencode
```

