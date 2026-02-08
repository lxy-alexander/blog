---
title: "Remote VSCode For CUDA"
published: 2026-02-07
description: "Remote VSCode For CUDA"
image: ""
tags: ["tools","Remote VSCode For CUDA"]
category: tools
draft: false
lang: ""
---





# **HPC GPU 集群** 

```
本地 VSCode
    ↓ Remote-SSH
远程 CUDA 工程
    ↓ CMake 构建
    ↓ cuda-gdb 调试
    ↓ GPU kernel 断点
```

具备：

-   ✔ C/C++ 代码提示（IntelliSense / clangd）
-   ✔ CUDA `.cu` 语法解析
-   ✔ 多文件工程 CMake 构建
-   ✔ VSCode F5 直接断进 GPU kernel
-   ✔ 适配 HPC module / 计算节点
-   ✔ 长期稳定，不依赖个人 hack

