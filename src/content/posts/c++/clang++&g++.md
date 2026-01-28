---
title: "clang++&g++"
published: 2026-01-27
description: "clang++&g++"
image: ""
tags: ["c++","clang++&g++"]
category: c++
draft: false
lang: ""
---

## 看看g++和clang+是否支持-std=c++17

```shell
g++ -std=c++17 -x c++ -dM -E - < /dev/null | grep __cplusplus

clang++ -std=c++17 -x c++ -dM -E - < /dev/null | grep __cplusplus
```

-   clang++：使用 Clang 的 C++ 编译器前端
-   -std=c++17：指定使用 C++17 语言标准

-   -x c++：强制把输入当作 C++ 源码
-   -dM：输出 所有预定义的宏

-   -E：只执行 预处理阶段
-   -：从 标准输入（stdin） 读取源码

-   < /dev/null：给 stdin 一个 空输入
-   |：把前一个命令的输出传给下一个命令

-   grep __cplusplus：只显示 __cplusplus 宏

-   合起来的意思：用 C++17 模式检查编译器当前使用的 C++ 标准。



