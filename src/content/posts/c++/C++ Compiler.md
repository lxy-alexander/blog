---
title: "C++ Compiler"
published: 2026-01-29
description: "C++ Compiler"
image: ""
tags: ["c++","C++ Compiler"]
category: c++
draft: false
lang: ""
---

# gcc / g++ / clang / clang++ / LLVM

-   **GCC** 是一套工具链，里面有 **gcc / g++**
-   **LLVM** 是另一套工具链，里面有 **clang / clang++**
-   macOS 的 **g++ 只是名字，实际是 Apple clang++**
-   `bits/stdc++.h` 主要是 **GCC 生态的非标准头文件**



# Compiler Relationship Diagram

```
                         Compiler Toolchains（编译器工具链）
                                   |
          ---------------------------------------------------------
          |                                                       |
      GCC Toolchain（GCC 工具链）                           LLVM Toolchain（LLVM 工具链）
          |                                                       |
   -------------------                                     -------------------
   |                 |                                     |                 |
 gcc (C compiler)    g++ (C++ compiler)                clang (C compiler)   clang++ (C++ compiler)
   |                 |                                     |                 |
   |                 |                                     |                 |
libstdc++ (C++ std library)                            libc++ (C++ std library)
(GCC C++ 标准库)                                        (LLVM C++ 标准库)
```

macOS Special Case

```
/usr/bin/g++   --->   Apple clang++ (不是 GCC)
```

bits/stdc++.h Location

```
bits/stdc++.h  -> usually in GCC environment (GCC 环境常见)
               -> usually NOT in clang/libc++ on macOS
```





## GCC 阵营（GNU）

-   项目名字：**GCC（GNU Compiler Collection）**
-   常见命令：
    -   `gcc`（编译 C）
    -   `g++`（编译 C++）

✅ 特点：

-   C++ 标准库常用：**libstdc++**
-   `bits/stdc++.h` 经常就在 GCC 的头文件体系里（非标准）

------



## LLVM 阵营（Clang）

-   项目名字：**LLVM**
-   C/C++ 前端编译器叫：**Clang**
-   常见命令：
    -   `clang`（编译 C）
    -   `clang++`（编译 C++）

✅ 特点：

-   macOS 默认就是这一套
-   C++ 标准库常用：**libc++**
-   通常没有 `bits/stdc++.h`

------





## clang 和 LLVM 是什么关系？

你可以理解成：

LLVM = 一整套编译器工具链（平台/基础设施）

里面包含很多东西，比如：

-   后端优化器（optimizer）
-   生成机器码（codegen）
-   linker（lld）
-   debugger（lldb）
-   运行时库等



Clang = LLVM 里面负责 “读 C/C++ 代码” 的编译器

也就是：

-   Clang 负责解析 `.c/.cpp`
-   然后把结果交给 LLVM 后端做优化 + 生成机器码

✅ 所以：
**Clang 是 LLVM 项目的一部分。**

------



## gcc 和 g++ 是什么关系？

它们是一套 GCC 工具链里的两个“入口命令”：

-   `gcc`：偏 C
-   `g++`：偏 C++

本质区别：`g++` 会自动按 C++ 方式处理并链接 C++ 标准库。

------



## macOS 里为啥 `g++` 显示 Apple clang？

你之前输出：

```
Apple clang version 17.0.0
```

说明 macOS 做了这件事：

✅ `/usr/bin/g++` 这个命令名
➡️ 实际指向 **Apple 的 clang++**

原因：

-   Apple 主推 LLVM/Clang
-   它为了兼容老习惯（很多人敲 `g++`）
-   所以把 `g++` 这个名字“借给 clang++ 用”

⚠️ 但它**不是 GCC**。



| 你看到的命令            | 它属于谁    | 用来编译 | 常用标准库 | `bits/stdc++.h` |
| ----------------------- | ----------- | -------- | ---------- | --------------- |
| `gcc`                   | GCC         | C        | (C库)      | ❌               |
| `g++`                   | GCC         | C++      | libstdc++  | ✅ 常见          |
| `clang`                 | LLVM/Clang  | C        | (C库)      | ❌               |
| `clang++`               | LLVM/Clang  | C++      | libc++     | ❌               |
| macOS 的 `/usr/bin/g++` | Apple Clang | C++      | libc++     | ❌               |



## macOS 安装并启用 `<bits/stdc++.h>`

1）安装 Homebrew GCC

```bash
brew install gcc
```

安装完成后，会在这里：

```bash
/usr/local/Cellar/gcc/15.2.0
```



2）确认 `g++-15` 是否存在

```bash
ls /usr/local/bin/g++-*
```

你应该看到类似：

```bash
/usr/local/bin/g++-15
```



3）测试 `<bits/stdc++.h>` 是否生效

新建文件 `test.cpp`：

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    cout << "bits/stdc++.h works!\n";
    return 0;
}
```

编译运行：

```bash
g++-15 test.cpp -std=c++17
./a.out
```

✅ 如果输出：

```
bits/stdc++.h works!
```

说明成功！

4）（可选）让 `g++` 默认指向 `g++-15`

这样以后不用每次写 `g++-15`。

```bash
echo 'alias g++="g++-15"' >> ~/.zshrc
source ~/.zshrc
```

验证：

```bash
g++ --version
```

✅ 看到类似 `Homebrew GCC 15.2.0` 就成功生效。





