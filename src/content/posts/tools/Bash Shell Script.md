---
title: "Bash Shell Script"
published: 2026-02-07
description: "Bash Shell Script"
image: ""
tags: ["tools","Bash Shell Script"]
category: tools
draft: false
lang: ""
---



# 整体概念

## 1）什么是 Bash 脚本

Bash Shell Script 是基于 **Unix Shell（Bash）解释执行** 的脚本语言，用来：

-   自动化命令执行
-   管理环境变量
-   控制程序流程（判断、循环等）
-   作为 Linux / HPC 系统初始化与运维的核心工具

它**不是编译型语言**，而是：

>   **逐行解释执行的命令脚本语言**

------

## 2）Bash 脚本的执行方式

常见三种：

-   **直接执行**

    ```bash
    bash script.sh
    ```

-   **赋予可执行权限后运行**

    ```bash
    chmod +x script.sh
    ./script.sh
    ```

-   **在当前 shell 中执行（source）**

    ```bash
    source script.sh
    # 或
    . script.sh
    ```

区别关键：

| 方式               | 是否新开子 shell | 变量是否保留 |
| ------------------ | ---------------- | ------------ |
| `bash script.sh`   | 是               | ❌ 不保留     |
| `source script.sh` | 否               | ✔ 保留       |

------



# 基础语法结构

## 1）变量与环境变量

### （1）普通变量

```bash
name="Alice"
echo $name
```

### （2）环境变量

```bash
export PATH=/usr/bin:$PATH
```

特点：

-   子进程可见
-   常用于 **PATH / CUDA / Conda / MPI**

------





## 2）条件判断（if）

基本结构：

```bash
if 条件; then
    语句
fi
```

示例：

```bash
if [ -f file.txt ]; then
    echo "文件存在"
fi
```

常见测试符：

| 符号 | 含义     |
| ---- | -------- |
| `-f` | 普通文件 |
| `-d` | 目录     |
| `-e` | 存在     |
| `-x` | 可执行   |

------



## 3）循环结构

### （1）for 循环（最常见）

```bash
for f in *.txt; do
    echo $f
done
```

作用：

>   遍历文件 / 列表 / 命令结果

------

### （2）while 循环

```bash
while read line; do
    echo $line
done < file.txt
```

------



# 命令执行机制

## 1）子 shell 执行

```bash
bash script.sh
```

特点：

-   新进程
-   变量不影响当前终端

------



## 2）source（`.`）执行 ⭐

```bash
source script.sh
# 或
. script.sh
```

作用：

>   **在当前 shell 中执行脚本内容**

常用于：

-   `.bashrc`
-   环境变量初始化
-   HPC module / CUDA 加载

------





# 文件与路径操作

## 1）文件测试

```bash
[ -d dir ]
[ -f file ]
```

属于：

>   **POSIX test 语法**

------



## 2）路径与通配符

```bash
~/.bashrc.d/*
```

含义：

-   `~` → 用户家目录
-   `*` → 匹配所有文件

------





# 控制流组合示例（典型 `.bashrc` 结构）

`. "$rc"`在“当前 shell”中执行文件 `$rc` 的内容。等价于：`source "$rc"`

```bash
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        if [ -f "$rc" ]; then
            . "$rc"
        fi
    done
fi
```

功能：

>   **遍历目录并 source 所有脚本，实现模块化配置加载**

这是：

-   Linux 默认设计
-   HPC 常见初始化方式

------







