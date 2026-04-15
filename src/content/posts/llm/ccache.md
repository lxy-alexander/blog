---
title: "ccache"
published: 2026-02-11
description: "ccache"
image: ""
tags: ["llm","ccache"]
category: llm
draft: false
lang: ""
---

# I. ccache — Compiler Cache (编译缓存)

>   ccache wraps your compiler (`gcc`, `g++`, `nvcc`) and caches object files. Same source + same flags = instant replay, no recompilation.

------

## 1. How It Works (工作原理)

$$ \text{Cache Key (缓存键)} = \text{Hash}(\text{source content} + \text{flags} + \text{compiler version} + \text{headers}) $$

On a **hit (命中)**: return cached `.o` file immediately. On a **miss (未命中)**: compile normally, store result.

Two modes: **Direct mode (直接模式)** — fastest, hashes source directly. **Preprocessed mode (预处理模式)** — slower, more accurate, used as fallback.

------

## 2. Installation (无root安装)

```bash
# Download & link
wget https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz
tar xf ccache-4.10.2-linux-x86_64.tar.xz -C $HOME/local
ln -s $HOME/local/ccache-4.10.2-linux-x86_64/ccache $HOME/.local/bin/ccache
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

------

## 3. Integration (接入构建系统)

```bash
# CMake projects
cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache ..

# pip installs (e.g. vLLM)
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
pip install -e .
```

------

## 4. Reading Stats (读懂统计)

```bash
ccache -s
```

| Field                  | Meaning                                                |
| ---------------------- | ------------------------------------------------------ |
| Hits = 0% on 1st build | Normal — cache is cold (冷缓存), next build hits ~100% |
| Hits = 0% on 2nd build | ccache **not intercepting** — check `CMAKE_*_LAUNCHER` |
| Cleanups: 240          | Cache limit too small → run `ccache --max-size=20G`    |

------

## 5. HPC-Specific Tips (HPC注意事项)

```bash
# Avoid $HOME quota — move cache to scratch (临时文件系统)
export CCACHE_DIR=/scratch/$USER/.ccache

# Normalize absolute paths across nodes (跨节点路径归一化)
export CCACHE_BASEDIR=$HOME

# Increase size limit (扩大缓存上限)
ccache --max-size=20G
```

------

## 6. Quick Reference (命令速查)

```bash
ccache -s              # stats (统计)
ccache -z              # reset stats (重置)
ccache -C              # clear cache (清空)
CCACHE_DISABLE=1 make  # disable temporarily (临时禁用)
```
