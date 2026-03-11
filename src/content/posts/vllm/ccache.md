---
title: "ccache"
published: 2026-02-11
description: "ccache"
image: ""
tags: ["vllm","ccache"]
category: vllm
draft: false
lang: ""
---

# **I. ccache — Compiler Cache (编译缓存) for C/C++/CUDA**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">ccache</span> is a compiler cache (编译缓存) that wraps your existing compiler (e.g., <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gcc</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">g++</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nvcc</code>) and stores the compilation output in a local cache. When the <strong>same source file + same compilation flags</strong> are seen again, ccache returns the cached object file instantly — skipping recompilation entirely. On large projects like vLLM or PyTorch, this can turn a 30-minute build into under 1 minute on the second run. </div>

------

## 1. How ccache Works (工作原理)

### 1) The Cache Key (缓存键)

**Interviewer:** *"How does ccache know when it can reuse a cached result?"*

>   ccache hashes a combination of inputs to produce a unique cache key (缓存键). If the key matches an existing entry, it returns the cached object file directly — no compiler invoked.

```
Cache Key (缓存键) = Hash of:
  ├── Source file content (源文件内容)
  ├── Compilation flags (编译参数)  e.g. -O2 -std=c++17 -DNDEBUG
  ├── Compiler version (编译器版本) e.g. gcc 11.3.0
  ├── Include file contents (头文件内容)
  └── Environment variables (环境变量) that affect compilation
```

### 2) Two Lookup Modes (两种查找模式)

| Mode                                                         | Description                                     | Speed                 |
| ------------------------------------------------------------ | ----------------------------------------------- | --------------------- |
| <span style="color:#E8600A;font-weight:700">Direct mode (直接模式)</span> | Hash source + flags, skip preprocessor entirely | Fastest ⚡             |
| <span style="color:#E8600A;font-weight:700">Preprocessed mode (预处理模式)</span> | Run preprocessor first, then hash result        | Slower, more accurate |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Direct mode is the default and covers most use cases. If you see unexpected cache misses (缓存未命中), it may fall back to preprocessed mode automatically.</div>

------

## 2. Manual Installation on HPC (在HPC上手动安装)

>   No root access needed — install entirely under <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">$HOME</code>.

### 1) Download & Extract (下载并解压)

```bash
mkdir -p $HOME/local
cd $HOME/local

wget https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz
tar xf ccache-4.10.2-linux-x86_64.tar.xz
```

### 2) Symlink to PATH (创建符号链接)

```bash
ln -s $HOME/local/ccache-4.10.2-linux-x86_64/ccache $HOME/.local/bin/ccache
```

### 3) Add to PATH permanently (写入 PATH)

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 4) Verify (验证安装)

```bash
ccache --version
# ccache version 4.10.2
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> On HPC clusters (高性能计算集群), <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">$HOME/.local/bin</code> may already be on your PATH. Run <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">echo $PATH</code> to check before editing <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/.bashrc</code>.</div>

------

## 3. Reading the Stats Output (读懂统计输出)

Run <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ccache -s</code> to see a full status report. Here's how to interpret each field:

```bash
ccache -s
Cacheable calls:    772 / 772 (100.0%)
  Hits:               0 / 772 ( 0.00%)
    Direct:           0
    Preprocessed:     0
  Misses:           772 / 772 (100.0%)
Local storage:
  Cache size (GiB): 1.0 / 5.0 (19.48%)
  Cleanups:         240
  Hits:               0 / 772 ( 0.00%)
  Misses:           772 / 772 (100.0%)
```

### 1) Hit Rate = 0% (命中率为零)

<span style="color:#C0392B;font-weight:600">This is the most important signal.</span>

| Field                                                        | Value     | Meaning                                              |
| ------------------------------------------------------------ | --------- | ---------------------------------------------------- |
| <span style="color:#E8600A;font-weight:700">Cacheable calls (可缓存调用)</span> | 772 / 772 | Every compilation was eligible for caching           |
| <span style="color:#E8600A;font-weight:700">Hits (命中)</span> | 0 / 772   | Zero reuse — compiler ran from scratch every time    |
| <span style="color:#E8600A;font-weight:700">Misses (未命中)</span> | 772 / 772 | All 772 results were written to cache as new entries |

**Three reasons for 0% hit rate:**

<span style="color:#2980B9">① First-time build (首次编译)</span> — Cache is cold (冷缓存). This is normal. The next build will hit close to 100%.

<span style="color:#2980B9">② Flags or paths changed (参数或路径变化)</span> — Even one flag difference (e.g., a different <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">-I/path</code>) invalidates the cache key. Absolute paths in flags are a common culprit (常见原因) on HPC where job nodes differ.

<span style="color:#2980B9">③ ccache not actually intercepting (ccache未真正生效)</span> — The build system may be calling the compiler directly, bypassing ccache entirely.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> After a first full build with 0% hits, run <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ccache -z</code> (reset stats) and rebuild. If you still see 0% hits on the second run, ccache is <span style="color:#C0392B;font-weight:600">not being invoked by your build system</span>.</div>

### 2) Cache Size Is Small (缓存体积较小)

```
Cache size (GiB): 1.0 / 5.0 (19.48%)
```

>   Only 1 GiB used out of 5 GiB limit — the first build just populated the cache. This is expected after one cold build. <span style="color:#2980B9">Do not delete the cache now</span>; it is exactly what will accelerate the next build.

### 3) High Cleanup Count (清理次数多)

```
Cleanups: 240
```

>   ccache auto-cleans when the cache exceeds its size limit. 240 cleanups suggests the limit was hit repeatedly — likely because the default 5 GiB cap is too small for a large project like vLLM (which can generate 10–20 GiB of object files).

**Fix — increase the cache size limit (扩大缓存上限):**

```bash
ccache --max-size=20G    # set to 20 GiB
ccache -s                # confirm new limit
```

------

## 4. Verifying ccache Is Actually Intercepting (验证ccache真正生效)

### 1) Check which compiler is being called (检查调用的编译器)

```bash
which gcc        # should show $HOME/.local/bin/gcc  ← ccache wrapper
which ccache     # should show $HOME/.local/bin/ccache
```

### 2) Compiler symlink method (编译器符号链接方法)

Create wrappers so the build system transparently calls ccache:

```bash
ln -s $(which ccache) $HOME/.local/bin/gcc
ln -s $(which ccache) $HOME/.local/bin/g++
ln -s $(which ccache) $HOME/.local/bin/nvcc   # for CUDA (CUDA编译)
```

>   When called as <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gcc</code>, ccache detects its own name via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">argv[0]</code> and automatically uses the real <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gcc</code> as the backend.

### 3) CMake / pip build integration (CMake集成)

```bash
# For CMake projects
cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      ..

# For pip installs (e.g., vLLM)
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
pip install -e .
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">CMAKE_*_COMPILER_LAUNCHER</code> approach is the <strong>cleanest and most reliable</strong> method for modern projects — it tells CMake to prefix every compiler call with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ccache</code> without modifying the build system files.</div>

------

## 5. Useful Commands Cheatsheet (常用命令速查)

```bash
ccache -s                  # show stats (查看统计)
ccache -z                  # reset stats to zero (重置统计)
ccache -C                  # clear entire cache (清空缓存)
ccache --max-size=20G      # set cache size limit (设置缓存上限)
ccache --show-config       # show all config values (查看配置)
CCACHE_DISABLE=1 make      # temporarily disable ccache (临时禁用)
CCACHE_DEBUG=1 make        # verbose debug output (调试输出)
```

------

## 6. Diagnosing Common Issues on HPC (HPC常见问题诊断)

| Symptom (现象)                 | Likely Cause (可能原因)                                      | Fix (解决方法)                                               |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Hit rate = 0% after 2nd build  | ccache not intercepting                                      | Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CMAKE_*_LAUNCHER</code> env vars |
| Many Cleanups                  | Cache size too small                                         | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ccache --max-size=20G</code> |
| Cache misses after node change | Absolute paths in flags differ per node                      | Set <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CCACHE_BASEDIR=$HOME</code> to normalize paths |
| nvcc not cached                | CUDA compiler not wrapped                                    | Add <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CMAKE_CUDA_COMPILER_LAUNCHER=ccache</code> |
| Quota exceeded on HPC          | Cache in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">$HOME</code> eating disk | Move cache: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">export CCACHE_DIR=/scratch/$USER/.ccache</code> |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> On HPC systems, <code style="background:#HOME</code> quotas are often tight (e.g., 50 GiB). Store the ccache directory on a <strong>scratch filesystem (临时文件系统)</strong> like <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">/scratch</code> which typically has no quota. Set <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">export CCACHE_DIR=/scratch/$USER/.ccache</code> in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/.bashrc</code>.</div>

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>A 0% hit rate after the first build is <em>completely normal</em> — the cache is now warm; the next build will be dramatically faster. The real action item is to <strong>verify ccache is intercepting</strong> via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CMAKE_*_COMPILER_LAUNCHER</code>, <strong>increase the size limit</strong> with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--max-size=20G</code>, and <strong>move the cache to scratch</strong> (临时文件系统) if HPC disk quota is tight.</div>
