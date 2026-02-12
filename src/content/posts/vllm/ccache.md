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

>ccache is a compiler cache that accelerates C/C++/CUDA build by reusing previously compiled  file when the same source code and compilation parameters are encountered again.



## Install  manually

```shell
mkdir -p $HOME/local
cd $HOME/local

wget https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz
tar xf ccache-4.10.2-linux-x86_64.tar.xz
```

```shell
ln -s $HOME/local/ccache-4.10.2-linux-x86_64/ccache $HOME/.local/bin/ccache
```

```shell
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

```shell
ccache --version
```



## ccache -s



```
(vllm) [xli49@ghpc008 ~]$ ccache -s
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

### 1️⃣ 命中率 = 0%

```
Hits: 0 / 772 (0%)
Misses: 772 / 772 (100%)
```

说明：

-   **完全没有复用到任何编译缓存**
-   每次都是重新编译

👉 这通常意味着：

-   你只是**第一次编译**
-   或者 **编译参数/路径变化导致缓存失效**
-   或者 **ccache 没真正生效**

------

### 2️⃣ 缓存大小很小

```
Cache size: 1.0 GiB / 5.0 GiB
```

说明：

-   缓存**还没怎么用**
-   现在删除意义不大（只占 1GB）

------

### 3️⃣ Cleanups 很多

```
Cleanups: 240
```

一般代表：

-   之前频繁触发清理
-   可能磁盘配额比较紧（HPC 常见）



