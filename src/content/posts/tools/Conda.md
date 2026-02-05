---
title: "Conda"
published: 2026-02-03
description: "Conda"
image: ""
tags: ["tools","Conda"]
category: tools
draft: false
lang: ""
---

可以删除 conda 缓存（释放磁盘、减少环境残留）

------

## 1）先看 conda 缓存占用（可选）

```bash
conda info
conda config --show pkgs_dirs
du -sh ~/miniconda3/pkgs 2>/dev/null
```

------



## 2）一键清理 conda 缓存（推荐）

```bash
conda clean -a -y
```

`-a` = **all（清理所有类型的缓存/垃圾）**

`-y` = **yes（自动确认）**

它会清掉：

-   下载过的 tar.bz2 / .conda 包
-   解压缓存
-   index 缓存
-   未使用的包缓存

------



## 3）如果你还用过 pip（在 conda env 里）

pip 也会有缓存，可以清：

```bash
pip cache purge
```

------



## 4）验证释放空间（可选）

```bash
du -sh ~/miniconda3/pkgs 2>/dev/null
```

------

⚠️ 注意：`conda clean -a` 不会删你的环境，只是下次装包可能需要重新下载。
