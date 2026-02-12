---
title: "Git"
published: 2026-02-03
description: "Git"
image: ""
tags: ["tools","Git"]
category: tools
draft: false
lang: ""
---



# 删除工作区里 git 跟踪的文件

git restore .



# 删除工作区里 git 不跟踪的文件

| 命令                  | 危险程度 | 说明                         |
| --------------------- | -------- | ---------------------------- |
| `git clean -f`        | ⭐        | 只删少量文件                 |
| `git clean -fd`       | ⭐⭐       | 会删 build 目录              |
| **`git clean -fdx`**  | 🔥🔥🔥      | **几乎清空所有本地生成内容** |
| **`git clean -fdx`**n |          | `-n` = dry-run（只看不删）   |





# 更新一个 Fork 的仓库

## 方法 1：将上游仓库同步到你的 Fork（推荐）

### 1)：进入本地仓库

```bash
cd your-repo
```



### 2)：查看已有的远程仓库

```bash
git remote -v
```



### 3)：添加上游仓库（只需执行一次）

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/ORIGINAL_REPO.git
```



### 4)：从上游仓库获取最新更改

```bash
git fetch upstream
```



### 5) ：将上游更改合并到你的分支

#### 1）merge

如果上游分支是 `main`：

```bash
git checkout main
git merge upstream/main
```

如果上游分支是 `master`：

```bash
git checkout master
git merge upstream/master
```



#### 2) rebase

```
git checkout main
git fetch upstream
git rebase upstream/main
```



### 6)：将更新后的分支推送到你的 Fork

#### 1）merge

```bash
git push origin main
```

------

#### 2）rebase

```
git push origin main --force-with-lease
```



## 方法 2：直接在 GitHub 网页端更新

### 第 1 步：打开你的 Fork 仓库页面

进入你在 GitHub 上的 Fork 仓库。

### 第 2 步：点击同步按钮

依次点击：

```
Sync fork → Update branch
```

## 

# Git rebase 为什么更适合 fork 同步

在 **个人 fork 同步 upstream** 场景：**优先使用 rebase。**

## 常见场景

-   **upstream** 已经往前走了
-   你的 **fork 落后**
-   你本地还有自己的提交



<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206214648820" alt="image-20260206214648820" style="zoom: 50%;" /> 



## 用merge



-   会多一个 **Merge commit（M）**
-   历史变成“树状”，比较乱
-   PR 看起来不干净

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206215456002" alt="image-20260206215456002" style="zoom:50%;" /> 



## 用rebase

>   用了 `rebase` 之后：你的原来的 commit **会被丢弃（不再在当前分支历史里）**。Git 会创建：**内容一样，但 ID 全新的 commit**。==这也是为什么要 `--force-with-lease`，远程仓库还是D，本地变成D'==

提交历史干净

-   PR 清晰
-   是 **fork 同步 upstream 的标准做法**

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206215538616" alt="image-20260206215538616" style="zoom:50%;" /> 







# merge / rebase / Sync fork 冲突









