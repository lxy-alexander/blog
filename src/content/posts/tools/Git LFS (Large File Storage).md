---
title: "Git LFS (Large File Storage)"
published: 2026-03-14
description: "Git LFS (Large File Storage)"
image: ""
tags: ["tools","Git LFS (Large File Storage)"]
category: tools
draft: false
lang: ""
---

# **I. `git lfs install` (Git LFS 初始化命令)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Git LFS (Large File Storage，大文件存储)</span> is an extension of <span style="color:#E8600A;font-weight:700">Git (版本控制系统)</span> used to manage large files such as datasets, models, or binaries.   The command <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git lfs install</code> initializes <span style="color:#E8600A;font-weight:700">Git LFS</span> on your machine by configuring Git hooks and global settings.   <span style="color:#2980B9">In simple words</span>, it prepares Git so that large files will automatically be handled by the LFS system instead of normal Git storage. </div>

## <span style="color:#E8600A">1.</span> What `git lfs install` Does (命令作用)

Running

```
git lfs install
```

performs several setup steps:

| Step          | Explanation                                                  |
| ------------- | ------------------------------------------------------------ |
| Install hooks | Adds <span style="color:#E8600A;font-weight:700">Git Hooks (Git钩子)</span> such as `pre-push` |
| Configure Git | Enables <span style="color:#E8600A;font-weight:700">LFS Filters (LFS过滤器)</span> |
| Activate LFS  | Allows Git to replace large files with <span style="color:#E8600A;font-weight:700">pointer files (指针文件)</span> |

After installation, Git will automatically:

1.  detect large files
2.  store them in <span style="color:#E8600A;font-weight:700">LFS storage (LFS存储)</span>
3.  keep only small pointer references in the repository

## <span style="color:#E8600A">2.</span> How Git LFS Works (工作原理)

Normal Git workflow:

```
file → git repository
```

Git LFS workflow:

```
large file → LFS server
pointer file → git repository
```

Example pointer file:

```
version https://git-lfs.github.com/spec/v1
oid sha256:xxxx
size 104857600
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Note: </span> A <span style="color:#E8600A;font-weight:700">pointer file (指针文件)</span> is a small text file that references the real large file stored in the LFS server. </div>

## <span style="color:#E8600A">3.</span> Typical Usage Workflow (常见使用流程)

### 1) Install Git LFS

```
git lfs install
```

### 2) Track large files

Example:

```
git lfs track "*.pt"
```

This tells Git to manage `.pt` files using LFS.

### 3) Commit tracking rules

```
git add .gitattributes
git commit -m "track model files with LFS"
```

### 4) Add large file

```
git add model.pt
git commit -m "add model"
git push
```

The file will be stored in <span style="color:#E8600A;font-weight:700">LFS storage (LFS服务器)</span>.

## <span style="color:#E8600A">4.</span> When You Need `git lfs` (什么时候需要)

You should use Git LFS when managing:

| File Type               | Example            |
| ----------------------- | ------------------ |
| Machine learning models | `.pt`, `.pth`      |
| Datasets                | `.csv`, `.parquet` |
| Game assets             | textures, audio    |
| Large binaries          | compiled files     |

<span style="color:#C0392B;font-weight:600">Warning:</span>
 Normal Git performs poorly with very large files because the entire file history is stored inside the repository.

## <span style="color:#E8600A">5.</span> Verify Installation (验证安装)

Check whether Git LFS is installed:

```
git lfs version
```

Example output:

```
git-lfs/3.4.0
```

Check tracked files:

```
git lfs ls-files
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">git lfs install</code> initializes <span style="color:#E8600A;font-weight:700">Git LFS (Git大文件存储)</span> by configuring Git hooks and filters so that large files are stored in <span style="color:#E8600A;font-weight:700">LFS storage</span> instead of the Git repository. </div>
