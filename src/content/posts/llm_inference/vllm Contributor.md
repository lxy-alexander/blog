---
title: "vllm Contributor"
published: 2026-04-27
description: "vllm Contributor"
image: ""
tags: ["llm_inference","vllm Contributor"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-04-27T17:06:30.677.574426455Z"
---

# Contributing to vLLM 

## 1. 整体思路

-   **conda**（外层）：管 Python 版本、编译器（gcc/g++）、cmake/ninja、CUDA toolkit
-   **uv**（内层）：管所有 Python 包安装，包括 torch、vllm 本身的编译触发

------

## 2. 创建 conda 工具链环境

```bash
conda create -n vllm-toolchain -y \
    python=3.12 \
    cmake ninja \
    gcc_linux-64=12 gxx_linux-64=12 \
    -c conda-forge

conda activate vllm-toolchain
```

安装 CUDA toolkit（若集群没有对应 module）：

```bash
conda install -n vllm-toolchain -c nvidia cuda-toolkit=12.9 -y
```

>   ⚠️ 先用 `nvidia-smi` 确认驱动支持的最高 CUDA 版本 ≥ 12.9，否则会不兼容。

------

## 3. 关键坑：gcc/g++ 符号链接缺失

`gcc_linux-64` / `gxx_linux-64` 这两个 conda-forge 包只会生成**带平台前缀**的可执行文件：

```
x86_64-conda-linux-gnu-gcc
x86_64-conda-linux-gnu-g++
```

**不会**自动生成简化名 `gcc` / `g++`，导致 `which gcc` 会误命中系统里其它路径（比如集群自带的 `conda-compiler-shim`）。

**修复：手动建符号链接**

```bash
cd ~/miniconda3/envs/vllm-toolchain/bin/
ln -s x86_64-conda-linux-gnu-gcc gcc
ln -s x86_64-conda-linux-gnu-g++ g++
```

------

## 4. 让 activate 自动生效环境变量（不污染 .bashrc）

利用 conda 的环境激活钩子机制，只在 activate 这个环境时生效：

```bash
mkdir -p ~/miniconda3/envs/vllm-toolchain/etc/conda/activate.d
mkdir -p ~/miniconda3/envs/vllm-toolchain/etc/conda/deactivate.d

cat > ~/miniconda3/envs/vllm-toolchain/etc/conda/activate.d/env_vars.sh << 'EOF'
export CC=${CONDA_PREFIX}/bin/gcc
export CXX=${CONDA_PREFIX}/bin/g++
export PATH=${CONDA_PREFIX}/bin:$PATH
hash -r
EOF

cat > ~/miniconda3/envs/vllm-toolchain/etc/conda/deactivate.d/env_vars.sh << 'EOF'
unset CC
unset CXX
EOF
```

**验证：**

```bash
conda deactivate
conda activate vllm-toolchain
which gcc && which g++
gcc --version && g++ --version
```

>   `hash -r` 用于清除 bash 命令路径缓存，避免 PATH 顺序正确但仍命中旧缓存路径的问题。

------

## 5. Git 工作流（自己的 fork）

```bash
# 添加官方仓库为 upstream（只需一次）
git remote add upstream https://github.com/vllm-project/vllm.git

# 同步官方最新代码
git fetch upstream
git rebase upstream/main        # 无自己提交时，等价于 reset --hard upstream/main

# 推回自己的 fork（rebase 后需要强推）
git push origin main --force-with-lease
```

**merge vs rebase 选择：**

-   只有自己用这个 fork → 用 `rebase`，历史干净
-   有协作者/别人依赖你的分支 → 用 `merge`，避免强推覆盖他人历史
-   只是 `fetch` 没有自己的提交 → 两者结果一样，无影响

------

## 6. 编译 vLLM

```python
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT="$(
  curl -fsSL https://wheels.vllm.ai/nightly/cu130/vllm/metadata.json |
  jq -r '.[] |
    select(.platform_tag | endswith("x86_64")) |
    .path | split("/")[3]'
)" \
uv pip install -U -e . --torch-backend=auto
```

```bash
cd vllm

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

export CCACHE_DIR="${HOME}/.cache/ccache"
export CCACHE_NOHASHDIR=true
export MAX_JOBS=16
export TORCH_CUDA_ARCH_LIST="8.6;9.0;9.0a"

unset VLLM_USE_PRECOMPILED

uv pip install -e . \
  --torch-backend=auto \
  -v

uv pip install -r requirements/cuda.txt
```

------

## 7. 版本参考（截至 2026-08）

| 项目             | 版本              |
| ---------------- | ----------------- |
| vLLM 最新版      | v0.27.1           |
| 主线默认 CUDA    | 12.9（备选 13.0） |
| 对应 PyTorch     | ~2.9.0            |
| Python 推荐      | 3.10 – 3.12       |
| cmake 要求       | ≥ 3.26.1          |
| gcc/g++ 验证版本 | 12.x              |

>   ⚠️ P100（Compute Capability 6.0）较老，需确认目标 vLLM/CUDA 版本是否仍支持，必要时可能需要退回较早版本组合。

------

## 8. 常见排查命令速查

```bash
nvidia-smi              # 驱动支持的最高 CUDA 版本
nvcc --version           # 当前环境 CUDA toolkit 实际版本
gcc --version / g++ --version
cmake --version
which gcc / which g++ / which cmake
type -a gcc              # 排查是否有 alias/shim 干扰
echo $PATH | tr ':' '\n' | nl   # 查看 PATH 优先级顺序
```



```python
现在逻辑是动态的：
if [ -n "${CONDA_PREFIX-}" ] && [ -d "$CONDA_PREFIX/lib" ] ; then
    _OLD_VIRTUAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH
fi
我也验证了：激活后会把
/data/home/xli49/miniconda3/envs/triton-toolchain/lib
自动放到 LD_LIBRARY_PATH 最前面，并且：
hasattr(triton, "device_ops") == True
你以后只要在 (triton-toolchain) 里执行：
source .venv/bin/activate
```



# 提交

可以，不用 worktree，直接切分支即可。

## 1. 创建并切换分支

```
cd /data/home/xli49/lxy/vllm

git fetch upstream main

git switch -c fix/sparse-nccl-contiguous upstream/main
```

如果分支已经创建：

```
git switch fix/sparse-nccl-contiguous
```

确认：

```
git status
git branch --show-current
```

现有未跟踪文件会保留，因此后面不要使用 `git add .`。

## 2. 修改并测试

```
.venv/bin/python -m pytest \
  tests/distributed/test_weight_transfer.py \
  -k sparse_nccl \
  -v

pre-commit run --files \
  vllm/distributed/weight_transfer/sparse_nccl_engine.py \
  tests/distributed/test_weight_transfer.py
```

## 3. 提交

```
git diff

git add \
  vllm/distributed/weight_transfer/sparse_nccl_engine.py \
  tests/distributed/test_weight_transfer.py

git diff --cached

git commit -s \
  -m "[Bugfix][RL] Make sparse NCCL patch tensors contiguous"
```

## 4. 推送并创建 PR

```
git push -u origin fix/sparse-nccl-contiguous
```

然后创建 PR：

```
gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head lxy-alexander:fix/sparse-nccl-contiguous \
  --web
```

后续修改原 PR：

```
git add <修改文件>
git commit -s -m "[Bugfix][RL] Address review feedback"
git push
```

整个流程就是：

```
git switch -c fix/... upstream/main
→ 修改
→ 测试
→ git add 指定文件
→ git commit
→ git push
→ 创建 PR
```



核心规则：**PR 绑定的是分支，不是某个固定 commit。**
只要继续 push 到同一个个人分支，PR 会自动更新，不需要重新创建 PR。

先完成共同步骤：

```
git status
git diff

# 修改后运行相关测试
.venv/bin/python -m pytest <相关测试> -v

# 只添加相关文件
git add <file1> <file2>

git diff --cached --check
git diff --cached
```

## 情况一：已经 commit，但还没 push

如果修改属于刚才那个 commit，直接 amend 最干净：

```
git commit --amend --no-edit
```

如果需要修改 commit message：

```
git commit --amend
```

然后正常推送：

```
git push -u origin fix/sparse-nccl-contiguous
```

此时不需要 force，因为原 commit 从未推送。

如果修改是一个独立逻辑，也可以创建新 commit：

```
git commit -s -m "[Bugfix][RL] Add sparse NCCL regression coverage"
git push -u origin fix/sparse-nccl-contiguous
```

## 情况二：已经 push 到个人分支，但还没创建 PR

推荐直接增加新 commit：

```
git commit -s -m "[Bugfix][RL] Handle strided sparse patch tensors"
git push
```

个人 fork 上的分支会增加一个 commit。

如果特别希望保持单 commit，可以 amend 后安全强推：

```
git commit --amend --no-edit

git push --force-with-lease \
  origin fix/sparse-nccl-contiguous
```

使用 `--force-with-lease`，不要使用普通 `--force`。它会在远端分支被别人更新时拒绝覆盖。

## 情况三：PR 已经创建

最推荐增加一个新 commit：

```
git commit -s -m "[Bugfix][RL] Address sparse NCCL review feedback"
git push origin HEAD
```

结果是：

```
个人分支增加 commit
        ↓
现有 PR 自动出现新 commit
        ↓
GitHub 更新 Files changed
        ↓
CI 针对新 commit 重新运行
```

不需要：

-   关闭旧 PR
-   创建新 PR
-   修改 PR 的 base/head
-   手动把 commit 添加到 PR

如果是 reviewer 要求的修改，新 commit 通常更合适，因为 reviewer 可以清楚看到增量。

## PR 已创建，但想整理成一个 commit

如果只修改最后一个 commit：

```
git commit --amend --no-edit

git push --force-with-lease origin fix/sparse-nccl-contiguous
```

PR 会自动从旧 commit 切换到新 commit。

如果有多个本地 commit，需要整理：

```
git fetch upstream main
git rebase -i upstream/main
```

在编辑器里把后续 commit 改成 `fixup` 或 `squash`，完成后：

```
git push --force-with-lease \
  origin fix/sparse-nccl-contiguous
```

不过 vLLM 通常会在合并时 squash，因此 review 期间保留多个清晰的修复 commit 通常没问题，不必频繁重写历史。

## 修改 PR 描述

代码 push 后 PR 内容会自动更新，但 PR 描述不会自动变化。

如果测试命令、结果或设计发生变化，需要在网页中更新，或者：

```
gh pr edit <PR_NUMBER> \
  --repo vllm-project/vllm \
  --body-file pr-description.md
```

至少确保 PR 描述里的测试结果与最新 commit 一致。

## Review 修改的推荐流程

```
# 1. 根据 review 修改
git diff

# 2. 运行测试
.venv/bin/python -m pytest <相关测试> -v
pre-commit run --files <修改文件>

# 3. 精确暂存
git add <修改文件>
git diff --cached

# 4. 增加 review commit
git commit -s -m "[Bugfix][RL] Address review feedback"

# 5. 更新原 PR
git push origin HEAD
```

如果修改使用了 AI，新的 AI-assisted commit 也应保留相应 attribution trailer，例如：

```
Co-authored-by: OpenAI Codex <codex@openai.com>
```

## 特殊情况

-   PR 已关闭但未合并：push 仍会更新分支，但 PR 不会自动重新打开，需要在 GitHub 上 reopen。
-   PR 已经合并：不要继续复用原分支；从最新 `upstream/main` 创建新分支和新 PR。
-   Push 新 commit 后旧 CI 结果不再代表最新代码，需要看最新 head commit 的 checks。
-   不要用 `git add .`，避免把现有未跟踪文件混进 PR。


