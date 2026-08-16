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

```bash
cd vllm

uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
export CCACHE_DIR="${HOME}/.cache/ccache"
export CCACHE_NOHASHDIR=true
export MAX_JOBS=16
uv pip install -e . --torch-backend=auto -v

# 指定多架构，兼容集群内不同 GPU 型号（按需调整）
export TORCH_CUDA_ARCH_LIST="6.0;8.6;9.0"   # P100=6.0, A40=8.6, H100=9.0

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






---

## 1. Contributing to vLLM

Ways to contribute include:

- Reporting bugs / opening issues
- Adding support for new models
- Implementing new features
- Improving documentation
- Helping others, reviewing PRs
- Starring the repo, writing articles — these count too

---

## 2. Docker/Container

### 1) Step 1: Clone the Repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### 2) Use Container

```bash
cd container
apptainer pull vllm-openai.sif docker://vllm/vllm-openai:latest

export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
VLLM_PROJECT_PATH="/data/home/xli49/vllm"
cd "$VLLM_PROJECT_PATH"


apptainer shell --nv \
  --bind "${VLLM_PROJECT_PATH}:${VLLM_PROJECT_PATH}" \
  --bind "${HOME}/.cache/huggingface:${HOME}/.cache/huggingface" \
  --pwd "${VLLM_PROJECT_PATH}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  ../container/vllm-openai.sif
```



## 3.UV

### 1) Step 1: Clone the Repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

srun -p highmem32   --cpus-per-task=8   --mem=64G   --time=12:00:00   --pty /bin/bash
```

### 2) Step 2: Create a Python Environment (Recommended: uv)

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
export CCACHE_DIR="${HOME}/.cache/ccache"
export CCACHE_NOHASHDIR=true
export MAX_JOBS=16
uv pip install -e . --torch-backend=auto -v
```

If you don't have uv, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Why Python 3.12? Because vLLM's CI (official automated tests) primarily uses 3.12. Using the same version prevents situations where tests pass locally but fail in CI.</div>

To delete the virtual environment:

```bash
rm -rf .venv
uv cache clean


```

---

## 3. Installing vLLM (Two Paths)

### 1) Path A: Python-only Changes (Fastest, Recommended)

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

What this means:

- Installs in <span style="color:#E8600A;font-weight:700">Editable Mode</span> (`-e`) — changes to source files take effect immediately
- Does **not** compile C++/CUDA locally
- Downloads pre-compiled binaries from the corresponding pre-built wheel

👉 Advantage: Very fast, suitable for the majority of PRs.

---

### 2) Path B: CUDA/C++ Changes (Requires Local Compilation)

If you previously ran Path A, first **force-remove** the installed `vllm` Python package:

```bash
uv pip uninstall vllm
```

Install PyTorch (cu129):

```bash
uv pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

Install the current project in Editable Mode:

```bash
CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e . -v
CCACHE_NOHASHDIR="true" uv pip install -e . -v
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">uv pip install -e .</code> installs the project in the current directory in editable mode. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.</code> refers to the current directory (i.e., the vllm repo root). It reads <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pyproject.toml</code> (primary) or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">setup.py</code> (legacy), then installs the project into your virtual environment.</div>

#### Common Error: `ImportError: undefined symbol`

<span style="color:#C0392B;font-weight:600">If you encounter the following error:</span>

```
(vllm) [xli49@ghpc008 vllm]$ python examples/offline_inference/basic/basic.py
Traceback (most recent call last):
  ...
  File "/data/home/xli49/vllm/vllm/platforms/cuda.py", line 16, in <module>
    import vllm._C  # noqa
    ^^^^^^^^^^^^^^
ImportError: /data/home/xli49/vllm/vllm/_C.abi3.so: undefined symbol: _ZN3c104cuda9SetDeviceEa
```

The cause is a mismatch between the torch ABI used at compile time and the torch version at runtime. Ensure you use `--no-build-isolation` and recompile with the correct CUDA version:

```bash
uv pip install -e . --no-build-isolation
```

#### Why Does vLLM Require `--no-b`

#### `uild-isolation`?

Because compiling vLLM's C++/CUDA extensions depends heavily on:

- The `torch` installed in your current environment
- The matching CUDA version (cu129/cu128, etc.)
- Other compilation-related packages

Without this flag, the build system uses an isolated temporary environment, which may result in:

- A mismatched `torch` being installed in the temporary environment
- The current torch's CUDA configuration not being found
- Compilation failures or incompatible binaries being generated

---

## 4. Linting (Code Style & Formatting)

vLLM uses <span style="color:#E8600A;font-weight:700">pre-commit</span> to enforce a unified code style.

- <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">uv pip install pre-commit</code>: installs the pre-commit tool
- <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pre-commit install</code>: installs hooks into `.git/hooks/` so that checks run automatically on every `git commit`

### 1) Install and Enable

```bash
uv pip install pre-commit
pre-commit install
```

From now on, every `git commit` will automatically run the checks ✅

### 2) Run Manually

```bash
pre-commit run      # Check only staged files
pre-commit run -a   # Check all files (= --all-files)
```

### 3) CI-only Hooks (Trigger Locally on Demand)

```bash
pre-commit run --hook-stage manual markdownlint
pre-commit run --hook-stage manual mypy-3.10
```

---

## 5. Documentation

vLLM's docs are built with <span style="color:#E8600A;font-weight:700">MkDocs</span>.

### 1) Install Documentation Dependencies

```bash
uv pip install -r requirements/docs.txt
```

### 2) Preview the Docs Site Locally

```bash
mkdocs serve
```

### 3) Faster Preview (Skip API Reference Generation)

Controls whether the API Reference is generated.

```bash
API_AUTONAV_EXCLUDE=vllm mkdocs serve
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Ensure your Python version is compatible with the plugins. For example, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">mkdocs-awesome-nav</code> requires Python 3.10+.</div>

### 4) Forward the Port from a Remote Server

<code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-L</code> = Local port forwarding: **maps a port on the remote machine to a port on your local machine**.

```bash
ssh -L 8000:127.0.0.1:8000 xli49@spiedie.binghamton.edu
```

### 5) Connect to a Remote GPU Node via Jump Host

<code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-J</code> = Jump host: **connect to a target machine by hopping through an intermediate host first**.

```bash
ssh -J xli49@spiedie.binghamton.edu -L 8000:127.0.0.1:8000 xli49@ghpc005
```

---

## 6. Testing

vLLM uses <span style="color:#E8600A;font-weight:700">pytest</span>.

### 1) Path A: Full CI-equivalent Setup (CUDA)

```bash
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto
pytest tests/
```

### 2) Path B: Minimal Test Tooling Only

```bash
uv pip install pytest pytest-asyncio
pytest tests/
```

### 3) Run a Single Test File (Useful for Debugging)

```bash
pytest -s -v tests/test_logger.py
```

---

## 7. Common Errors

### 1) Missing `Python.h`

If you encounter the following error during compilation or dependency installation:

```
Python.h: No such file or directory
```

Fix on Ubuntu:

```bash
sudo apt install python3-dev
```

---

## 8. Important Warnings

<span style="color:#C0392B;font-weight:600">✅ The repository is not yet fully covered by mypy</span> — do not rely on mypy being fully green.

<span style="color:#C0392B;font-weight:600">⚠️ Not all tests pass on CPU</span> — without a GPU, many tests will fail locally. The official stance is: rely on CI for those tests.

---

## 9. PR Submission Guidelines

### 1) DCO Sign-off

Every commit must include a `Signed-off-by` line:

```bash
git commit -s -m "xxx"
```

### 2) PR Title Must Include a Category Prefix

Examples:

- `[Bugfix] ...`
- `[Kernel] ...`
- `[Core] ...`
- `[Doc] ...`
- `[CI/Build] ...`

<span style="color:#C0392B;font-weight:600">PRs without a valid prefix may not be reviewed.</span>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> For Python-only changes, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">VLLM_USE_PRECOMPILED=1 uv pip install -e .</code> to get started in seconds; for CUDA/C++ changes, always compile with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--no-build-isolation</code> and match your torch CUDA version to avoid ABI symbol errors.</div>
