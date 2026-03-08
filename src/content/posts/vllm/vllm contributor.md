---
title: "vllm contributor"
published: 2026-03-08
description: "vllm contributor"
image: ""
tags: ["vllm","vllm contributor"]
category: vllm
draft: false
lang: ""
---



# **I. Contributing to vLLM — Development Guide**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> This document covers the complete workflow for contributing to vLLM, including environment setup, two installation paths (Python-only vs. CUDA/C++ compilation), linting, documentation preview, test execution, and PR submission guidelines. Whether you are contributing for the first time or working on daily development, this guide serves as a handy reference.
</div>

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

## 2. Developing

### 1) Step 1: Clone the Repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### 2) Step 2: Create a Python Environment (Recommended: uv)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
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

#### Why Does vLLM Require `--no-build-isolation`?

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

