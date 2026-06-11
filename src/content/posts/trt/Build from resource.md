---
title: "Build from resource"
published: 2026-05-23
description: "Build from resource"
image: ""
tags: ["trt","Build from resource"]
category: trt
draft: false
lang: ""
createdAt: "2026-05-23T14:29:42.639.760153809Z"
---



# TensorRT-LLM Development Workflow Notes

Use the official TensorRT-LLM version that matches the container version. 1. Install Git LFS in Conda Base

Git LFS (Large File Storage，大文件存储) is required for downloading model and dependency files.

```bash
conda install -c conda-forge git git-lfs -y

git lfs install
```

Example:

```bash
$ git lfs install
Git LFS initialized.
```

## 2. Clone TensorRT-LLM Repository

Always pull submodules (子模块) and LFS files after cloning.

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git

cd TensorRT-LLM

git submodule update --init --recursive

git lfs pull
```

Example:

```bash
$ git lfs pull
Downloading LFS objects: 100% (xxx/xxx), done.
```

## 3. Pull TensorRT-LLM Development Container

Use the official development container (开发容器) to avoid dependency issues.

```bash
apptainer pull tensorrt_llm_devel_1.3.0rc15.sif \
docker://nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc15
```

Example:

```bash
$ apptainer pull tensorrt_llm_devel_1.3.0rc15.sif
INFO: Download complete
```

## 4. Enter the Container

Bind (挂载) the TensorRT-LLM source directory into the container.

```bash
PROJECT_PATH="/data/home/xli49/trt/TensorRT-LLM"

apptainer shell --nv \
--bind "${PROJECT_PATH}:${PROJECT_PATH}" \
--pwd "${PROJECT_PATH}" \
../container/tensorrt_llm_devel_1.3.0rc15.sif
```

Verify the working directory.

```bash
pwd
```

Output:

```bash
/data/home/xli49/trt/TensorRT-LLM
```

## 5. Install TensorRT-LLM

Use precompiled binaries (预编译二进制文件) whenever possible.

```bash
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

Example error:

```bash
No matching distribution found for tensorrt_llm==1.3.0rc16
```

Reason:

The container is based on `1.3.0rc15`, but pip tries to install `1.3.0rc16`.

## 6. Fix Version Mismatch

The forked repository may not contain official tags (标签).

Check current remote:

```bash
git remote -v
```

Output:

```bash
origin https://github.com/lxy-alexander/TensorRT-LLM.git
```

Add NVIDIA upstream repository:

```bash
git remote add upstream \
https://github.com/NVIDIA/TensorRT-LLM.git
```

Fetch tags:

```bash
git fetch upstream --tags
```

List available tags:

```bash
git tag | grep 1.3
```

Output:

```bash
v1.3.0rc0
v1.3.0rc1
...
v1.3.0rc15
```

Verify package version:

```bash
grep version tensorrt_llm/version.py
```

Output:

```bash
version = "1.3.0rc15"
```

The repository version must match the container version. (仓库版本必须与容器版本一致)

## 7. Rebuild Rules

Only rebuild when native code (本地代码) changes.

### 1) Python-Only Changes

Do not rebuild C++ code when only Python files are modified.

```bash
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

Or simply rerun the Python program if TensorRT-LLM is already installed.

### 2) C++ / CUDA Changes

Rebuild when modifying kernels (计算核心), plugins (插件), or C++ source files.

```bash
python3 scripts/build_wheel.py \
  --use_ccache \
  -a "90-real" \
  --skip_building_wheel \
  --linking_install_binary
```

Install again:

```bash
pip install -e .
```

Example:

```bash
$ python3 scripts/build_wheel.py ...
Build finished successfully.
```

## 8. Run Inference

Use the LLM API example for testing.

```bash
python3 examples/llm-api/llm_inference.py
```

Example error:

```bash
mpi4py.MPI.Exception: MPI_ERR_SPAWN:
could not spawn processes
```

Reason:

MPI (Message Passing Interface，消息传递接口) cannot create worker processes.

## 9. Fix MPI_ERR_SPAWN

### 1) Single GPU

Use a single process instead of spawning workers.

```bash
export TLLM_WORKER_USE_SINGLE_PROCESS=1
```

Run inference:

```bash
python3 examples/llm-api/llm_inference.py
```

Or:

```bash
mpirun -n 1 trtllm-serve ...
```

### 2) Multi-GPU

Use static MPI startup instead of dynamic spawning.

```bash
mpirun -n 4 trtllm-serve ...
```

The number of MPI processes should match the number of workers. (MPI 进程数应与 worker 数一致)

## 10. Enable Debug Logs

Verbose logging (详细日志) helps diagnose runtime issues.

```bash
export TLLM_LOG_LEVEL=verbose

export TLLM_LOG_LEVEL_BY_MODULE="debug:_torch"
```

Run with logging:

```bash
mpirun -n 1 \
python3 examples/llm-api/llm_inference.py \
2>&1 | grep "alexander lee"
```

Example:

```bash
[DEBUG] Loading checkpoint ...
[DEBUG] Creating executor ...
```

## 11. Check Environment Variables

Verify all TensorRT-LLM related variables.

```bash
env | grep -E '^(TLLM|TRTLLM)_'
```

Output:

```bash
TLLM_LOG_LEVEL=verbose
TLLM_WORKER_USE_SINGLE_PROCESS=1
TLLM_LOG_LEVEL_BY_MODULE=debug:_torch
```

Environment variables control runtime behavior. (环境变量控制运行时行为)

