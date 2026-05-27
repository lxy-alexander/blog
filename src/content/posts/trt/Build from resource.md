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



Use conda to install git lfs in base conda

conda install -c conda-forge git git-lfs -y
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull





Pull

apptainer pull tensorrt_llm_devel_1.3.0rc15.sif docker://nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc15





进入容器
PROJECT_PATH="/data/home/xli49/trt/TensorRT-LLM"

apptainer shell --nv \
  --bind "${PROJECT_PATH}:${PROJECT_PATH}" \
  --pwd "${PROJECT_PATH}" \
  ../container/tensorrt_llm_devel_1.3.0rc15.sif



pwd
/data/home/xli49/trt/TensorRT-LLM







TRTLLM_USE_PRECOMPILED=1 pip install -e .

-> No matching distribution found for tensorrt_llm==1.3.0rc16

但可用版本列表里最高只有：1.3.0rc15

Apptainer> git checkout v1.3.0rc15
error: pathspec 'v1.3.0rc15' did not match any file(s) known to git
Apptainer> 
Apptainer> git remote -v
origin  https://github.com/lxy-alexander/TensorRT-LLM.git (fetch)
origin  https://github.com/lxy-alexander/TensorRT-LLM.git (push)
Apptainer> 
Apptainer> git remote add upstream https://github.com/NVIDIA/TensorRT-LLM.git
Apptainer> git remote -v
origin  https://github.com/lxy-alexander/TensorRT-LLM.git (fetch)
origin  https://github.com/lxy-alexander/TensorRT-LLM.git (push)
upstream        https://github.com/NVIDIA/TensorRT-LLM.git (fetch)
upstream        https://github.com/NVIDIA/TensorRT-LLM.git (push)
Apptainer> git fetch upstream --tags

Apptainer> git tag | grep 1.3
v1.3.0rc0
v1.3.0rc1
v1.3.0rc10
v1.3.0rc11
v1.3.0rc12
v1.3.0rc12.post1
v1.3.0rc13
v1.3.0rc14
v1.3.0rc15

Apptainer> grep __version__ tensorrt_llm/version.py
__version__ = "1.3.0rc15"





编译

1）只改 Python 代码
 不需要重新 `build_wheel.py`。继续用：

```
TRTLLM_USE_PRECOMPILED=1 pip install -e .
```

或已经装过就直接改代码运行。

2）改了 C++ / CUDA / plugin / kernel
 需要重新编译：

```
python3 scripts/build_wheel.py \
  --use_ccache \
  -a "90-real" \
  --skip_building_wheel \
  --linking_install_binary

pip install -e .
```





运行

python3 examples/llm-api/llm_inference.py

-》mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes





fix :

单卡：export TLLM_WORKER_USE_SINGLE_PROCESS=1 : TP=1 时，不要 spawn worker，直接在当前 Python 进程里跑。

export TLLM_WORKER_USE_SINGLE_PROCESS=1 或者 mpirun -n 1 trtllm-serve ...

多卡/4 worker：不要用它，要用 mpirun/srun 静态启动. 
mpirun -n 4 trtllm-serve ...



日志
export TLLM_LOG_LEVEL=verbose
export TLLM_LOG_LEVEL_BY_MODULE="debug:_torch"
mpirun -n 1 python3 examples/llm-api/llm_inference.py 2>&1 | grep "alexander lee"



查看环境变量
Apptainer> env | grep -E '^(TLLM|TRTLLM)_'
TLLM_LOG_LEVEL=verbose
TLLM_WORKER_USE_SINGLE_PROCESS=1
TLLM_LOG_LEVEL_BY_MODULE=debug:_torch



