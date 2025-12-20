---
title: vLLM
published: 2025-12-19
description: "vLLM"
image: "./cover.jpeg"
tags: ["Blogging", "vLLM"]
category: Guides
draft: false

---



# vLLM Quickstart

## I. Overview 

- vLLM Quickstart is an introductory workflow that helps users quickly install vLLM and run high-performance large language model inference in offline and online serving modes.
- vLLM 快速入门是一套引导式流程，帮助用户快速安装 vLLM，并以离线和在线服务两种模式运行高性能大语言模型推理。

## II. Responsibilities and Applicability

- This workflow provides a minimal and reliable path to verify that vLLM is correctly installed and operational on the target system.
- 该流程提供了一条最小且可靠的路径，用于验证 vLLM 是否在目标系统上正确安装并可正常运行。
- It is typically used during initial setup, environment validation, or early-stage evaluation of inference performance and serving behavior.
- 它通常用于首次环境搭建、系统验证，或对推理性能与服务行为进行初步评估时。
- It is most relevant when efficient GPU utilization, high-throughput batching, OpenAI API compatibility, or scalable inference is required.
- 当需要高效 GPU 利用率、高吞吐批处理、OpenAI API 兼容性或可扩展推理能力时，该流程尤为重要。

## III. Representative Execution Paths or Usage Workflows

### Environment Setup and Installation Workflow

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Execution behavior explanation.
This workflow creates an isolated Python environment and installs vLLM with a PyTorch backend automatically matched to the detected CUDA or ROCm drivers.
该流程创建一个隔离的 Python 环境，并根据检测到的 CUDA 或 ROCm 驱动自动匹配并安装合适的 PyTorch 后端。

Performance characteristics explanation.
Correct backend selection enables optimized attention kernels and execution paths, directly affecting inference latency and throughput.
正确的后端选择可以启用优化后的注意力算子和执行路径，直接影响推理延迟和吞吐量。

Resource usage characteristics explanation.
Installation itself has negligible runtime cost, but determines which GPU kernels and memory optimizations are available during execution.
安装过程本身几乎不消耗运行时资源，但会决定推理阶段可用的 GPU 内核和内存优化能力。

------

### Offline Batch Inference Workflow

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Execution behavior explanation.
Prompts are submitted to the vLLM engine, which performs prompt prefill followed by token-by-token decoding using dynamic batching.
输入提示被提交到 vLLM 引擎，引擎先执行 prompt 预填充阶段，再通过动态批处理进行逐 token 解码。

Performance characteristics explanation.
Batching across multiple prompts improves GPU utilization and increases overall throughput compared to sequential generation.
对多个提示进行批处理可以提高 GPU 利用率，并显著提升整体吞吐量，相比顺序生成更高效。

Resource usage characteristics explanation.
GPU memory usage is dominated by model weights and KV cache blocks that grow with sequence length and batch size.
GPU 内存主要由模型权重和 KV cache 占用，其规模随序列长度和批大小增长。

------

### OpenAI-Compatible Online Serving Workflow

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Execution behavior explanation.
This workflow launches an HTTP server that translates OpenAI-style API requests into internal vLLM inference tasks.
该流程启动一个 HTTP 服务，将 OpenAI 风格的 API 请求转换为 vLLM 内部的推理任务。

Performance characteristics explanation.
The server continuously batches concurrent requests, balancing latency and throughput under varying load conditions.
服务端会持续对并发请求进行批处理，在不同负载条件下平衡延迟与吞吐量。

Resource usage characteristics explanation.
Memory usage scales with the number of active sequences and their KV cache requirements, while CPU usage handles request parsing and scheduling.
内存占用随活跃序列数量及其 KV cache 需求增长，CPU 主要用于请求解析和调度。

------

### Attention Backend Selection Workflow

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

Execution behavior explanation.
This configuration enforces a specific attention backend, overriding automatic selection logic in vLLM.
该配置强制使用指定的注意力后端，覆盖 vLLM 的自动选择逻辑。

Performance characteristics explanation.
Different backends provide different performance trade-offs depending on hardware, model shape, and sequence length.
不同后端在不同硬件、模型结构和序列长度下具有不同的性能权衡。

Resource usage characteristics explanation.
Backend choice affects shared memory usage, kernel fusion opportunities, and maximum achievable batch size.
后端选择会影响共享内存使用、内核融合机会以及可支持的最大批大小。

## IV. Key Invariants and Mental Model

- The engine must maintain a consistent mapping between active sequences and their KV cache blocks throughout inference.
- 推理过程中，引擎必须始终保持活跃序列与其 KV cache 块之间的一致映射关系。
- Request batching and scheduling must preserve correctness while maximizing shared computation.
- 请求批处理和调度在保证正确性的同时，必须尽可能最大化共享计算。
- A useful mental model is to view vLLM as a scheduler-driven inference system that separates prompt processing from decoding and aggressively reuses memory.
- 一个有效的心智模型是将 vLLM 视为由调度器驱动的推理系统，它将 prompt 处理与解码分离，并积极复用内存。
- Common pitfalls include mismatched CUDA versions, unintended generation configuration overrides, and forcing incompatible attention backends.
- 常见问题包括 CUDA 版本不匹配、生成配置被无意覆盖，以及强制使用不兼容的注意力后端。