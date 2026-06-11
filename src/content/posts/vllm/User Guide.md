---
title: "User Guide"
published: 2026-06-08
description: "User Guide"
image: ""
tags: ["vllm","User Guide"]
category: vllm
draft: false
lang: ""
createdAt: "2026-06-08T18:59:47.848.078811869Z"
---



# vLLM Quickstart

## 1. Prerequisites (前置条件)

vLLM runs on **Linux** with **Python 3.10–3.13**, and also supports **macOS** via vLLM-Metal for Apple Silicon GPU acceleration (苹果芯片 GPU 加速).

## 2. Installation (安装)

Use **uv** (a fast Python environment manager, 快速的 Python 环境管理器) to set up vLLM in one shot — it auto-detects your CUDA driver and picks the right PyTorch backend (后端).

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Quick one-liner without creating a permanent environment:

```bash
uv run --with vllm vllm --help
```

## 3. Offline Batched Inference (离线批量推理)

Use the **`LLM`** class to load a model and **`SamplingParams`** to specifies the parameters for the sampling process — then call `llm.generate()` to process a list of prompts with high throughput (高吞吐量).

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
```

**Key gotcha (易错点):** By default vLLM uses the model's `generation_config.json` from Hugging Face; pass `generation_config="vllm"` if you want vLLM's own defaults.

**Chat models (对话模型):** `llm.generate()` does **not** auto-apply the chat template (聊天模板) — use `llm.chat()` instead for Instruct/Chat models.

## 4. Online Serving (在线服务)

vLLM provides an **OpenAI-compatible HTTP server** (兼容 OpenAI 协议的 HTTP 服务), so it's a drop-in replacement (无缝替换) for any OpenAI client — defaults to `http://localhost:8000`.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Optional flags worth remembering: `--host`, `--port`, `--api-key` (supports multiple keys for key rotation, 密钥轮换), and `--generation-config vllm` to disable HF's default sampling.

## 5. Querying the Server (调用服务)

**Completions endpoint (文本补全接口):**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

**Chat Completions endpoint (对话补全接口)** — better for multi-turn context (多轮上下文):

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

Or just point the **OpenAI Python SDK** at the local URL with a dummy key:

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
```

## 6. Attention Backends (注意力后端)

vLLM **auto-picks the fastest Attention backend** for your hardware, but you can override it via `--attention-backend` when you want full control.

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN
```

Common choices to remember:

-   **NVIDIA CUDA (英伟达):** `FLASH_ATTN`, `FLASHINFER`
-   **AMD ROCm:** `TRITON_ATTN`, `ROCM_AITER_FA`, `TRITON_MLA`

## 7. Interview-Ready Takeaways (面试速记)

-   **What is vLLM?** → A high-throughput LLM serving engine (高吞吐量大模型推理引擎) with an OpenAI-compatible API.
-   **Two usage modes (两种用法):** offline batch inference (`LLM` class) and online serving (`vllm serve`).
-   **Why it's fast?** → PagedAttention (分页注意力) + continuous batching (连续批处理) + pluggable attention backends.
-   **Drop-in replacement:** any OpenAI SDK code works by just swapping the `base_url`.
