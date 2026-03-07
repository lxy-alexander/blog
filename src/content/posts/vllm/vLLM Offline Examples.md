---
title: "vLLM Offline Examples"
published: 2026-02-25
description: "vLLM Offline Examples"
image: ""
tags: ["vllm","vLLM Offline Examples"]
category: vllm
draft: false
lang: ""
---







`LLMEngine` 在 v1 ，路径是 `vllm/v1/engine/llm_engine.py`。

从代码可以看出：

1. **入口层**：`llm.py` 从 `vllm.v1.engine.llm_engine` 导入 `LLMEngine`（第 89 行）。

2. **LLMEngine 的定位**：类注释写的是 `"Legacy LLMEngine for backwards compatibility"`，说明它是为了兼容旧 API 的封装，但实现已经放在 v1 包下。

3. **内部依赖**：`LLMEngine` 内部用的是 v1 的组件：
   - `EngineCoreClient`：真正的推理核心
   - `InputProcessor`、`OutputProcessor`：输入/输出处理
   - `Executor`：执行器

4. **与 entrypoint 的交互**：`llm.py` 里的 `_run_engine` 调用的是 `self.llm_engine.step()`，而这个 `step()` 会通过 `engine_core.get_output()` 和 `output_processor.process_outputs()` 完成推理。

所以当前架构是：entrypoint（`llm.py`）→ v1 的 `LLMEngine` → v1 的 `EngineCoreClient` 等，整个 engine 链路都在 v1 里。









## Async LLM Streaming





Audio Language
Automatic Prefix Caching
Basic
Batch LLM Inference
Chat With Tools
Context Extension
Data Parallel
Disaggregated Prefill V1
Disaggregated Prefill
Encoder Decoder Multimodal
KV Load Failure Recovery Test
LLM Engine Example
LLM Engine Reset Kv
Load Sharded State
Logits Processor
LoRA With Quantization Inference
Metrics
Mistral-Small
MLPSpeculator
MultiLoRA Inference
New Weight Syncing
Offline Inference with the OpenAI Batch file format
Pause Resume
Prefix Caching
Prompt Embed Inference
Qwen2.5-Omni Offline Inference Examples
Qwen3 Omni
Qwen 1M
Reproducibility
RLHF
RLHF Colocate
RLHF Online Quant
RLHF Utils
Run One Batch
Save Sharded State
Simple Profiling
Skip Loading Weights In Engine Init
Spec Decode
Structured Outputs
Torchrun Dp Example
Torchrun Example
Vision Language
Vision Language Multi Image

