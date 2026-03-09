---
title: "offline inference"
published: 2026-02-19
description: "offline inference"
image: ""
tags: ["vllm","offline inference"]
category: vllm
draft: false
lang: ""
---



# **I. vLLM Inference Examples — A Structured Learning Guide**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> This note organizes <span style="color:#E8600A;font-weight:700">44 vLLM inference examples</span> into 8 functional categories, covering everything from basic Inference Modes (推理模式) to advanced Training Integration (训练集成) and Observability (可观测性). Each example is explained with its purpose, core mechanism, and representative code pattern — designed to build a <span style="color:#E8600A;font-weight:700">complete mental model</span> of how a production-grade LLM serving system is structured. </div>

------

## 1. Inference Modes (推理模式)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples demonstrate different execution pipelines (执行管线) for running model inference — including <span style="color:#E8600A;font-weight:700">Streaming (流式生成)</span>, <span style="color:#E8600A;font-weight:700">Batching (批处理)</span>, and <span style="color:#E8600A;font-weight:700">Offline Job Processing (离线任务推理)</span>. The choice of mode directly impacts latency (延迟), throughput (吞吐量), and resource utilization (资源利用率). </div>

### 1) Async LLM Streaming

<span style="color:#2980B9">Purpose:</span> Generate tokens and return them to the client <span style="color:#E8600A;font-weight:700">incrementally</span> as they are produced, rather than waiting for the full response.

<span style="color:#2980B9">Core Mechanism:</span> Uses Python's <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AsyncLLMEngine</code> with an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async for</code> loop to consume a token stream. Each yielded chunk contains a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">RequestOutput</code> with partial text.

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model="..."))

async def stream_response(prompt: str):
    sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
    # generate() returns an async generator (异步生成器)
    async for output in engine.generate(prompt, sampling_params, request_id="req-1"):
        # Each output.outputs[0].text is a partial completion
        print(output.outputs[0].text, end="", flush=True)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">AsyncLLMEngine</code> runs the model loop in a background thread and exposes an async interface. This is the foundation for building <span style="color:#E8600A;font-weight:700">real-time chat APIs</span> (e.g., OpenAI-compatible streaming endpoints).</div>

------

### 2) Batch LLM Inference

<span style="color:#2980B9">Purpose:</span> Submit <span style="color:#E8600A;font-weight:700">multiple prompts simultaneously</span> and collect all outputs after processing. Maximizes GPU utilization (GPU 利用率) when latency is not critical.

<span style="color:#2980B9">Core Mechanism:</span> Uses the synchronous <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LLM</code> class with a list of prompts. The engine internally groups requests into batches using its <span style="color:#E8600A;font-weight:700">Continuous Batching (连续批处理)</span> scheduler.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

prompts = [
    "Explain transformers in one sentence.",
    "What is KV cache?",
    "Define tensor parallelism.",
]

# generate() accepts a list → returns a list of RequestOutput
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}\n")
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Unlike Async Streaming, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LLM.generate()</code> is <span style="color:#C0392B;font-weight:600">blocking</span> — it returns only when all prompts are finished. Best suited for offline evaluation pipelines (离线评测管线) or data preprocessing.</div>

------

### 3) Run One Batch

<span style="color:#2980B9">Purpose:</span> Execute a <span style="color:#E8600A;font-weight:700">single, controlled forward pass</span> — useful for debugging scheduling behavior or testing engine initialization.

<span style="color:#2980B9">Core Mechanism:</span> Directly invokes the engine's step function (执行步函数) for fine-grained control over the execution loop, bypassing the high-level <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">generate()</code> API.

```python
# Low-level engine step — processes exactly one batch
engine_output = engine.step()  # returns List[RequestOutput]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This pattern is primarily used in <span style="color:#E8600A;font-weight:700">unit tests</span> and <span style="color:#E8600A;font-weight:700">engine-level debugging</span>. Most production users should prefer <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">generate()</code>.</div>

------

### 4) Offline Inference with the OpenAI Batch File Format

<span style="color:#2980B9">Purpose:</span> Process large-scale inference jobs using the <span style="color:#E8600A;font-weight:700">OpenAI Batch API format</span> (JSONL files), enabling compatibility with existing batch pipelines.

<span style="color:#2980B9">Core Mechanism:</span> Reads a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">.jsonl</code> file where each line is a JSON request object, runs inference, and writes results to an output JSONL file.

```python
# Input JSONL format (每行是一个独立请求)
# {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",
#  "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}}

from vllm.entrypoints.openai.run_batch import run_batch

run_batch(
    input_file="requests.jsonl",
    output_file="results.jsonl",
    model="meta-llama/Llama-3-8B-Instruct"
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The OpenAI Batch format uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">custom_id</code> fields so results can be matched back to inputs even if processed out of order. This is critical for <span style="color:#E8600A;font-weight:700">idempotent job processing (幂等任务处理)</span>.</div>

------

### 5) Prompt Embed Inference

<span style="color:#2980B9">Purpose:</span> Pass <span style="color:#E8600A;font-weight:700">pre-computed embeddings</span> (预计算嵌入向量) directly as model input, bypassing the tokenizer and embedding layer.

<span style="color:#2980B9">Core Mechanism:</span> Accepts a tensor of token embeddings instead of a string prompt — useful for models that receive continuous inputs (连续输入) from an upstream encoder.

```python
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="...", skip_tokenizer_init=True)

# Pre-computed embeddings shape: [seq_len, hidden_dim]
embeddings = torch.randn(10, 4096)

outputs = llm.generate(
    prompts=None,
    prompt_token_ids=None,
    prompt_embeds=embeddings,  # Pass embeddings directly
    sampling_params=SamplingParams(max_tokens=64)
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This is the entry point for <span style="color:#E8600A;font-weight:700">multimodal fusion pipelines</span> where image or audio encoders produce embeddings that are fed into the language model decoder.</div>

------

### 6) LLM Engine Example

<span style="color:#2980B9">Purpose:</span> Demonstrate direct usage of the <span style="color:#E8600A;font-weight:700">LLMEngine (推理引擎核心类)</span> — the lower-level API underneath the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LLM</code> convenience wrapper.

<span style="color:#2980B9">Core Mechanism:</span> Manually add requests via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">engine.add_request()</code> and drive the loop with repeated <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">engine.step()</code> calls.

```python
from vllm import LLMEngine, EngineArgs, SamplingParams

engine = LLMEngine.from_engine_args(EngineArgs(model="..."))
engine.add_request("req-0", "Hello, world!", SamplingParams(max_tokens=50))

while engine.has_unfinished_requests():
    request_outputs = engine.step()
    for output in request_outputs:
        if output.finished:
            print(output.outputs[0].text)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LLMEngine</code> gives full control over the <span style="color:#E8600A;font-weight:700">request lifecycle (请求生命周期)</span>. It is the recommended interface for building custom serving frameworks on top of vLLM.</div>

------

### 7) Chat With Tools

<span style="color:#2980B9">Purpose:</span> Enable <span style="color:#E8600A;font-weight:700">Tool Calling (工具调用)</span> — allow the model to invoke external functions by generating structured JSON tool-call outputs.

<span style="color:#2980B9">Core Mechanism:</span> Passes a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">tools</code> list in the chat completion request. The model outputs a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tool_calls</code> field instead of plain text when it decides to call a function.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="...",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"   # model decides whether to call a tool
)
# response.choices[0].message.tool_calls → structured call output
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Tool calling requires the model to be fine-tuned for function-calling formats (e.g., Llama-3-Instruct, Mistral-Instruct). <span style="color:#C0392B;font-weight:600">Base models will not reliably produce valid tool_calls JSON.</span></div>

------

### 8) Structured Outputs

<span style="color:#2980B9">Purpose:</span> Force the model to produce output that <span style="color:#E8600A;font-weight:700">conforms to a JSON schema (JSON 模式约束)</span>, ensuring parseable, typed results.

<span style="color:#2980B9">Core Mechanism:</span> Uses <span style="color:#E8600A;font-weight:700">Guided Decoding (引导解码)</span> via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">GuidedDecodingParams</code> or the OpenAI-compatible <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">response_format</code> field to constrain the token sampling space at each step.

```python
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

class Person(BaseModel):
    name: str
    age: int
    city: str

guided = GuidedDecodingParams(json=Person.model_json_schema())
params = SamplingParams(guided_decoding=guided, max_tokens=100)

llm = LLM(model="...")
output = llm.generate("Extract: John is 30 years old from New York.", params)
# Output is guaranteed to be valid JSON matching Person schema
import json
person = Person(**json.loads(output[0].outputs[0].text))
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> vLLM uses <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">outlines</code> or <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">lm-format-enforcer</code> backends for guided decoding. This adds a small overhead per token step but <span style="color:#E8600A;font-weight:700">eliminates all JSON parsing errors</span> in production pipelines.</div>

------

## 2. Distributed / System Architecture (分布式与系统架构)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples show how to scale inference across <span style="color:#E8600A;font-weight:700">multiple GPUs or nodes</span> and how to decompose the inference pipeline into separate stages. Key strategies include <span style="color:#E8600A;font-weight:700">Tensor Parallelism (张量并行)</span>, <span style="color:#E8600A;font-weight:700">Data Parallelism (数据并行)</span>, and <span style="color:#E8600A;font-weight:700">Disaggregated Prefill (分离式预填充, Prefill-Decode Separation)</span>. </div>

### 1) Data Parallel

<span style="color:#2980B9">Purpose:</span> Run <span style="color:#E8600A;font-weight:700">multiple independent engine replicas</span> in parallel, each handling a different subset of requests, to maximize overall throughput.

<span style="color:#2980B9">Core Mechanism:</span> Launches N worker processes, each with a full model copy. A front-end router (路由器) distributes incoming requests across replicas using load balancing (负载均衡).

```python
# Launch with data_parallel_size=4 (4 independent replicas)
llm = LLM(
    model="meta-llama/Llama-3-8B",
    data_parallel_size=4,      # 4 model replicas
    tensor_parallel_size=1     # each replica uses 1 GPU
)
```

| Strategy                                                     | Splits        | Best For                             |
| ------------------------------------------------------------ | ------------- | ------------------------------------ |
| <span style="color:#E8600A;font-weight:700">Data Parallel (数据并行)</span> | Requests      | High QPS, smaller models             |
| Tensor Parallel (张量并行)                                   | Model weights | Large models that don't fit on 1 GPU |
| Pipeline Parallel (流水线并行)                               | Model layers  | Very deep models across nodes        |

------

### 2) Torchrun Example

<span style="color:#2980B9">Purpose:</span> Launch vLLM using <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun</code> for <span style="color:#E8600A;font-weight:700">multi-GPU / multi-node distributed inference</span> with PyTorch's native distributed backend.

<span style="color:#2980B9">Core Mechanism:</span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun</code> sets up <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">RANK</code>, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">WORLD_SIZE</code>, and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">MASTER_ADDR</code> environment variables, which vLLM reads to configure its distributed communication (分布式通信) via NCCL.

```bash
# Launch tensor-parallel inference on 4 GPUs (single node)
torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  inference_script.py --tensor-parallel-size 4
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> vLLM can also launch distributed workers internally without <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun</code> via its <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">ray</code> backend. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun</code> when you need tighter integration with existing PyTorch training infrastructure (训练基础设施).</div>

------

### 3) Torchrun DP Example

<span style="color:#2980B9">Purpose:</span> Combine <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torchrun</code> with <span style="color:#E8600A;font-weight:700">Data Parallelism</span> — run multiple replicas each launched as a separate torchrun group.

<span style="color:#2980B9">Core Mechanism:</span> Each process group handles its own model shard; a coordinator broadcasts requests to the correct group based on <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">RANK</code>.

------

### 4) Disaggregated Prefill

<span style="color:#2980B9">Purpose:</span> Separate the <span style="color:#E8600A;font-weight:700">Prefill Phase (预填充阶段)</span> — which processes the input prompt — from the <span style="color:#E8600A;font-weight:700">Decode Phase (解码阶段)</span> onto different hardware, reducing latency interference.

<span style="color:#2980B9">Core Mechanism:</span> A dedicated set of <span style="color:#E8600A;font-weight:700">Prefill Workers (预填充节点)</span> computes KV caches for incoming prompts and ships them over the network to <span style="color:#E8600A;font-weight:700">Decode Workers (解码节点)</span>, which then generate tokens.

```
┌─────────────────────────┐      KV Cache Transfer      ┌──────────────────────────┐
│   Prefill Workers        │  ─────────────────────────► │   Decode Workers          │
│  (compute-heavy, batchy) │      (over RDMA / NVLink)   │  (memory-bandwidth bound) │
└─────────────────────────┘                              └──────────────────────────┘
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Disaggregated prefill is a key technique for reducing <span style="color:#E8600A;font-weight:700">Time To First Token (TTFT, 首字节延迟)</span> in production serving, since prefill and decode have very different computational profiles and can be sized independently.</div>

------

### 5) Disaggregated Prefill V1

<span style="color:#2980B9">Purpose:</span> An updated implementation of Disaggregated Prefill with improved <span style="color:#E8600A;font-weight:700">KV cache transfer protocols (KV cache 传输协议)</span> and better scheduler coordination.

<span style="color:#2980B9">Core Mechanism:</span> V1 introduces a unified <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">KVTransferAgent</code> that manages bidirectional cache migration and handles failures during transfer more gracefully.

------

## 3. Performance Optimization (性能优化)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples target <span style="color:#E8600A;font-weight:700">throughput, latency, and memory efficiency</span>. Core techniques include <span style="color:#E8600A;font-weight:700">KV Cache Reuse (KV cache 复用)</span>, <span style="color:#E8600A;font-weight:700">Speculative Decoding (推测解码)</span>, and <span style="color:#E8600A;font-weight:700">Context Length Extension (上下文长度扩展)</span>. </div>

### 1) Prefix Caching

<span style="color:#2980B9">Purpose:</span> Manually manage a <span style="color:#E8600A;font-weight:700">Prefix Cache (前缀缓存)</span> to reuse KV computations across requests that share a common prompt prefix (e.g., system prompt).

<span style="color:#2980B9">Core Mechanism:</span> The engine hashes prompt prefixes and stores their KV blocks. On a cache hit (缓存命中), it skips recomputation of the matching prefix tokens.

```python
llm = LLM(model="...", enable_prefix_caching=True)

# Both requests share the same system prompt → prefix KV is computed once
system = "You are a helpful assistant. Answer concisely."
outputs = llm.generate([
    f"{system}\n\nQ: What is 2+2?",
    f"{system}\n\nQ: What is the capital of France?",
])
# Second request reuses KV cache for system prompt tokens → lower latency
```

------

### 2) Automatic Prefix Caching

<span style="color:#2980B9">Purpose:</span> Enable <span style="color:#E8600A;font-weight:700">transparent, automatic</span> prefix cache management — no manual intervention required. The engine handles cache eviction (缓存驱逐) and promotion automatically.

<span style="color:#2980B9">Core Mechanism:</span> Uses an <span style="color:#E8600A;font-weight:700">LRU eviction policy (最近最少使用驱逐策略)</span> on a block-level hash table. Compatible with most vLLM features including chunked prefill (分块预填充).

```python
# Just set enable_prefix_caching=True — no other changes needed
llm = LLM(model="...", enable_prefix_caching=True)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Automatic Prefix Caching gives the best results when requests share long, stable prefixes (e.g., few-shot examples, RAG context). Cache Hit Rate (缓存命中率) can be monitored via the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">/metrics</code> endpoint.</div>

------

### 3) Spec Decode

<span style="color:#2980B9">Purpose:</span> Reduce decoding latency using <span style="color:#E8600A;font-weight:700">Speculative Decoding (推测解码)</span> — a small draft model proposes multiple tokens at once, and the large target model verifies them in parallel.

<span style="color:#2980B9">Core Mechanism:</span> A lightweight <span style="color:#E8600A;font-weight:700">Draft Model (草稿模型)</span> generates a token sequence of length K. The target model runs a single batched forward pass to accept or reject each draft token. Accepted tokens are committed; rejected tokens trigger a fallback.

```python
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",       # target model (目标模型)
    speculative_model="meta-llama/Llama-3-8B",     # draft model  (草稿模型)
    num_speculative_tokens=5,   # propose 5 tokens per step
    speculative_draft_tensor_parallel_size=1,
)
```

| Metric                | Without Spec Decode | With Spec Decode           |
| --------------------- | ------------------- | -------------------------- |
| Draft tokens per step | 1                   | 5 (proposed)               |
| Target model calls    | N tokens            | ~N/acceptance_rate steps   |
| Latency reduction     | baseline            | up to 2–3× on greedy tasks |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Speculative decoding preserves <span style="color:#E8600A;font-weight:700">exact output distribution</span> — it is mathematically equivalent to sampling from the target model alone. Speedup depends on the <span style="color:#E8600A;font-weight:700">Token Acceptance Rate (token 接受率)</span>, which is higher for predictable/repetitive text.</div>

------

### 4) MLPSpeculator

<span style="color:#2980B9">Purpose:</span> Use a lightweight <span style="color:#E8600A;font-weight:700">MLP-based speculative head (MLP 推测头)</span> attached to the target model itself — eliminating the need for a separate draft model.

<span style="color:#2980B9">Core Mechanism:</span> A small MLP is trained to predict the next K tokens from the current hidden state (隐藏状态), all within the same model forward pass. This avoids extra KV cache memory for a separate draft model.

```python
llm = LLM(
    model="ibm/llama-3-8b-accelerator",  # model with built-in MLP speculator head
    speculative_model="[mlp_speculator]", # special keyword to use embedded head
    num_speculative_tokens=3,
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> MLPSpeculator is more memory-efficient than a full draft model and has <span style="color:#C0392B;font-weight:600">lower overhead</span> when batch sizes are large. However, its acceptance rate may be lower than a well-aligned draft model.</div>

------

### 5) Context Extension

<span style="color:#2980B9">Purpose:</span> Extend the model's effective <span style="color:#E8600A;font-weight:700">Context Window (上下文窗口)</span> beyond its training length using positional encoding interpolation techniques.

<span style="color:#2980B9">Core Mechanism:</span> Applies <span style="color:#E8600A;font-weight:700">RoPE Scaling (旋转位置编码缩放)</span> — either linear or dynamic — to rescale positional frequencies so that longer sequences fall within the model's trained positional range.

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    rope_scaling={
        "type": "dynamic",          # dynamic NTK scaling (动态 NTK 缩放)
        "factor": 4.0               # extend context 4× beyond training length
    },
    max_model_len=131072            # set new maximum context length (128K tokens)
)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Context extension works best when combined with <span style="color:#E8600A;font-weight:700">FlashAttention (闪存注意力)</span> and chunked prefill (分块预填充). Without these, VRAM usage scales as O(n²) with sequence length.</div>

------

## 4. Model Capability / Modalities (模型能力与多模态)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples show how vLLM supports models beyond text-only generation — integrating <span style="color:#E8600A;font-weight:700">Vision Encoders (视觉编码器)</span>, <span style="color:#E8600A;font-weight:700">Audio Encoders (音频编码器)</span>, and <span style="color:#E8600A;font-weight:700">Encoder-Decoder Architectures (编码器-解码器架构)</span> into the inference pipeline. </div>

### 1) Vision Language

<span style="color:#2980B9">Purpose:</span> Run inference on <span style="color:#E8600A;font-weight:700">Vision-Language Models (视觉语言模型)</span> such as LLaVA, InternVL, or Qwen-VL with single image inputs.

<span style="color:#2980B9">Core Mechanism:</span> The image is encoded by a Vision Encoder (e.g., CLIP ViT) into patch embeddings (图像块嵌入), which are then interleaved with text tokens in the language model's input sequence.

```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")
image = Image.open("photo.jpg")

outputs = llm.generate({
    "prompt": "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    "multi_modal_data": {"image": image},   # pass image alongside prompt
}, SamplingParams(max_tokens=256))
```

------

### 2) Vision Language Multi Image

<span style="color:#2980B9">Purpose:</span> Handle prompts containing <span style="color:#E8600A;font-weight:700">multiple images</span> — enabling tasks like image comparison, visual storytelling, or multi-frame video analysis.

<span style="color:#2980B9">Core Mechanism:</span> Each image is encoded separately and its embeddings are inserted at the corresponding <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px"><image></code> placeholder positions in the prompt token sequence.

```python
outputs = llm.generate({
    "prompt": "USER: <image>\n<image>\nHow do these two images differ?\nASSISTANT:",
    "multi_modal_data": {"image": [image1, image2]},  # list of images
}, SamplingParams(max_tokens=256))
```

------

### 3) Audio Language

<span style="color:#2980B9">Purpose:</span> Process <span style="color:#E8600A;font-weight:700">audio inputs (音频输入)</span> alongside text — enabling speech-to-text, audio QA, and spoken dialogue tasks.

<span style="color:#2980B9">Core Mechanism:</span> An audio encoder (e.g., Whisper encoder) converts raw waveforms into feature embeddings, which are concatenated with text embeddings before the language model decoder.

```python
llm = LLM(model="Qwen/Qwen2-Audio-7B-Instruct")

outputs = llm.generate({
    "prompt": "<|audio_bos|><|AUDIO|><|audio_eos|>\nTranscribe and summarize.",
    "multi_modal_data": {"audio": (audio_array, sample_rate)},
}, SamplingParams(max_tokens=256))
```

------

### 4) Encoder Decoder Multimodal

<span style="color:#2980B9">Purpose:</span> Run inference on <span style="color:#E8600A;font-weight:700">Encoder-Decoder models (编码器-解码器模型)</span> like BLIP-2 or Flamingo, where a separate encoder processes multimodal inputs and a decoder generates text.

<span style="color:#2980B9">Core Mechanism:</span> Unlike decoder-only LLMs, these models use cross-attention (交叉注意力) between the encoder's output and the decoder's hidden states. vLLM manages separate KV caches for encoder and decoder.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Encoder-decoder models have <span style="color:#C0392B;font-weight:600">two separate KV caches</span> — one for the encoder (computed once per input) and one for the decoder (grown token by token). This doubles memory planning complexity.</div>

------

### 5) Qwen3 Omni

<span style="color:#2980B9">Purpose:</span> Run Qwen3-Omni, an <span style="color:#E8600A;font-weight:700">omni-modal model (全模态模型)</span> that accepts text, image, audio, and video inputs simultaneously.

------

### 6) Qwen2.5-Omni Offline Inference

<span style="color:#2980B9">Purpose:</span> Offline batch inference for <span style="color:#E8600A;font-weight:700">Qwen2.5-Omni</span> with pre-loaded multimodal data, optimized for high-throughput processing of large multimodal datasets.

------

### 7) Qwen 1M

<span style="color:#2980B9">Purpose:</span> Inference with <span style="color:#E8600A;font-weight:700">Qwen's 1M-token context model</span> — demonstrating vLLM's ability to handle extremely long sequences for document-level reasoning.

<span style="color:#2980B9">Core Mechanism:</span> Requires chunked prefill (分块预填充) and optimized memory management to fit 1M-token contexts within GPU VRAM constraints.

------

### 8) Mistral-Small

<span style="color:#2980B9">Purpose:</span> Demonstrate inference with <span style="color:#E8600A;font-weight:700">Mistral-Small</span>, showcasing vLLM's support for the Mistral model family's unique tokenizer and sliding window attention (滑动窗口注意力).

------

## 5. Model Adaptation / Extensions (模型扩展与适配)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples demonstrate how to extend base models with <span style="color:#E8600A;font-weight:700">LoRA Adapters (低秩适配器)</span>, <span style="color:#E8600A;font-weight:700">Custom Logits Processors (自定义 logits 处理器)</span>, and <span style="color:#E8600A;font-weight:700">Dynamic Weight Synchronization (动态权重同步)</span> — enabling fine-tuned or customized model behavior without retraining from scratch. </div>

### 1) LoRA With Quantization Inference

<span style="color:#2980B9">Purpose:</span> Serve a <span style="color:#E8600A;font-weight:700">quantized base model</span> with a <span style="color:#E8600A;font-weight:700">LoRA adapter</span> — combining memory savings from quantization (量化) with task-specific fine-tuning.

<span style="color:#2980B9">Core Mechanism:</span> The base model is loaded in INT4/INT8 precision. LoRA weights (low-rank delta matrices, 低秩增量矩阵) are kept in FP16 and added at inference time via a fused kernel.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="bitsandbytes",   # INT4 quantization (4-bit 量化)
    enable_lora=True,
    max_lora_rank=64
)

lora_req = LoRARequest(
    lora_name="my_adapter",
    lora_int_id=1,
    lora_local_path="/path/to/lora/weights"
)

outputs = llm.generate("Classify the sentiment:", lora_request=lora_req)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Not all quantization formats are LoRA-compatible.</span> GPTQ and AWQ support LoRA via dequantize-then-add patterns, but this may reduce throughput. BitsAndBytes NF4 quantization generally offers the best compatibility.</div>

------

### 2) MultiLoRA Inference

<span style="color:#2980B9">Purpose:</span> Serve <span style="color:#E8600A;font-weight:700">multiple LoRA adapters simultaneously</span> from a single base model — different requests can use different adapters in the same batch.

<span style="color:#2980B9">Core Mechanism:</span> vLLM maintains a pool of loaded LoRA weight tensors and uses a per-request <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">LoRARequest</code> tag to select the correct adapter during the forward pass.

```python
llm = LLM(model="...", enable_lora=True, max_loras=4)  # hold up to 4 adapters in VRAM

# Different requests use different adapters — served in the same batch
lora_a = LoRARequest("sentiment",   1, "/adapters/sentiment")
lora_b = LoRARequest("summarize",   2, "/adapters/summarize")
lora_c = LoRARequest("translation", 3, "/adapters/translation")

outputs = llm.generate(
    ["Classify: I love this!", "Summarize: ...", "Translate: Hello"],
    sampling_params=SamplingParams(max_tokens=64),
    lora_request=[lora_a, lora_b, lora_c]  # per-request adapter assignment
)
```

------

### 3) Custom Logits Processors

<span style="color:#2980B9">Purpose:</span> Inject custom logic to <span style="color:#E8600A;font-weight:700">modify the raw logits (原始 logits)</span> at each generation step — enabling custom constraints, biasing, or token filtering.

<span style="color:#2980B9">Core Mechanism:</span> A user-defined callable receives the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">logits</code> tensor before sampling and returns a modified tensor. Runs on GPU inside the generation loop.

```python
import torch
from vllm import SamplingParams

def no_repeat_ngram_processor(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
    """Block 3-gram repetition by setting repeated tokens to -inf."""
    if len(token_ids) >= 3:
        last_bigram = tuple(token_ids[-2:])
        for i, tok in enumerate(token_ids[:-2]):
            if tuple(token_ids[i+1:i+3]) == last_bigram:
                logits[token_ids[i+2]] = float("-inf")  # suppress repeated token
    return logits

params = SamplingParams(
    logits_processors=[no_repeat_ngram_processor],
    max_tokens=256
)
```

------

### 4) New Weight Syncing

<span style="color:#2980B9">Purpose:</span> <span style="color:#E8600A;font-weight:700">Dynamically update model weights</span> in a running vLLM engine — enabling online learning (在线学习) or hot-swapping updated checkpoints without restarting the server.

<span style="color:#2980B9">Core Mechanism:</span> Uses a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">WeightSyncWorker</code> that receives updated weight tensors over a distributed channel and applies them to the live model in-place, coordinating across all tensor-parallel workers.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Weight syncing is the key mechanism enabling <span style="color:#E8600A;font-weight:700">RLHF colocated training</span> — the trainer updates policy weights and immediately syncs them to the inference engine for online rollout generation.</div>

------

## 6. Reliability / State Management (可靠性与状态管理)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples ensure the system behaves correctly under <span style="color:#E8600A;font-weight:700">failure conditions and long-running workloads</span>. Key mechanisms include <span style="color:#E8600A;font-weight:700">KV Cache Persistence (KV cache 持久化)</span>, <span style="color:#E8600A;font-weight:700">Checkpoint Sharding (检查点分片)</span>, and <span style="color:#E8600A;font-weight:700">Request Pause/Resume (请求暂停与恢复)</span>. </div>

### 1) KV Load Failure Recovery Test

<span style="color:#2980B9">Purpose:</span> Verify that the engine recovers gracefully when a <span style="color:#E8600A;font-weight:700">KV cache load operation fails</span> — e.g., due to corrupted cache files or storage errors.

<span style="color:#2980B9">Core Mechanism:</span> Simulates cache load failures and verifies that the engine falls back to recomputing the KV cache from the original prompt tokens rather than crashing.

------

### 2) LLM Engine Reset KV

<span style="color:#2980B9">Purpose:</span> <span style="color:#E8600A;font-weight:700">Flush and reset the KV cache</span> of a running engine — useful for multi-tenant scenarios where cache isolation between users is required.

<span style="color:#2980B9">Core Mechanism:</span> Calls <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">engine.reset_kv_cache()</code> to invalidate all cached blocks and return physical memory to the free pool, without restarting the engine process.

```python
# After serving one tenant's requests:
engine.reset_kv_cache()  # clear all KV blocks (清空所有 KV 块)
# Now safe to serve a new tenant with a clean cache
```

------

### 3) Load Sharded State

<span style="color:#2980B9">Purpose:</span> Load model weights that have been saved as <span style="color:#E8600A;font-weight:700">sharded checkpoints (分片检查点)</span> — enabling fast startup for very large models distributed across multiple files.

<span style="color:#2980B9">Core Mechanism:</span> Each tensor-parallel worker loads its corresponding weight shard directly from disk, avoiding the need to load the full model on a single process first.

```python
llm = LLM(
    model="...",
    load_format="sharded_state",    # load from pre-sharded checkpoint
    tensor_parallel_size=8          # must match shard count
)
```

------

### 4) Save Sharded State

<span style="color:#2980B9">Purpose:</span> Save the current model state as <span style="color:#E8600A;font-weight:700">sharded checkpoints</span> that can be reloaded with `Load Sharded State` — enabling faster subsequent startups.

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3-70B", tensor_parallel_size=8)

# Save each worker's weight shard to its own file
llm.save_sharded_state(path="/checkpoints/llama3-70b-sharded/")
```

------

### 5) Pause Resume

<span style="color:#2980B9">Purpose:</span> <span style="color:#E8600A;font-weight:700">Pause active requests</span> mid-generation and resume them later — enabling preemption (抢占) of lower-priority requests to serve high-priority ones.

<span style="color:#2980B9">Core Mechanism:</span> The scheduler can evict a request's KV blocks (写回磁盘或丢弃), freeing GPU memory. When resumed, the prefix is recomputed or restored from a saved state.

------

### 6) Reproducibility

<span style="color:#2980B9">Purpose:</span> Guarantee that the <span style="color:#E8600A;font-weight:700">same prompt always produces the same output</span> — critical for regression testing, benchmarking, and debugging.

<span style="color:#2980B9">Core Mechanism:</span> Sets a fixed <span style="color:#E8600A;font-weight:700">Random Seed (随机种子)</span> for the sampling process and ensures deterministic CUDA kernels are used.

```python
from vllm import SamplingParams

# seed=42 guarantees identical outputs across runs
params = SamplingParams(temperature=0.8, top_p=0.95, seed=42)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">True reproducibility across different GPU counts or CUDA versions is not guaranteed</span>, as floating-point reduction orders may differ. Reproducibility is best-effort within the same hardware and software environment.</div>

------

## 7. Training / RLHF Integration (训练与 RLHF 集成)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples demonstrate how <span style="color:#E8600A;font-weight:700">Reinforcement Learning from Human Feedback (RLHF, 基于人类反馈的强化学习)</span> training loops integrate with vLLM as the inference backend. Key challenges include synchronizing policy weights (策略权重同步), sharing GPU resources between training and inference, and handling online quantization (在线量化). </div>

### 1) RLHF

<span style="color:#2980B9">Purpose:</span> Demonstrate the basic <span style="color:#E8600A;font-weight:700">RLHF rollout loop</span> — use vLLM to generate responses, score them with a reward model (奖励模型), and feed the rewards back to the trainer.

<span style="color:#2980B9">Core Mechanism:</span> vLLM acts as a fast <span style="color:#E8600A;font-weight:700">Rollout Engine (展开引擎)</span>. The trainer (e.g., TRL, OpenRLHF) sends prompts, collects token-level log probabilities and completions, then computes PPO/GRPO gradients.

```
Training Loop (训练循环):

  1. Trainer sends batch of prompts to vLLM
  2. vLLM generates completions + log_probs
  3. Reward model scores each completion
  4. Trainer computes policy gradient (e.g., PPO loss)
  5. Trainer updates weights → syncs to vLLM (see New Weight Syncing)
  6. Repeat
```

------

### 2) RLHF Colocate

<span style="color:#2980B9">Purpose:</span> Run training and inference on the <span style="color:#E8600A;font-weight:700">same GPU hardware simultaneously</span> — reducing infrastructure cost by time-multiplexing compute.

<span style="color:#2980B9">Core Mechanism:</span> The training process and vLLM engine alternate GPU access using <span style="color:#E8600A;font-weight:700">CUDA stream switching (CUDA 流切换)</span>. During rollout generation, the optimizer is idle; during gradient updates, vLLM is paused.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Colocated RLHF requires careful memory budgeting — both the optimizer states (优化器状态, typically 12× model size for Adam) and vLLM's KV cache must fit in the same GPU memory simultaneously.</div>

------

### 3) RLHF Online Quant

<span style="color:#2980B9">Purpose:</span> Apply <span style="color:#E8600A;font-weight:700">quantization on-the-fly during RLHF training</span> — reducing memory requirements for the inference copy of the policy model.

<span style="color:#2980B9">Core Mechanism:</span> The trainer maintains FP16/BF16 weights for gradient computation; before syncing to vLLM, weights are quantized to INT4/INT8. This halves the KV cache memory needed for rollout generation.

------

### 4) RLHF Utils

<span style="color:#2980B9">Purpose:</span> Shared utility functions for RLHF pipelines — including weight reshaping (权重重塑), log probability extraction (log 概率提取), and reward normalization (奖励归一化).

------

## 8. Debugging / Observability (调试与可观测性)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> These examples provide tools to <span style="color:#E8600A;font-weight:700">inspect internal model behavior and system performance</span>. They rely on <span style="color:#E8600A;font-weight:700">Profiling Hooks (性能分析钩子)</span>, <span style="color:#E8600A;font-weight:700">Hidden State Extraction (隐藏状态提取)</span>, and <span style="color:#E8600A;font-weight:700">Runtime Metrics (运行时指标)</span>. </div>

### 1) Metrics

<span style="color:#2980B9">Purpose:</span> Expose <span style="color:#E8600A;font-weight:700">Prometheus-compatible metrics (Prometheus 兼容指标)</span> for monitoring vLLM in production — covering latency, throughput, cache hit rate, and queue depth.

<span style="color:#2980B9">Core Mechanism:</span> vLLM's OpenAI-compatible server exposes a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">/metrics</code> endpoint. Key metrics include:

| Metric                                                       | Description                             |
| ------------------------------------------------------------ | --------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vllm:e2e_request_latency_seconds</code> | End-to-end request latency (端到端延迟) |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vllm:num_requests_running</code> | Requests currently being processed      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vllm:gpu_cache_usage_perc</code> | KV cache utilization percentage         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vllm:tokens_per_second</code> | Generation throughput (生成吞吐量)      |

```bash
# Start server with metrics enabled (default)
vllm serve meta-llama/Llama-3-8B --host 0.0.0.0 --port 8000

# Scrape metrics
curl http://localhost:8000/metrics
```

------

### 2) Simple Profiling

<span style="color:#2980B9">Purpose:</span> Profile vLLM's execution using <span style="color:#E8600A;font-weight:700">CUDA profiling tools (CUDA 性能分析工具)</span> to identify GPU kernel bottlenecks (内核瓶颈).

<span style="color:#2980B9">Core Mechanism:</span> Wraps the engine's step loop with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">torch.profiler</code> or NVTX markers, generating a timeline that can be visualized in NVIDIA Nsight Systems.

```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True
) as prof:
    outputs = llm.generate(prompts, sampling_params)

# Export to Chrome trace format (Chrome 追踪格式)
prof.export_chrome_trace("vllm_trace.json")
```

------

### 3) Extract Hidden States

<span style="color:#2980B9">Purpose:</span> Capture <span style="color:#E8600A;font-weight:700">intermediate hidden state tensors (中间隐藏状态张量)</span> from specific transformer layers — useful for probing, feature extraction, and representation analysis.

<span style="color:#2980B9">Core Mechanism:</span> Registers <span style="color:#E8600A;font-weight:700">forward hooks (前向传播钩子)</span> on target transformer layers. The hook captures the output activation tensor and stores it alongside the final generated text.

```python
hidden_states = {}

def capture_hook(module, input, output):
    # output shape: [batch, seq_len, hidden_dim]
    hidden_states["layer_16"] = output[0].detach().cpu()

# Register hook on layer 16's self-attention output
model.model.layers[16].register_forward_hook(capture_hook)

outputs = llm.generate("What is attention?", SamplingParams(max_tokens=10))
print(hidden_states["layer_16"].shape)  # → [1, seq_len, 4096]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Hidden state extraction adds memory overhead proportional to <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">batch_size × seq_len × hidden_dim × num_hooked_layers</code>. <span style="color:#C0392B;font-weight:600">Do not enable in production</span> without careful memory budgeting.</div>

------

### 4) Skip Loading Weights In Engine Init

<span style="color:#2980B9">Purpose:</span> Initialize the vLLM engine <span style="color:#E8600A;font-weight:700">without loading model weights</span> — useful for testing engine infrastructure (调度器, memory allocator) in isolation without GPU memory.

<span style="color:#2980B9">Core Mechanism:</span> Sets <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">load_format="dummy"</code>, which initializes all weight tensors to zero or random values without reading from disk.

```python
llm = LLM(
    model="meta-llama/Llama-3-70B",
    load_format="dummy",      # skip weight loading (跳过权重加载)
    enforce_eager=True        # disable CUDA graph for debugging
)
# Engine is fully initialized — scheduler, KV allocator, etc. are operational
# Model outputs will be meaningless, but infrastructure can be tested
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This is invaluable for <span style="color:#E8600A;font-weight:700">CI/CD pipelines (持续集成/持续部署)</span> that test scheduling logic, API routing, or batching behavior on CPU-only machines without requiring an actual model checkpoint.</div>

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> vLLM's 44 examples span a clear hierarchy: <span style="color:#E8600A;font-weight:700">Inference Modes</span> define <em>how</em> you call the model, <span style="color:#E8600A;font-weight:700">Architecture examples</span> define <em>where</em> it runs, <span style="color:#E8600A;font-weight:700">Optimization examples</span> define <em>how fast</em> it runs, <span style="color:#E8600A;font-weight:700">Modality examples</span> define <em>what</em> it can process, <span style="color:#E8600A;font-weight:700">Adaptation examples</span> define <em>what behavior</em> it exhibits, and <span style="color:#E8600A;font-weight:700">Reliability / RLHF / Observability examples</span> ensure it runs <em>correctly, safely, and measurably</em> in production. </div>
