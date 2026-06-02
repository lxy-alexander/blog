---
title: "vllm questions"
published: 2026-06-01
description: "vllm questions"
image: ""
tags: ["llm","vllm questions"]
category: llm
draft: false
lang: ""
createdAt: "2026-06-01T17:22:57.424.420304947Z"
---

### **1）What is vLLM?**

vLLM is a high-performance inference and serving framework for large language models. It is used to run LLMs efficiently in production or offline environments. Its main purpose is to increase throughput, reduce memory waste, and make GPU usage more efficient.

vLLM 是一个用于大语言模型的高性能推理和服务框架。它用于在生产环境或离线环境中高效运行大语言模型。它的主要目的是提高吞吐量、减少内存浪费，并让 GPU 使用更加高效。

<br>

------

## 2）What problem does vLLM solve?

vLLM solves the problem of inefficient LLM serving. In normal LLM inference, KV cache memory can be wasted, GPU utilization may be low, and requests may be handled inefficiently. vLLM improves these areas through better batching, scheduling, and memory management.

vLLM 解决的是大语言模型服务效率低的问题。在普通的大语言模型推理中，KV 缓存内存可能被浪费，GPU 利用率可能较低，请求处理也可能不够高效。vLLM 通过更好的批处理、调度和内存管理来改善这些问题。

<br>

------

### 3）Why is vLLM widely used for LLM serving?

vLLM is widely used because it can serve many requests efficiently while keeping latency acceptable. It supports important production features such as continuous batching, PagedAttention, OpenAI-compatible APIs, quantization, LoRA, and distributed inference.

vLLM 被广泛使用，因为它可以高效服务大量请求，同时保持可接受的延迟。它支持许多重要的生产特性，例如 continuous batching、PagedAttention、OpenAI-compatible APIs、量化、LoRA 和分布式推理。

<br>

------

### 4）What is PagedAttention?

PagedAttention is a KV cache memory management technique used by vLLM. It divides the KV cache into smaller blocks instead of storing each sequence in one large continuous memory area. This makes memory allocation more flexible and reduces memory fragmentation.

PagedAttention 是 vLLM 使用的一种 KV 缓存内存管理技术。它将 KV 缓存拆分成较小的块，而不是把每个序列存储在一大块连续内存中。这样可以让内存分配更加灵活，并减少内存碎片。

<br>

------

### 5）Why is PagedAttention important?

PagedAttention is important because KV cache can consume a large amount of GPU memory during generation. Without efficient cache management, memory may be wasted and the system may support fewer concurrent requests. PagedAttention helps vLLM serve more requests with the same GPU memory.

PagedAttention 很重要，因为 KV 缓存在生成过程中会消耗大量 GPU 内存。如果没有高效的缓存管理，内存可能被浪费，系统能支持的并发请求也会更少。PagedAttention 帮助 vLLM 用相同的 GPU 内存服务更多请求。

<br>

------

### 6）What is KV cache?

KV cache stores the key and value tensors generated from previous tokens in the attention mechanism. During decoding, the model can reuse these tensors instead of recomputing attention for all previous tokens. This makes token generation much faster.

KV 缓存存储注意力机制中之前 token 生成的 key 和 value 张量。在解码过程中，模型可以复用这些张量，而不用重新计算所有之前 token 的注意力。这样可以让 token 生成速度更快。

<br>

------

### 7）Why does KV cache use so much memory?

KV cache grows with the number of requests, the sequence length, the number of layers, and the hidden size of the model. Longer prompts and longer outputs require more KV cache. This is why memory management is a key challenge in LLM serving.

KV 缓存会随着请求数量、序列长度、模型层数和模型隐藏维度的增加而增长。更长的提示词和更长的输出需要更多 KV 缓存。这就是为什么内存管理是大语言模型服务中的关键挑战。

<br>

------

### 8）What is continuous batching?

Continuous batching an optimization technique for LLM inference that ==vLLM(inference system)== can dynamically add new requests and remove finished requests ==while generation is running.== At each generation step, after produce a token,  it will check if any requests have finished, if new requests are waiting, and if there are enough resources, such as GPU memory, KV cache space, and batch capacity, to add new requests to the current batch.

连续批处理意味着vLLM可以在生成过程中动态地添加新请求并删除已完成的请求。在每个生成步骤中，vLLM检查是否有任何请求已经完成，是否有新的请求正在等待，以及是否有足够的资源（如GPU内存、KV缓存空间和批处理容量）将新请求添加到当前批处理中。

<br>

------

### 9）Why is continuous batching better than static batching?

Static batching is when ==first collects a group of requests, forms them into a fixed batch, and processes them together.== Before this batch finishes, new requests can not be added to it. This can waste GPU resources. Continuous batching improves GPU utilization because continuous batching can remove finished requests and add new requests during generation.

静态批处理是指系统首先收集一组请求，将它们形成固定的批处理，然后一起处理它们。在此批处理完成之前，通常不会动态地向其中添加新请求。这可能会浪费GPU资源。连续批处理提高了GPU利用率，因为连续批处理可以在生成过程中删除已完成的请求并添加新请求。

<br>

------

### 10）What are prefill and decode in LLM inference?

Prefill is the stage where the model processes the input prompt and creates the initial KV cache. Decode is the stage where the model generates new tokens one by one. Prefill is usually compute-heavy, while decode is often memory-heavy.

Prefill 是模型处理输入提示词并创建初始 KV 缓存的阶段。Decode 是模型逐个生成新 token 的阶段。Prefill 通常计算量较大，而 decode 通常更依赖内存。

<br>

------

11）What is chunked prefill?

Chunked prefill splits a long prompt into smaller chunks during the prefill stage. Instead of processing a very long prompt all at once, vLLM can process it in parts. This allows other decoding requests to be scheduled between chunks.

Chunked prefill 在 prefill 阶段将长提示词拆分成较小的块。vLLM 不是一次性处理一个很长的提示词，而是分段处理。这样可以让其他解码请求在这些块之间被调度。

------

12）Why is chunked prefill useful?

Chunked prefill is useful because long prompts can block other requests if they are processed all at once. By splitting long prefill work into chunks, vLLM can reduce waiting time for other users and improve overall scheduling fairness.

Chunked prefill 很有用，因为如果一次性处理长提示词，它可能阻塞其他请求。通过将长 prefill 工作拆成多个块，vLLM 可以减少其他用户的等待时间，并提高整体调度公平性。

------

13）What is prefix caching?

Prefix caching means reusing the KV cache of a prompt prefix that has already been computed. If many requests share the same beginning, vLLM does not need to recompute that shared part every time.

Prefix caching 表示复用已经计算过的提示词前缀的 KV 缓存。如果许多请求共享相同的开头，vLLM 就不需要每次都重新计算这部分共享内容。

------

14）When is prefix caching most useful?

Prefix caching is most useful when many requests have the same system prompt, instruction, or document prefix. For example, in a chatbot, every request may start with the same system message, so caching that prefix can save computation.

当许多请求有相同的系统提示词、指令或文档前缀时，prefix caching 最有用。例如，在聊天机器人中，每个请求可能都以相同的系统消息开头，因此缓存这个前缀可以节省计算。

------

15）What is the OpenAI-compatible API server in vLLM?

The OpenAI-compatible API server allows vLLM to provide endpoints similar to OpenAI’s API format. This means applications that already use OpenAI-style requests can often connect to vLLM with fewer code changes.

OpenAI-compatible API server 允许 vLLM 提供类似 OpenAI API 格式的接口。这意味着已经使用 OpenAI 风格请求的应用，通常可以用较少代码修改连接到 vLLM。

------

16）How do you usually start a vLLM server?

A common way to start a vLLM server is to use the `vllm serve` command with a model name or model path. After the server starts, clients can send chat completion or completion requests to the server.

启动 vLLM 服务器的一种常见方式是使用 `vllm serve` 命令，并指定模型名称或模型路径。服务器启动后，客户端可以向服务器发送聊天补全或文本补全请求。

------

17）What is offline inference in vLLM?

Offline inference means using vLLM directly in Python without starting an HTTP server. It is useful for local testing, batch generation, model evaluation, and experiments where an API service is not required.

Offline inference 表示直接在 Python 中使用 vLLM，而不启动 HTTP 服务器。它适用于本地测试、批量生成、模型评估，以及不需要 API 服务的实验场景。

------

18）What is SamplingParams used for?

SamplingParams is used to control the generation behavior of the model. It can configure parameters such as temperature, top-p, maximum output tokens, stop sequences, and the number of generated outputs.

SamplingParams 用于控制模型的生成行为。它可以配置 temperature、top-p、最大输出 token 数、停止序列和生成结果数量等参数。

------

19）What does temperature mean in text generation?

Temperature controls how random the model output is. A lower temperature makes the model choose safer and more likely tokens, so the output is more stable. A higher temperature allows more randomness, so the output is more diverse.

Temperature 控制模型输出的随机程度。较低的 temperature 会让模型选择更安全、更可能的 token，因此输出更稳定。较高的 temperature 允许更多随机性，因此输出更加多样。

------

20）What does top-p sampling mean?

Top-p sampling chooses tokens from the smallest group of tokens whose cumulative probability reaches p. It removes very unlikely tokens from consideration while still allowing some diversity in the output.

Top-p 采样从累计概率达到 p 的最小 token 组中选择 token。它会排除概率很低的 token，同时仍然允许输出具有一定多样性。

------

21）What does max_tokens mean?

max_tokens is the maximum number of tokens that the model is allowed to generate. It is important because longer outputs increase generation time, KV cache usage, and GPU memory usage.

max_tokens 是模型被允许生成的最大 token 数量。它很重要，因为更长的输出会增加生成时间、KV 缓存使用量和 GPU 内存使用量。

------

22）What is tensor parallelism in vLLM?

Tensor parallelism splits the model’s tensors across multiple GPUs. It is useful when one GPU cannot hold the whole model or when multiple GPUs are needed to speed up inference.

Tensor parallelism 将模型的张量拆分到多个 GPU 上。当单个 GPU 无法容纳整个模型，或需要多个 GPU 加速推理时，它很有用。

------

23）What is pipeline parallelism?

Pipeline parallelism splits different layers of the model across different GPUs. Each GPU handles a different part of the model. This is useful when the model is too large to fit on one GPU or one GPU group.

Pipeline parallelism 将模型的不同层拆分到不同 GPU 上。每个 GPU 处理模型的不同部分。当模型太大，无法放在单个 GPU 或单个 GPU 组上时，它很有用。

------

24）What is quantization in vLLM?

Quantization reduces the precision of model weights or activations, such as using 8-bit or 4-bit formats. It reduces memory usage and may improve speed, but it can sometimes slightly reduce model quality.

量化降低模型权重或激活值的精度，例如使用 8 位或 4 位格式。它减少内存使用，并可能提升速度，但有时也可能轻微降低模型质量。

------

25）Why are AWQ, GPTQ, and FP8 useful?

AWQ, GPTQ, and FP8 are quantization-related methods or formats supported in many LLM serving scenarios. They help reduce GPU memory usage, which makes it possible to run larger models or serve more requests on the same hardware.

AWQ、GPTQ 和 FP8 是许多大语言模型服务场景中支持的量化相关方法或格式。它们帮助减少 GPU 内存使用，从而可以在相同硬件上运行更大的模型或服务更多请求。

------

26）What is LoRA serving in vLLM?

LoRA serving means loading lightweight fine-tuned adapters on top of a base model. Instead of deploying many full fine-tuned models, vLLM can use one base model with multiple LoRA adapters for different tasks or styles.

LoRA serving 表示在基础模型之上加载轻量级微调适配器。vLLM 不需要部署许多完整的微调模型，而是可以用一个基础模型搭配多个 LoRA 适配器来处理不同任务或风格。

------

27）What is speculative decoding?

Speculative decoding uses a smaller or faster draft model to propose several tokens first. Then the larger target model checks whether those tokens are acceptable. If many proposed tokens are accepted, generation can become faster.

Speculative decoding 使用一个更小或更快的草稿模型先提出多个 token。然后更大的目标模型检查这些 token 是否可以接受。如果许多提出的 token 被接受，生成速度就可以变快。

------

28）What are throughput, latency, TTFT, and inter-token latency?

Throughput means how many tokens or requests the system can process per second. Latency is the total time a request takes. TTFT is the time to first token. Inter-token latency is the time gap between generated tokens.

Throughput 表示系统每秒可以处理多少 token 或请求。Latency 是一个请求花费的总时间。TTFT 是首个 token 的时间。Inter-token latency 是生成 token 之间的时间间隔。

------

29）How do you troubleshoot out-of-memory errors in vLLM?

To troubleshoot OOM errors, first check whether the model is too large for the GPU memory. Then check max model length, batch size, concurrency, KV cache usage, and GPU memory utilization. Common solutions include reducing sequence length, reducing concurrency, using quantization, or adding more GPUs.

要排查 OOM 错误，首先检查模型是否对 GPU 内存来说太大。然后检查最大模型长度、批大小、并发数、KV 缓存使用量和 GPU 内存利用率。常见解决方案包括减少序列长度、降低并发、使用量化，或增加更多 GPU。

------

30）How do you improve vLLM throughput and latency?

To improve throughput, increase batching efficiency, tune concurrency, use proper parallelism, enable useful caching features, and use quantization when appropriate. To reduce latency, use streaming, reduce prompt length, limit output length, and avoid unnecessary long requests.

要提高吞吐量，可以提高批处理效率、调整并发、使用合适的并行方式、启用有用的缓存特性，并在合适时使用量化。要降低延迟，可以使用流式输出、减少提示词长度、限制输出长度，并避免不必要的长请求。
