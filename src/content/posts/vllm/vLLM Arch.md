---
title: "vLLM Arch"
published: 2026-02-19
description: "vLLM Arch"
image: ""
tags: ["vllm","vLLM Arch"]
category: vllm
draft: false
lang: ""
---



# Entrypoints

vLLM provides a number of entrypoints for interacting with the system. The following diagram shows the relationship between them.



![image-20260219195018919](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260219195018919)

## LLM Class([vllm/entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py).)

```python
from vllm import LLM, SamplingParams

# Define a list of input prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The largest ocean is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the LLM engine with the OPT-125M model
llm = LLM(model="facebook/opt-125m")

# Generate outputs for the input prompts
outputs = llm.generate(prompts, sampling_params)

# Print the generated outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```



## OpenAI-Compatible API Server

The second primary interface to vLLM is via its OpenAI-compatible API server. This server can be started using the `vllm serve` command.

```
vllm serve <model>
```









## AsyncLLMEngine (vllm/engine/async_llm_engine.py)

这是一个**向后兼容的别名**，不是真正的包装。

`AsyncLLMEngine = AsyncLLM` 只是一行赋值，让旧代码无需修改就能继续运行：

异步推理引擎

"""The `AsyncLLMEngine` class is an alias of [vllm.v1.engine.async_llm.AsyncLLM][]."""

实现类：vllm/v1/engine/async_llm.py 中的 AsyncLLM

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.async_llm import AsyncLLM

AsyncLLMEngine = AsyncLLM  # type: ignore
"""The `AsyncLLMEngine` class is an alias of [vllm.v1.engine.async_llm.AsyncLLM][]."""
```





## LLMEngine(vllm/v1/engine/llm_engine.py)

同步推理引擎

```
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats
        
        .....
```



