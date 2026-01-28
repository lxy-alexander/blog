---
title: "Params on vLLM"
published: 2026-01-27
description: "Params on vLLM"
image: ""
tags: ["vllm","Params on vLLM"]
category: vllm
draft: false
lang: ""
---

# `EngineArgs` 和 `SamplingParams` 的区别

EngineArgs configures the runtime engine, including model loading, GPU parallelism, and memory management.
SamplingParams configures decoding behavior, including randomness, length control, stopping criteria, and constraints.↳

 vLLM 当成一个“写作工厂”：

-   **EngineArgs**：选厂房、选机器、接多少电、配多少工人
     → 决定产能和能不能生产
-   **SamplingParams**：决定写作风格、写几篇、写多长、什么时候停
     → 决定具体产出的内容

EngineArgs：初始化用一次

```
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.9
)
```

SamplingParams：每次生成可变

```
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
out = llm.generate(["Hello my name is"], params)
```

## **EngineArgs = Engine / Runtime Configuration**

它主要管这些事情：

-   **加载什么模型**：`model`, `tokenizer`, `revision`
-   **怎么用 GPU 跑**：`tensor_parallel_size`, `dtype`, `quantization`
-   **显存怎么分配**：`gpu_memory_utilization`, `swap_space`, `cpu_offload_gb`
-   **性能/兼容开关**：`enforce_eager`, `compilation_config`

📌 特点：

-   一般在 **初始化 LLM/Engine 的时候设置一次**
-   影响 **性能、并行、显存、能不能加载成功**



## **SamplingParams = Decoding / Generation Configuration**

它管的是“输出文本怎么生成”：

-   **随机程度**：`temperature`, `top_p`, `top_k`
-   **输出长度**：`max_tokens`, `min_tokens`
-   **停止规则**：`stop`, `stop_token_ids`, `ignore_eos`
-   **生成几条**：`n`, `best_of`
-   **防重复**：`repetition_penalty`, `presence_penalty`↳
-   **高级约束**：`logit_bias`, `guided_decoding`, `allowed_token_ids`

📌 特点：

-   通常在 **每次 generate() 的时候传入**
-   影响 **内容风格、长度、稳定性、格式约束**







# SamplingParams

## ① n

-   **Meaning**: It specifies how many candidate outputs to return in one generation.
-   **Default**: `1`
-   **Example**: `n=3` → returns 3 different continuations for the same prompt.

------

## ② best_of

-   **Meaning**: Generate `best_of` candidates first, then choose the best ones to return (like “generate more, pick the best”).
-   **Default**: `None`
-   **Requirement**: `best_of >= n`
-   **Use case**: Better quality, but slower and more expensive.

------

## ③ _real_n (internal)

-   **Meaning**: When `best_of` is set, this stores the original `n` requested by the user.
-   **Default**: `None`
-   **Note**: You typically don’t need to set this manually.

```python
# 用户创建 SamplingParams
params = SamplingParams(n=3, best_of=10)

# 初始化后的状态：
# params.n = 10          # 被改为 best_of 的值（用于生成 10 个候选）
# params._real_n = 3      # 保存原始的 n 值（用于返回最好的 3 个）
# params.best_of = 10
```

------

## ④ presence_penalty

-   ==**Meaning**: Penalizes tokens that already appeared → encourages introducing new words/topics.==
-   **Default**: `0.0`
-   **Range**: `[-2, 2]`
-   **Effect**: Higher values make outputs more diverse and less repetitive.

------

## ⑤ frequency_penalty

-   **Meaning**: Penalizes tokens more heavily the more often they appear → reduces repeated phrases/loops.
-   **Default**: `0.0`
-   **Range**: `[-2, 2]`
-   **Effect**: Higher values reduce repetitive looping even more strongly.

------

## ⑥ repetition_penalty

-   **Meaning**: A repetition penalty (common in Hugging Face style decoding) to discourage repeated tokens.
-   **Default**: `1.0`
-   **Range**: `(0, 2]`
-   **Typical values**: `1.05 ~ 1.2`; too high may make output weird.

------

## ⑦ temperature

**temperature：决定“随机程度”**

**top_k：决定“最多允许多少个候选词”**

**top_p：决定“只允许最靠谱的一批候选词”**

**min_p：决定“太小概率的一律不准选”**

**seed：决定“随机结果能不能复现”**

==**As `temperature` increases, the output becomes more random.**==

-   **Meaning**: Sampling temperature that ==controls randomness.==
-   **Default**: `1.0`
-   **Range**: `>= 0`
-   **Effect**:
    -   `0` → greedy smapling (always choose the highest-probability token)
    -   `0.2` → very stable
    -   `0.8` → commonly used
    -   `1.0` → original distribution → logits unchanged
    -   `1.2+` → more random / creative but less reliable

------

## ⑧ top_p

-   **Meaning**: Nucleus sampling; sample only from tokens whose cumulative probability reaches `top_p`.
-   **Default**: `1.0`
-   **Range**: `(0, 1]`
-   **Effect**:
    -   `0.9 ~ 0.95` → common settings
    -   Smaller = safer/more conservative, larger = more diverse

------

## ⑨ top_k

-   **Meaning**: Sample only from the top K most likely tokens.
-   **Default**: `-1` (disabled)
-   **Valid values**: `-1` or `>=1` (cannot be 0)
-   **Effect**:
    -   `top_k=50` is common
    -   Often used together with `top_p` for stability

------

## ⑩ min_p

-   **Meaning**: Filters out tokens with too small a probability (below a threshold).
-   **Default**: `0.0`
-   **Range**: `[0, 1]`
-   **Effect**: Higher values produce “cleaner” outputs but reduce diversity.

------

## ⑪ seed

-   **Meaning**: Random seed (makes the output reproducible).
-   **Default**: `None`
-   **Special case**: If you pass `-1`, it becomes `None`.
-   **Use case**: Debugging, comparing parameter changes, reproducibility.

------

## ⑫ stop

-   **Meaning**: Stop strings; generation stops when any stop string appears.
-   **Default**: `None` (will become `[]`)
-   **Supported**: a string or a list of strings
-   **Note**: Cannot include an empty string `""`.

------

## ⑬ stop_token_ids

-   **Meaning**: Stop token IDs; generation stops when any of these tokens appears.
-   **Default**: `None` (will become `[]`)
-   **Use case**: Lower-level and more precise stopping than stop strings.

------

## ⑭ ignore_eos

-   **Meaning**: Whether to ignore the model EOS (end-of-sequence) token.
-   **Default**: `False`
-   **Effect**:
    -   `False` → EOS stops generation
    -   `True` → EOS does not stop (more likely to run until max_tokens)

------

## ⑮ max_tokens

-   **Meaning**: Maximum number of tokens to generate (output length limit).
-   **Default**: `16`
-   **Requirement**: If not `None`, must be `>=1`
-   **Suggestion**: Usually set it higher, e.g. `max_tokens=128`.

------

## ⑯ min_tokens

-   **Meaning**: Minimum number of tokens that must be generated (cannot stop before this).
-   **Default**: `0`
-   **Requirement**: `>=0` and `<= max_tokens`
-   **Use case**: Prevents the model from stopping too early.

------

## ⑰ logprobs

-   **Meaning**: Returns top-k log probabilities for each generated token.
-   **Default**: `None`
-   **Special case**: If set to `True`, it becomes `1`.
-   **Use case**: Scoring, debugging, analyzing confidence.

------

## ⑱ prompt_logprobs

-   **Meaning**: Returns log probabilities for prompt tokens.
-   **Default**: `None`
-   **Special case**: If set to `True`, it becomes `1`.
-   **Use case**: Analyze how the model “understands” the input prompt.

------

## ⑲ detokenize

-   **Meaning**: Whether to convert tokens back into text output.
-   **Default**: `True`
-   **Note**: If `detokenize=False`, stop strings (`stop`) cannot be used.

------

## ⑳ skip_special_tokens

-   **Meaning**: Skip special tokens during decoding (like `<s>`, `</s>`).
-   **Default**: `True`
-   **Effect**: Cleaner final output.

------

## ㉑ spaces_between_special_tokens

-   **Meaning**: Whether to insert spaces between special tokens.
-   **Default**: `True`
-   **Usually**: No need to change.

------

## ㉒ logits_processors

-   **Meaning**: Custom logits processor list (can directly modify token probabilities before sampling).
-   **Default**: `None`
-   **Use case**: Advanced controls like forcing formats, banning tokens, custom decoding logic.

------

## ㉓ include_stop_str_in_output

-   **Meaning**: Whether to keep the stop string itself in the output.
-   **Default**: `False`
-   **Example**: stop="###"
    -   False → stops before "###", output does not include it
    -   True → output ends with "###"

------

## ㉔ truncate_prompt_tokens

-   **Meaning**: If prompt is too long, keep only the last N tokens.
-   **Default**: `None`
-   **Requirement**: Must be `>=1` if set.
-   **Use case**: Control context window and avoid OOM.

------

## ㉕ output_kind

-   **Meaning**: Output format style.
-   **Default**: `CUMULATIVE`
-   **Common values**:
    -   `CUMULATIVE` → returns the full cumulative text each time
    -   `DELTA` → returns only the newly generated chunk (streaming-friendly)
-   **Constraint**: With `DELTA`, `best_of` must equal `n`.

------

## Internal Fields (not required as user inputs)

## ㉖ output_text_buffer_length (internal)

-   **Meaning**: Buffers a few characters at the end to correctly detect stop strings.
-   **When used**: Automatically computed when stop strings are enabled and stop strings should not be included in output.

------

## ㉗ _all_stop_token_ids (internal)

-   **Meaning**: A combined set of all stop token IDs, including stop_token_ids and eos_token_id.
-   **Use case**: Used internally to decide when generation should stop.

------

## ㉘ guided_decoding

-   **Meaning**: Guided decoding parameters (e.g., force output to follow JSON/grammar constraints).
-   **Default**: `None`
-   **Use case**: Very useful when you need strict structured output.

------

## ㉙ logit_bias

-   **Meaning**: Adds bias to specific token IDs (make them more or less likely).
-   **Default**: `None`
-   **Range**: Clamped to `[-100, 100]`
-   **Example**: `{token_id: +5}` makes a token more likely; `-100` almost bans it.

------

## ㉚ allowed_token_ids

-   **Meaning**: A whitelist of tokens that the model is allowed to generate.
-   **Default**: `None`
-   **Use case**: Very strict constraint (e.g., only allow digits).

------

## ㉛ extra_args

-   **Meaning**: Extra arguments dictionary (reserved for extensions).
-   **Default**: `None`

------

## ㉜ bad_words

-   **Meaning**: A list of banned words that the model is not allowed to generate.
-   **Default**: `None` (will become `[]`)
-   **Use case**: Prevent certain words from appearing (closer to filtering than stopping).

------

## ㉝ _bad_words_token_ids (internal)

-   **Meaning**: Token-ID sequences converted from `bad_words`.
-   **Default**: `None`
-   **Filled by**: `update_from_tokenizer()` automatically.



---

✅ **Chat / General**

```python
SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
```

✅ **More Stable (Less Random)**

```python
SamplingParams(temperature=0.2, top_p=0.9, max_tokens=256)
```

✅ **Reduce Repetition**

```python
SamplingParams(temperature=0.7, top_p=0.9, repetition_penalty=1.1, max_tokens=256)
```

------



# EngineArgs



## ① `model`

-   **What it does**: Specifies which model checkpoint to load (HF name or local path).

```python
EngineArgs(model="facebook/opt-125m")
```

-   Local path:

```python
EngineArgs(model="/data/models/llama-7b")
```

------

## ② `task`

-   **What it does**: Specifies the task type (`auto` is common).

```python
EngineArgs(model="facebook/opt-125m", task="auto")
```

------

## ③ `tokenizer`

-   **What it does**: Specifies which tokenizer to use.

```python
EngineArgs(
    model="facebook/opt-125m",
    tokenizer="facebook/opt-125m"
)
```

------

## ④ `tokenizer_mode`

-   **What it does**: Controls whether vLLM uses fast or slow tokenizer.

```python
EngineArgs(model="facebook/opt-125m", tokenizer_mode="auto")
```

-   Force slow tokenizer:

```python
EngineArgs(model="facebook/opt-125m", tokenizer_mode="slow")
```

------

## ⑤ `skip_tokenizer_init`

-   **What it does**: Skips tokenizer initialization (useful when providing token IDs directly).

```python
EngineArgs(model="facebook/opt-125m", skip_tokenizer_init=True)
```

------

## ⑥ `trust_remote_code`

-   **What it does**: Allows loading models that require custom HF code.

```python
EngineArgs(model="some-org/custom-model", trust_remote_code=True)
```

------

## ⑦ `allowed_local_media_path`

-   **What it does**: Allows the engine to read local images/videos (multimodal use).

```python
EngineArgs(
    model="some-mm-model",
    allowed_local_media_path="/data/media"
)
```

------

## ⑧ `tensor_parallel_size`

-   **What it does**: Number of GPUs used for tensor parallelism.
    Single GPU:

```python
EngineArgs(model="facebook/opt-125m", tensor_parallel_size=1)
```

-   4 GPUs:

```python
EngineArgs(model="meta-llama/Llama-2-13b-hf", tensor_parallel_size=4)
```

------

## ⑨ `dtype`

-   **What it does**: Precision of model weights/activations.
    Auto dtype:

```python
EngineArgs(model="facebook/opt-125m", dtype="auto")
```

-   Force FP16:

```python
EngineArgs(model="facebook/opt-125m", dtype="float16")
```

-   Force BF16:

```python
EngineArgs(model="facebook/opt-125m", dtype="bfloat16")
```

------

## ⑩ `quantization`

-   **What it does**: Enables quantized inference (AWQ/GPTQ/FP8).
    AWQ:

```python
EngineArgs(model="some-awq-model", quantization="awq")
```

-   GPTQ:

```python
EngineArgs(model="some-gptq-model", quantization="gptq")
```

-   FP8 (experimental):

```python
EngineArgs(model="some-fp8-model", quantization="fp8")
```

------

## ⑪ `revision`

-   **What it does**: Pins a specific model version (branch/tag/commit).

```python
EngineArgs(
    model="facebook/opt-125m",
    revision="main"
)
```

-   Commit hash:

```python
EngineArgs(
    model="facebook/opt-125m",
    revision="a1b2c3d4e5f6"
)
```

------

## ⑫ `tokenizer_revision`

-   **What it does**: Pins a specific tokenizer version.

```python
EngineArgs(
    model="facebook/opt-125m",
    tokenizer_revision="main"
)
```

------

## ⑬ `seed`

-   **What it does**: Sets a random seed for reproducible sampling.

```python
EngineArgs(model="facebook/opt-125m", seed=42)
```

------

## ⑭ `gpu_memory_utilization`

-   **What it does**: Fraction of GPU memory reserved for weights + KV cache.
    Conservative:

```python
EngineArgs(model="facebook/opt-125m", gpu_memory_utilization=0.6)
```

-   Aggressive (higher throughput, higher OOM risk):

```python
EngineArgs(model="facebook/opt-125m", gpu_memory_utilization=0.95)
```

------

## ⑮ `swap_space`

-   **What it does**: CPU swap memory per GPU (GiB), useful when `best_of > 1`.
    Disable swap (only if you always use best_of=1):

```python
EngineArgs(model="facebook/opt-125m", swap_space=0)
```

-   Enable swap:

```python
EngineArgs(model="facebook/opt-125m", swap_space=8)
```

------

## ⑯ `cpu_offload_gb`

-   **What it does**: Offloads part of model weights to CPU RAM (GiB).

```python
EngineArgs(model="meta-llama/Llama-2-13b-hf", cpu_offload_gb=10)
```

------

## ⑰ `enforce_eager`

-   **What it does**: Forces eager execution (disables CUDA graph optimizations).

```python
EngineArgs(model="facebook/opt-125m", enforce_eager=True)
```

------

## ⑱ `max_seq_len_to_capture`

-   **What it does**: Max sequence length covered by CUDA graphs.
    Smaller (less graph coverage):

```python
EngineArgs(model="facebook/opt-125m", max_seq_len_to_capture=2048)
```

-   Larger (covers longer contexts):

```python
EngineArgs(model="facebook/opt-125m", max_seq_len_to_capture=8192)
```

------

## ⑲ `disable_custom_all_reduce`

-   **What it does**: Disables vLLM’s custom all-reduce implementation.

```python
EngineArgs(
    model="meta-llama/Llama-2-13b-hf",
    tensor_parallel_size=2,
    disable_custom_all_reduce=True
)
```

------

## ⑳ `disable_async_output_proc`

-   **What it does**: Disables async output processing (may reduce performance).

```python
EngineArgs(model="facebook/opt-125m", disable_async_output_proc=True)
```

------

## ㉑ `hf_overrides`

-   **What it does**: Overrides Hugging Face config values.
    Dict override:

```python
EngineArgs(
    model="facebook/opt-125m",
    hf_overrides={"max_position_embeddings": 4096}
)
```

------

## ㉒ `mm_processor_kwargs`

-   **What it does**: Extra kwargs for multimodal processing.

```python
EngineArgs(
    model="some-mm-model",
    mm_processor_kwargs={"do_resize": True, "size": 512}
)
```

------

## ㉓ `override_pooler_config`

-   **What it does**: Overrides pooler config (mostly for embedding tasks).

```python
EngineArgs(
    model="sentence-transformers/all-MiniLM-L6-v2",
    override_pooler_config={"pooling_type": "mean"}
)
```

------

## ㉔ `compilation_config`

-   **What it does**: Controls compilation optimization level or detailed config.
    Simple level:

```python
EngineArgs(model="facebook/opt-125m", compilation_config=2)
```

-   Full dict config:

```python
EngineArgs(
    model="facebook/opt-125m",
    compilation_config={"level": 2, "enable_cuda_graph": True}
)
```

------

## ㉕ `**kwargs`

-   **What it does**: Extra engine-level settings passed through to vLLM.
    Example (passing a custom worker class):

```python
EngineArgs(
    model="facebook/opt-125m",
    worker_cls="vllm.worker.worker.Worker"
)
```

------





