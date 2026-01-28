---
title: "Offline Batching on vLLM"
published: 2026-01-26
description: "Offline Batching on vLLM"
image: ""
tags: ["vllm","Offline Batching on vLLM"]
category: vllm
draft: false
lang: ""

---

# Offline Batching Example

## 1) 导入

```
from vllm import LLM, SamplingParams
```

-   `LLM`：vLLM 的模型执行器，用来加载模型并做推理/生成。
-   `SamplingParams`：采样参数对象，用来控制生成的随机性、截断策略等。

------

## 2) 准备输入 prompts

```
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
```

-   `prompts` 是一个字符串列表。
-   vLLM 会把它们当成**4 个独立请求**，一次批量生成（batching），效率更高。

------

## 3) 设置采样参数 SamplingParams

`vllm/sampling_params.py`

https://lxy-alexander.github.io/blog/posts/vllm/params-on-vllm/

```
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

这两项最常见：

-   `temperature=0.8`
     控制“随机性/发散程度”。
    -   更小（如 0.2）：更保守、更确定
    -   更大（如 1.2）：更随机、更有创意但更飘
-   `top_p=0.95`（核采样 / nucleus sampling）
     每一步只在“累计概率达到 0.95 的候选 token 集合”里采样，避免长尾极小概率 token 造成胡言乱语。

（你也可以常用 `max_tokens=...` 控制输出长度，否则默认可能生成较长。）

------

## 4) 创建并加载模型 LLM

```
llm = LLM(model="facebook/opt-125m")
```

-   这里指定 Hugging Face 模型名：`facebook/opt-125m`。
-   vLLM 会：
    1.  从 HF 下载/读取模型权重（本地缓存则直接用缓存）
    2.  初始化推理引擎（CUDA / CPU 取决于你的环境）
    3.  准备 KV cache、调度器等以支持高吞吐生成









------

## 5) 调用 generate 批量生成

```
outputs = llm.generate(prompts, sampling_params)
```

-   输入：prompt 列表 + 采样参数
-   输出：一个列表，每个元素对应一个 prompt 的结果对象（`RequestOutput`）
-   vLLM 内部做了：
    -   自动 batching、动态调度
    -   对每个 prompt 进行 tokenization
    -   自回归逐 token 生成，直到满足停止条件（EOS、长度限制、stop tokens 等）

------

## 6) 读取并打印每条结果

```
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

这里有几个层次：

-   `output.prompt`：原始 prompt 字符串
-   `output.outputs`：**一个列表**，因为一次请求可能生成多条候选（当你设置 `n>1` 或类似参数时）
-   `output.outputs[0].text`：取第一条候选的生成文本（仅“续写部分”，不一定包含原 prompt）

打印格式里：

-   `{prompt!r}` 用 `repr()` 形式打印字符串，会把换行、引号等显示得更清晰。

------

## 7)（建议）让输出更可控的常用补充

如果你想每条只生成固定长度，比如 32 token：

```
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=32,
)
```

如果你想遇到某些字符串就停止：

```
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    stop=["\n\n", "###"]
)
```

------

```python
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

with LLM.deprecate_legacy_api():
    # 在这个 with 块里，LLM.DEPRECATE_LEGACY == True
    # 这里调用一些“legacy 用法”时，会按装饰器里的 is_deprecated=lambda: LLM.DEPRECATE_LEGACY
    # 去决定是否触发弃用提示/行为（例如某些 deprecated 参数路径）
    outputs = llm.generate(
        prompts="Hello",
        sampling_params=SamplingParams(),
        prompt_token_ids=None,  # legacy 路径相关参数（示例）
    )

# 出了 with 块，LLM.DEPRECATE_LEGACY == False
```





# LLM 调用流程

`vllm/entrypoints/llm.py `

```python
def generate(self, prompts, sampling_params=None, ...):
    # 1. 验证 runner_type
    runner_type = self.llm_engine.model_config.runner_type
    
    # 2. 转换输入格式（处理 prompt_token_ids）
    if prompt_token_ids is not None:
        parsed_prompts = self._convert_v1_inputs(prompts, prompt_token_ids)
    else:
        parsed_prompts = prompts
    
    # 3. 处理 guided_options
    if isinstance(guided_options_request, dict):
        guided_options_request = GuidedDecodingRequest(**guided_options_request)
    
    # 4. 使用默认采样参数（如果未提供）
    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()
    
    # 5. 验证并添加所有请求
    self._validate_and_add_requests(
        prompts=parsed_prompts,
        params=sampling_params,
        lora_request=lora_request,
        ...
    )
    
    # 6. 运行引擎直到所有请求完成
    outputs = self._run_engine(use_tqdm=use_tqdm)
    
    # 7. 验证并返回输出
    return self.engine_class.validate_outputs(outputs, RequestOutput)
```

`vllm/entrypoints/llm.py`

```python
def generate(
    self,
    prompts: Union[Union[PromptType, Sequence[PromptType]],
                   Optional[Union[str, list[str]]]] = None,
    sampling_params: Optional[Union[SamplingParams,
                                    Sequence[SamplingParams]]] = None,
    prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
    use_tqdm: bool = True,
    lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    guided_options_request: Optional[Union[LLMGuidedOptions,
                                           GuidedDecodingRequest]] = None,
    priority: Optional[list[int]] = None,
) -> list[RequestOutput]:
    # -------------------------------------------
    # 1. 验证 runner_type
    # -------------------------------------------
    # 读取当前 engine 初始化时指定的 runner 类型（即 task 类型）
    # 常见是 generate / embedding / scoring / transcription 等
    runner_type = self.llm_engine.model_config.runner_type

    # generate() 这个入口只允许在“生成类模型”或“转写类模型”上使用
    # 否则直接报错（这就是你前面说的：验证输入参数合法性的一部分）
    if runner_type not in ["generate", "transcription"]:
        messages = [
            "LLM.generate() is only supported for (conditional) generation "
            "models (XForCausalLM, XForConditionalGeneration).",
        ]

        # 当前模型支持哪些 runner 类型（例如 generate / embedding 等）
        supported_runner_types = self.llm_engine.model_config \
            .supported_runner_types

        # 如果模型其实支持 generate，但是用户初始化的时候没选 generate task
        # 就提示用户用 --task generate 启动
        if "generate" in supported_runner_types:
            messages.append(
                "Your model supports the 'generate' runner, but is "
                f"currently initialized for the '{runner_type}' runner. "
                "Please initialize vLLM using `--task generate`.")

        # 直接抛异常，阻断后续流程
        raise ValueError(" ".join(messages))

    # -------------------------------------------
    # 下面开始处理用户输入 prompt（processor 的核心部分）
    # 对应你前面描述的：
    # 2）process req（原始 req -> EngineCoreRequest 的前置处理阶段）
    # -------------------------------------------

    # 如果用户直接传了 prompt_token_ids（已经tokenize好的token id序列）
    # 说明用户可能绕过文本prompt，直接输入 token ids
    if prompt_token_ids is not None:

        # 将 v1 版本的输入参数统一转换成内部结构 parsed_prompts
        # 包括 prompts / prompt_token_ids 的兼容处理
        # 这里的核心作用：
        # - 对输入格式做规范化
        # - 可能做一些基本合法性检查
        parsed_prompts = self._convert_v1_inputs(
            prompts=cast(Optional[Union[str, list[str]]], prompts),
            prompt_token_ids=prompt_token_ids,
        )
    else:
        # 否则走普通的 prompt 输入路径（字符串 or PromptType）
        # cast 的作用只是做类型提示，不改变运行逻辑
        parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                              prompts)

    # -------------------------------------------
    # guided decoding 参数处理（属于 req 参数预处理/校验）
    # -------------------------------------------

    # 如果 guided_options_request 是 dict，说明用户是以 json-style传参
    # 那么这里把 dict 转换成内部对象 GuidedDecodingRequest
    if isinstance(guided_options_request, dict):
        # 强限制：只能同时指定一种 guided decoding（例如 json schema / regex）
        if len(guided_options_request) > 1:
            raise ValueError(
                "You can only use one guided decoding but multiple is "
                f"specified: {guided_options_request}")

        # dict -> GuidedDecodingRequest（内部统一结构）
        guided_options_request = GuidedDecodingRequest(
            **guided_options_request)

    # -------------------------------------------
    # sampling_params 处理（属于 req 参数预处理/校验）
    # -------------------------------------------

    # sampling_params如果没传，就用默认采样参数
    # 比如 temperature/top_p/max_tokens 等默认值
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = self.get_default_sampling_params()

    # -------------------------------------------
    # 关键：validate + add requests（你说的 processor 核心入口）
    # 对应你前面描述的：
    # 1）add req
    # 2）process req（tokenize/校验/封装）
    # 3）encode & send req（进入 engine 后续流程）
    # -------------------------------------------

    self._validate_and_add_requests(
        prompts=parsed_prompts,             # 规范化后的 prompts
        params=sampling_params,             # 采样参数
        lora_request=lora_request,          # LoRA（如果有）
        prompt_adapter_request=prompt_adapter_request,  # Prompt Adapter（如果有）
        guided_options=guided_options_request,          # guided decoding配置
        priority=priority)                  # 请求优先级（调度相关）

    # -------------------------------------------
    # 启动 engine 跑推理
    # 对应你前面描述的：
    # 5）add req to scheduler
    # 6）one step inference（循环多步）
    # 7）put output to output_queue
    # 8~9）多进程通信回传 outputs
    # -------------------------------------------

    # _run_engine 内部会不断 step() 执行调度 + 推理
    # offline batching 模式下通常会一直跑到所有 request 完成
    outputs = self._run_engine(use_tqdm=use_tqdm)

    # -------------------------------------------
    # outputs 后处理 & 类型校验
    # 对应你前面描述的：
    # 10）process_output（比如 detokenize）
    # 11）输出所有 req 结果
    # -------------------------------------------

    # engine_class.validate_outputs:
    # 1) 检查 outputs 结构合法
    # 2) 可能会做 detokenize / 组装成 RequestOutput 这种面向用户的输出对象
    return self.engine_class.validate_outputs(outputs, RequestOutput)

```





## 1.验证 runner_type

```python
# -------------------------------------------
# 验证 runner_type
# -------------------------------------------

# 读取当前 engine 初始化时指定的 runner 类型（即 task 类型）
# 常见是 generate / embedding / scoring / transcription 等
runner_type = self.llm_engine.model_config.runner_type

# generate() 这个入口只允许在“生成类模型”或“转写类模型”上使用
# 否则直接报错（这就是你前面说的：验证输入参数合法性的一部分）
if runner_type not in ["generate", "transcription"]:
    messages = [
        "LLM.generate() is only supported for (conditional) generation "
        "models (XForCausalLM, XForConditionalGeneration).",
    ]

    # 当前模型支持哪些 runner 类型（例如 generate / embedding 等）
    supported_runner_types = self.llm_engine.model_config \
    .supported_runner_types

    # 如果模型其实支持 generate，但是用户初始化的时候没选 generate task
    # 就提示用户用 --task generate 启动
    if "generate" in supported_runner_types:
        messages.append(
            "Your model supports the 'generate' runner, but is "
            f"currently initialized for the '{runner_type}' runner. "
            "Please initialize vLLM using `--task generate`.")

        # 直接抛异常，阻断后续流程
        raise ValueError(" ".join(messages))

```





### `_convert_v1_inputs`

使用`_convert_v1_inputs`兼容Prompt的各种写法

```python
# LEGACY  （旧版本兼容逻辑）
def _convert_v1_inputs(
    self,
    prompts: Optional[Union[str, list[str]]],  # 传入的 prompt：可以是单个字符串或字符串列表
    prompt_token_ids: Optional[Union[list[int], list[list[int]]]],  # 传入的 token id：可以是 1维或2维
):
    # skip_tokenizer_init 现在在 engine 里检查，所以这里不处理

    # 如果 prompts 不为空，把它统一转换成 batch 格式
    if prompts is not None:
        # parse_and_batch_prompt 会把输入规范化为 list，并转成统一结构
        # 每个元素大概是 {"content": xxx, ...}
        # 这里只取 "content" 部分，得到纯文本 prompt 列表
        prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]

    # 如果 prompt_token_ids 不为空，也统一转换成 batch 格式
    if prompt_token_ids is not None:
        # 同样 parse_and_batch_prompt 做规范化
        # 这里只取 "content" 部分，得到 token id 列表（可能是 list[int] 或 list[list[int]]）
        prompt_token_ids = [
            p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
        ]

    # num_requests 用来表示最终要处理多少条请求（batch size）
    num_requests = None

    # 如果 prompts 有值，那么请求数 = prompts 的长度
    if prompts is not None:
        num_requests = len(prompts)

    # 如果 prompt_token_ids 有值：
    if prompt_token_ids is not None:
        # 如果 num_requests 已经由 prompts 决定了，
        # 那么必须保证 prompts 和 prompt_token_ids 的长度一致
        if (num_requests is not None
                and num_requests != len(prompt_token_ids)):
            raise ValueError(
                "The lengths of prompts and prompt_token_ids "
                "must be the same."
            )

        # 否则（如果 num_requests 还没定），用 token ids 的长度作为请求数
        num_requests = len(prompt_token_ids)

    # 如果 prompts 和 prompt_token_ids 都是 None，则报错（必须提供一个）
    if num_requests is None:
        raise ValueError(
            "Either prompts or prompt_token_ids must be "
            "provided."
        )

    # parsed_prompts：最终统一输出的 PromptType 列表（每条请求一个 PromptType）
    parsed_prompts: list[PromptType] = []

    # 循环生成每一条请求的 PromptType 对象
    for i in range(num_requests):
        item: PromptType  # 每条请求的 prompt 表示

        # 优先用 prompts（文本 prompt）
        if prompts is not None:
            # TextPrompt 是封装后的结构，例如 TextPrompt(prompt="hello")
            item = TextPrompt(prompt=prompts[i])

        # 如果没有 prompts，就用 token ids
        elif prompt_token_ids is not None:
            # TokensPrompt 是 token 形式封装，例如 TokensPrompt(prompt_token_ids=[1,2,3])
            item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])

        # 理论上不会走到这里，因为前面 num_requests 已经保证至少一个非 None
        else:
            raise AssertionError

        # 加入最终列表
        parsed_prompts.append(item)

    # 返回统一后的 prompts 列表
    return parsed_prompts

```



### ParsedText 

**这两段 TypedDict 是在定义两种“解析后的输入格式”：文本版和 token 版，并用 `is_tokens` 作为标签来区分。**

```python
class ParsedText(TypedDict):
    content: str
    is_tokens: Literal[False]


class ParsedTokens(TypedDict):
    content: list[int]
    is_tokens: Literal[True]

```

“判别联合类型”（Discriminated Union / Tagged Union）

`is_tokens` 就是“标签”。

这样类型检查器（mypy/pyright）就能推断：

-   如果 `is_tokens == False` → `content` 一定是 `str`↳
-   如果 `is_tokens == True` → `content` 一定是 `list[int]`

例如：

```
x = parse_and_batch_prompt(...)
if x["is_tokens"]:
    # 这里 content 被推断为 list[int]
    ids = x["content"]
else:
    # 这里 content 被推断为 str
    text = x["content"]
```



## 2.处理用户输入 prompt

**有 token ids 就走 v1 转换；没有就直接用 prompts，最终都得到统一的 `parsed_prompts`**。

```python
# -------------------------------------------
# 下面开始处理用户输入 prompt（processor 的核心部分）
# 对应你前面描述的：
# 2）process req（原始 req -> EngineCoreRequest 的前置处理阶段）
# -------------------------------------------

# 如果用户直接传了 prompt_token_ids（已经tokenize好的token id序列）
# 说明用户可能绕过文本prompt，直接输入 token ids
if prompt_token_ids is not None:

    # 将 v1 版本的输入参数统一转换成内部结构 parsed_prompts
    # 包括 prompts / prompt_token_ids 的兼容处理
    # 这里的核心作用：
    # - 对输入格式做规范化
    # - 可能做一些基本合法性检查
    parsed_prompts = self._convert_v1_inputs(
        prompts=cast(Optional[Union[str, list[str]]], prompts),
        prompt_token_ids=prompt_token_ids,
    )
    else:
        # 否则走普通的 prompt 输入路径（字符串 or PromptType）
        # cast 的作用只是做类型提示，不改变运行逻辑
        parsed_prompts = cast(Union[PromptType,Sequence[PromptType]], prompts)

```



## 3.guided decoding 参数处理

guided decoding = **给模型生成过程加“格式约束/语法约束”**，确保输出能被机器稳定解析（比如严格 JSON / 正则）。

 `**` 的作用就是：**把字典展开成参数**（dict unpacking）。



```python
# -------------------------------------------
# guided decoding 参数处理（属于 req 参数预处理/校验）
# -------------------------------------------
  
# 如果 guided_options_request 是 dict，说明用户是以 json-style传参
# 那么这里把 dict 转换成内部对象 GuidedDecodingRequest
if isinstance(guided_options_request, dict):
# 强限制：只能同时指定一种 guided decoding（例如 json schema / regex）
if len(guided_options_request) > 1:
raise ValueError(
"You can only use one guided decoding but multiple is "
f"specified: {guided_options_request}")

# dict -> GuidedDecodingRequest（内部统一结构）
guided_options_request = GuidedDecodingRequest(
**guided_options_request)
```

### 为什么只能同时指定一种 guided decoding？

因为不同 guided decoding 规则之间**可能互相冲突**，系统无法同时满足。

比如你同时写：

```
guided_options_request = {
  "regex": r"\d+",
  "json_schema": {"type": "object"}
}
```

这就矛盾了：

-   `regex` 要求输出只能是 **数字字符串**，例如 `"123"`
-   `json_schema` 要求输出是 **JSON 对象**，例如 `{"x":123}`

两者不可能同时成立。



### `GuidedDecodingRequest`

`vllm/model_executor/guided_decoding/guided_fields.py`

```python
@dataclass
class GuidedDecodingRequest:
    """One of the fields will be used to retrieve the logit processor."""
    guided_json: Optional[Union[Dict, BaseModel, str]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None
    guided_decoding_backend: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
    guided_json_object: Optional[bool] = None

    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        guide_count = sum([
            self.guided_json is not None, self.guided_regex is not None,
            self.guided_choice is not None, self.guided_grammar is not None,
            self.guided_json_object is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding but multiple are "
                f"specified: {self.__dict__}")

```





## 4.sampling_params 处理

```python
# -------------------------------------------
# sampling_params 处理（属于 req 参数预处理/校验）
# -------------------------------------------

# sampling_params如果没传，就用默认采样参数
# 比如 temperature/top_p/max_tokens 等默认值
if sampling_params is None:
    # Use default sampling params.
    sampling_params = self.get_default_sampling_params()
```



`vllm/entrypoints/llm.py`

```python
def get_default_sampling_params(self) -> SamplingParams:
    if self.default_sampling_params is None:
        self.default_sampling_params = (
            self.llm_engine.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()
```



`vllm/config.py`

提取 HuggingFace generation_config 里“会影响采样”的参数，转换成 vLLM 格式返回，用来覆盖 vLLM 默认采样设置；如果你指定 generation_config=vllm，就永远不覆盖。

vLLM 默认采样参数被模型作者的 HF generation config 覆盖了 如果不想这样，请用 `--generation-config vllm`



```python
def get_diff_sampling_param(self) -> dict[str, Any]:
    """
    返回一个字典：里面只包含“和 vLLM 默认采样参数不同/需要覆盖的采样参数”。

    特殊情况：
    - 如果 generation_config == "vllm"
      表示强制使用 vLLM 默认采样参数
      -> 返回空字典 {}
    """

    # 1）决定采样参数 config 从哪里来
    if self.generation_config == "vllm":
        # 用户明确要求使用 vLLM 默认采样参数
        # 所以不从 HuggingFace generation_config 里读取任何设置
        config = {}
    else:
        # 读取 HuggingFace 模型的 generation_config（可能包含模型作者推荐的采样参数）
        config = self.try_get_generation_config()

    # 2）再用用户显式传入的 override_generation_config 覆盖（优先级更高）
    # 比如用户手动指定 temperature/top_p 等
    config.update(self.override_generation_config)

    # 3）只关心这些“采样相关”的参数（其余的忽略）
    available_params = [
        "repetition_penalty",  # 重复惩罚
        "temperature",         # 温度
        "top_k",               # top-k 采样
        "top_p",               # nucleus sampling
        "min_p",               # 最小概率阈值
        "max_new_tokens",      # HuggingFace 的“最多生成 token 数”
    ]

    # 4）如果 config 里至少出现了一个采样参数，就开始抽取
    if any(p in config for p in available_params):
        # 只把 config 中存在且不是 None 的采样参数挑出来
        diff_sampling_param = {
            p: config.get(p)
            for p in available_params
            if config.get(p) is not None
        }

        # 5）字段名映射：
        # HuggingFace 用 max_new_tokens
        # vLLM 用 max_tokens
        # 所以做一次改名，保证后续 vLLM 能识别
        if "max_new_tokens" in diff_sampling_param:
            diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                "max_new_tokens"
            )
    else:
        # config 里完全没有这些采样参数 -> 没必要覆盖 vLLM 默认值
        diff_sampling_param = {}

    # 6）如果确实产生了覆盖参数，则打印一次 warning 提醒用户
    if diff_sampling_param:
        logger.warning_once(
            "Default sampling parameters have been overridden by the "
            "model's Hugging Face generation config recommended from the "
            "model creator. If this is not intended, please relaunch "
            "vLLM instance with `--generation-config vllm`."
        )

    # 7）返回最终的“差异采样参数字典”
    return diff_sampling_param

```

```python
get_default_sampling_params()
  ↓
【首次调用】
  ↓
model_config.get_diff_sampling_param()
  ├─ 检查 generation_config 设置
  │   ├─ 如果 "vllm" → 返回 {}
  │   └─ 否则 → try_get_generation_config()
  │       ├─ 从 HuggingFace 加载 GenerationConfig
  │       └─ 转换为差异字典
  ├─ 应用 override_generation_config
  └─ 提取可用参数（temperature, top_p, top_k, ...）
  ↓
【如果有差异参数】
  ↓
SamplingParams.from_optional(**diff_params)
  └─ 创建 SamplingParams 实例
  ↓
【如果没有差异参数】
  ↓
SamplingParams()  ← 使用默认值
```





## 5. validate + add requests（ processor 核心入口）

```python
# -------------------------------------------
# 关键：validate + add requests（processor 核心入口）
# 对应你前面描述的：
# 1）add req
# 2）process req（tokenize/校验/封装）
# 3）encode & send req（进入 engine 后续流程）
# -------------------------------------------

self._validate_and_add_requests(
    prompts=parsed_prompts,             # 规范化后的 prompts
    params=sampling_params,             # 采样参数
    lora_request=lora_request,          # LoRA（如果有）
    prompt_adapter_request=prompt_adapter_request,  # Prompt Adapter（如果有）
    guided_options=guided_options_request,          # guided decoding配置
    priority=priority)                  # 请求优先级（调度相关）

```



`vllm/entrypoints/llm.py`

-   **校验输入是否匹配**（prompt 数量、params 数量、lora 数量）
-   **把请求逐条塞到引擎队列**（每条 prompt 对应一条 request）



`vllm/inputs/data.py`

```
PromptType = Union[SingletonPrompt, ExplicitEncoderDecoderPrompt]
```

PromptType` 要么是 `SingletonPrompt`，要么是 `ExplicitEncoderDecoderPrompt



```
SingletonPrompt = Union[str, TextPrompt, TokensPrompt]
```

SingletonPrompt = 普通字符串 str 或 TextPrompt 类型对象 TextPrompt 或 TokensPrompt 类型对象 TokensPrompt

:::note

Python 3.10+ 简写

```
SingletonPrompt = str | TextPrompt | TokensPrompt
```

:::



```
class TextPrompt(TypedDict):
    prompt: str
    multi_modal_data: NotRequired["MultiModalDataDict"]
    mm_processor_kwargs: NotRequired[dict[str, Any]]
```

-   **`prompt: str`（必填）**
     你直接输入的原始文本，后续会先 tokenization，再喂给模型。
-   **`multi_modal_data`（可选）**
     如果模型支持多模态，这里可以附带图片/音频等额外输入数据。
-   **`mm_processor_kwargs`（可选）**
     给多模态处理器用的额外参数，比如图像 resize、归一化、采样率等配置。





`vllm/entrypoints/llm.py`

`_validate_and_add_requests()` 做的是：

**批量处理 + 输入格式统一 + 一次性校验**

它负责：

-   把单条 prompt 统一变成 list
-   检查 `prompts` 和 `params / lora_request` 长度是否匹配
-   处理 guided_options 旧接口兼容
-   给 SamplingParams 做统一设置（比如 `FINAL_ONLY`）
-   最后循环逐个调用 `_add_request()`

✅ 就像一个“批量装配线入口”

```python
def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:

        # -----------------------------
        # 1）兼容老接口：guided_options 参数已废弃
        # -----------------------------
        if guided_options is not None:
            warnings.warn(
                # 提醒：guided_options_request 已弃用
                # 现在推荐把 guided decoding 放到 SamplingParams.guided_decoding 里
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        # -----------------------------
        # 2）如果 prompts 是单条（str 或 dict），统一包装成 list
        # -----------------------------
        if isinstance(prompts, (str, dict)):
            # 统一成 batch 格式：后续逻辑就只需要处理 list 了
            prompts = [prompts]

        # 请求数量 = prompt 条数
        num_requests = len(prompts)

        # -----------------------------
        # 3）合法性检查：params 如果是 list，必须跟 prompts 等长
        # params
        # -----------------------------
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")

        # -----------------------------
        # 4）合法性检查：lora_request 如果是 list，也必须跟 prompts 等长
        # 正确：每条 prompt 对应一个 LoRA lora_request = [lora1, lora2, lora3]
        # -----------------------------
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        # -----------------------------
        # 5）遍历 params（可能是单个，也可能是 list）
        # 对 SamplingParams 做额外处理
        # -----------------------------
        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):

                # 将老的 guided_options 合并/写入到 SamplingParams 中
                # （兼容旧 guided_options_request 的入口）
                self._add_guided_params(sp, guided_options)

                # 只返回最终结果（不返回中间 token 流式输出）
                # 即：FINAL_ONLY 代表只关心最终输出
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # -----------------------------
        # 6）核心：把每一条请求加入引擎队列（engine）
        # -----------------------------
        for i, prompt in enumerate(prompts):
            self._add_request(
                # 第 i 条 prompt
                prompt,

                # 如果 params 是 list，用 params[i]
                # 否则说明所有 prompt 共用同一个 params
                params[i] if isinstance(params, Sequence) else params,

                # 如果 lora_request 是 list，用 lora_request[i]
                # 否则说明所有 prompt 共用同一个 lora_request
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,

                # prompt adapter 通常是“全局统一的一个”，所有请求共享
                prompt_adapter_request=prompt_adapter_request,

                # priority 如果给了 list，每条请求用自己的 priority[i]
                # 否则默认优先级为 0
                priority=priority[i] if priority else 0,
            )
```

```python
def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )
```





### When would you NOT force FINAL_ONLY?

They force `FINAL_ONLY` to **standardize outputs, simplify engine logic, improve batching performance, and avoid breaking callers**. 强制 `FINAL_ONLY` 是为了 **统一返回格式、降低引擎复杂度、提升 batch 性能、避免调用方出错**。

Many frameworks separate generation into:

-   non-streaming (final result only)
-   streaming (token-by-token)

This function is likely the non-streaming path, so it forces `FINAL_ONLY`.



### 什么是lora

LoRA = Low-Rank Adaptation（低秩适配）这次生成不要只用基础模型，而是额外加载一个叫 **sql-lora** 的 LoRA 适配器，让模型更擅长“把问题转成 SQL”。

**LoRA = 给模型“外挂一组小参数”，让模型临时学会某个特定能力/风格/领域**，但不需要改动整个大模型权重。

 **LoRA 就是一个“为特定任务训练的小型增量权重”，请求时通过 `LoRARequest(...)` 挂载到基础模型上使用。**

```python
class LoRARequest(msgspec.Struct):
    # 必需字段
    lora_name: str                    # LoRA 适配器的名称（人类可读）
    lora_int_id: int                 # LoRA 适配器的全局唯一整数ID（必须 > 0）
    
    # 可选字段
    lora_path: str = ""              # LoRA 适配器的路径（本地或远程）
    lora_local_path: Optional[str] = None  # 已弃用，使用 lora_path
    long_lora_max_len: Optional[int] = None  # LongLoRA 的最大长度
    base_model_name: Optional[str] = None    # 基础模型名称
```

举个效果例子

同样问：“写一个 SQL 查询 Lilongwe International Airport 的 ICAO”

不用 LoRA（普通模型可能不稳）

可能输出带解释、SQL 不标准

用 `sql-lora`

更可能输出标准 SQL，例如：

```
SELECT icao
FROM table_name_74
WHERE airport = 'lilongwe international airport';
```





### prompt_adapter_request是什么？

**Prompt Adapter**：更像“自动加一个固定的系统提示/前缀”（改输入形式）

1）训练/导出时自己命名（最常见）

比如你训练了一个 prompt adapter（soft prompt / p-tuning v2），保存时就会有一个名字：

```
sql_prompt_adapter
customer_support_adapter
json_format_adapter
```

之后推理时就用这个名字来指定加载哪个 adapter。



 2）部署时在服务端“注册表/配置文件”里定义

很多推理服务会有一个 mapping，例如：

```
prompt_adapters:
  sql_adapter:
    path: /models/adapters/sql_adapter
  cs_adapter:
    path: /models/adapters/customer_support
```

那调用时只能用这些登记过的名字：

```
prompt_adapter_request = {"adapter_name": "sql_adapter"}
```



3）来自 HuggingFace / 目录名（约定俗成）

如果 adapter 存在某个目录：

```
/models/adapters/sql_adapter/
```

很多系统就直接把文件夹名当 adapter name：

```
adapter_name = "sql_adapter"
```

------



### Prompt Adapter 常见训练方式

训练完会保存一份小权重，比如：

```python
/models/adapters/sql_adapter/
adapter_config.json
adapter_model.bin  (或 safetensors)
```

1）Prompt Tuning（最常见）

-   只训练“soft prompt embedding”↳
-   很轻量
-   适合：格式控制、任务指令强化

2）Prefix Tuning

-   在每一层 attention 里加 prefix KV
-   更强一些，但实现复杂

3）P-Tuning v2

-   更强、更稳定
-   常用于多任务适配



### Prompt Adapter 和 LoRA 的训练区别

-   **Prompt Adapter**：训练“输入前缀向量”，更像“隐形 prompt”
-   **LoRA**：训练模型内部的一部分线性层增量权重，更像“改模型能力”





`vllm/engine/llm_engine.py`

`add_request()` 做的是：

**单条 request 的严格处理 + 转换成 scheduler 里的可运行对象**

它负责：

-   兼容旧参数 `inputs` → prompt
-   校验 LoRA / priority / multi-step decoding 限制
-   设置 arrival_time↳
-   tokenizer 校验 prompt↳
-   preprocess → process↳
-   `_add_processed_request()` 真正加入 scheduler

✅ 就像“真正把请求塞进引擎执行系统里”的那一步

```python
@deprecate_kwargs(
    "inputs",
    additional_message="Please use the 'prompt' parameter instead.",
)
# 1) 装饰器：标记 inputs 参数已废弃（deprecated）
#    如果用户仍然传 inputs=xxx，会提示“请改用 prompt 参数”

def add_request(
        self,
        request_id: str,
        prompt: Optional[PromptType] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None,  # DEPRECATED
) -> None:
    """
    ... 省略 docstring ...
    """

    if inputs is not None:
        prompt = inputs
    # 2) 兼容旧参数 inputs：
    #    如果用户传入 inputs，就把它当成 prompt 用（向后兼容）

    assert prompt is not None and params is not None
    # 3) 断言检查：prompt 和 params 必须都有
    #    否则直接报错（AssertionError）

    if lora_request is not None and not self.lora_config:
        raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                         "not enabled!")
    # 4) LoRA 检查：
    #    如果用户传了 lora_request，但 engine 没有启用 LoRA 功能，就报错

    if priority != 0 and not self.scheduler_config.policy == "priority":
        raise ValueError(f"Got priority {priority} but "
                         "Priority scheduling is not enabled.")
    # 5) priority 调度检查：
    #    如果用户设置 priority（非0），但 scheduler 不是 priority 模式，就报错

    if isinstance(params, SamplingParams) \
        and (params.guided_decoding or params.logits_processors) \
        and self.scheduler_config.num_scheduler_steps > 1:
        raise ValueError(
            "Guided decoding and logits processors are not supported "
            "in multi-step decoding")
    # 6) 多步解码限制：
    #    如果 params 是 SamplingParams（生成模式）
    #    且使用 guided decoding 或 logits processors
    #    且 scheduler 处于 multi-step decoding（num_scheduler_steps > 1）
    #    就报错：因为这些功能在多步解码下不支持

    if arrival_time is None:
        arrival_time = time.time()
    # 7) arrival_time 默认值：
    #    如果没传 arrival_time，就用当前时间（秒级时间戳）
    #    用于调度器判断请求先后/等待时间等

    if self.tokenizer is not None:
        self._validate_token_prompt(
            prompt,
            tokenizer=self.get_tokenizer(lora_request=lora_request))
    # 8) prompt token 校验：
    #    如果 engine 有 tokenizer，则验证 prompt 的格式/内容是否合法
    #    注意这里 tokenizer 可能和 lora_request 绑定（LoRA可能换词表）

    preprocessed_inputs = self.input_preprocessor.preprocess(
        prompt,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
    )
    # 9) 输入预处理 preprocess：
    #    将用户传入的 prompt（各种 PromptType）
    #    统一转换成 vLLM 内部标准结构（可能包含 tokenization、multi-modal解析等）

    processed_inputs = self.input_processor(preprocessed_inputs)
    # 10) 输入处理 input_processor：
    #     在 preprocess 基础上进一步加工成 engine 真正可执行的结构
    #     比如 prompt_token_ids、attention metadata、batch 信息等

    self._add_processed_request(
        request_id=request_id,
        processed_inputs=processed_inputs,
        params=params,
        arrival_time=arrival_time,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        trace_headers=trace_headers,
        priority=priority,
    )
    # 11) 把请求加入调度器/队列（关键一步）：
    #     创建内部的 Sequence / SequenceGroup，并注册到 scheduler 中
    #     之后通过 engine.step() 执行推理，逐步生成 token

```

### PromptType









## 6.运行引擎直到所有请求完成

```
# 6. 运行引擎直到所有请求完成
outputs = self._run_engine(use_tqdm=use_tqdm)
```





## 7.验证并返回输出

```
# 7. 验证并返回输出
return self.engine_class.validate_outputs(outputs, RequestOutput)
```







