---
title: "vLLM v0&v1"
published: 2026-01-30
description: "vLLM v0&v1"
image: ""
tags: ["vllm","vLLM v0&v1"]
category: vllm
draft: false
lang: ""
---

# v0.8.2

| 特性        | v0                                                           | v1                                                           |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 异步        | ✅ AsyncLLMEngine<br>vllm/engine/async_llm_engine.py          | ✅ AsyncLLM<br>vllm/v1/engine/async_llm.py                    |
| 多进程      | ✅ MQLLMEngine + MQLLMEngineClient + ZMQ<br>引擎：vllm/engine/multiprocessing/engine.py<br>客户端：vllm/engine/multiprocessing/client.py | ✅ SyncMPClient / AsyncMPClient + ZMQ<br>vllm/v1/engine/core_client.py |
| 架构        | AsyncLLMEngine 包装 LLMEngine<br>引擎：vllm/engine/llm_engine.py<br>异步包装：vllm/engine/async_llm_engine.py | EngineCoreClient 抽象层，统一接口<br>vllm/v1/engine/core_client.py<br>引擎：vllm/v1/engine/llm_engine.py |
| 多进程+异步 | ✅ 支持（MQLLMEngine + MQLLMEngineClient）<br>引擎：vllm/engine/multiprocessing/engine.py<br>客户端：vllm/engine/multiprocessing/client.py<br>入口：vllm/entrypoints/openai/api_server.py（build_async_engine_client_from_engine_args） | ✅ 支持（AsyncMPClient）<br>vllm/v1/engine/core_client.py<br>入口：vllm/entrypoints/openai/api_server.py（V1 分支用 AsyncLLM.from_vllm_config） |



# v0

## 1）入口调用链

用户调用 **`LLM.generate()`**（`vllm/entrypoints/llm.py` 第 374 行）：对外入口，把一批 prompt 和 sampling_params 交给引擎做生成。内部会调用 **`_add_request()`**（`vllm/entrypoints/llm.py` 第 1317 行）：为单个请求生成 request_id，并调用 **`self.llm_engine.add_request()`**（`vllm/entrypoints/llm.py` 第 1326 行）：把该请求交给 LLMEngine 的请求池。

### llm.generate() (examples/offline_inference/basic/basic.py 第19行)

```python
outputs = llm.generate(prompts, sampling_params)
```

 

### generate()（vllm/entrypoints/llm.py 第 374 行）

```python
@deprecate_kwargs(
    "prompt_token_ids",
    # 是否启用弃用：由 LLM.DEPRECATE_LEGACY 开关控制
    is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
    # 额外提示信息：告诉用户用 prompts 替代 prompt_token_ids
    additional_message="Please use the 'prompts' parameter instead.",
)
def generate(
    self,
    # prompts：输入的 prompt，可以是单个/多个 prompt，也可以是字符串列表等形式
    prompts: Union[Union[PromptType, Sequence[PromptType]],
                   Optional[Union[str, list[str]]]] = None,

    # sampling_params：采样参数（temperature/top_p/max_tokens 等）
    # 可以是一个 SamplingParams（对所有 prompts 生效）
    # 或一个 SamplingParams 列表（和 prompts 一一对应）
    sampling_params: Optional[Union[SamplingParams,
                                    Sequence[SamplingParams]]] = None,

    # prompt_token_ids：旧接口，直接传 token id（已弃用趋势）
    prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,

    # use_tqdm：是否显示 tqdm 进度条
    use_tqdm: bool = True,

    # lora_request：是否使用 LoRA（可选）
    lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,

    # prompt_adapter_request：Prompt Adapter（可选）
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,

    # guided_options_request：引导解码（如 JSON schema/CFG/regex 等）
    guided_options_request: Optional[Union[LLMGuidedOptions,
                                           GuidedDecodingRequest]] = None,

    # priority：请求优先级（用于优先级调度策略时）
    priority: Optional[list[int]] = None,
) -> list[RequestOutput]:
    """对输入的 prompts 进行文本生成，并返回结果。

    该类会自动把 prompts 按照显存/内存约束进行 batching。
    性能最佳实践：把所有 prompts 放进一个 list 里一次性调用。

    参数说明略...

    Note:
        使用 prompts 和 prompt_token_ids 作为关键字参数属于 legacy 写法，
        未来可能会弃用。推荐改成通过 inputs 参数传入（更统一）。
    """

    # 当前模型的 runner 类型（类似任务模式），比如 generate/transcription/embed/score 等
    runner_type = self.llm_engine.model_config.runner_type

    # generate 只支持生成类模型（generate / transcription）
    # 如果不是这两种模式，就直接报错
    if runner_type not in ["generate", "transcription"]:
        messages = [
            "LLM.generate() is only supported for (conditional) generation "
            "models (XForCausalLM, XForConditionalGeneration).",
        ]

        # supported_runner_types：该模型本身支持的 runner 类型
        supported_runner_types = self.llm_engine.model_config \
            .supported_runner_types

        # 如果模型支持 generate，但当前初始化不是 generate 模式
        # 提示用户用 --task generate 重新初始化
        if "generate" in supported_runner_types:
            messages.append(
                "Your model supports the 'generate' runner, but is "
                f"currently initialized for the '{runner_type}' runner. "
                "Please initialize vLLM using `--task generate`.")

        raise ValueError(" ".join(messages))

    # 如果用户传了 prompt_token_ids（旧接口）
    if prompt_token_ids is not None:
        # 将旧参数形式转换成 v1 的 inputs 格式
        # prompts 这里强制 cast 成 str 或 list[str]（因为 _convert_v1_inputs 期望这种）
        parsed_prompts = self._convert_v1_inputs(
            prompts=cast(Optional[Union[str, list[str]]], prompts),
            prompt_token_ids=prompt_token_ids,
        )
    else:
        # 否则直接使用 prompts（新接口/标准接口）
        parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                              prompts)

    # guided_options_request 如果是 dict（说明用户传的是配置字典）
    if isinstance(guided_options_request, dict):
        # 这里要求 guided_options_request 只能包含一个 guided decoding 配置
        # （比如只能选一种 guided decoding，不允许多种同时开启）
        if len(guided_options_request) > 1:
            raise ValueError(
                "You can only use one guided decoding but multiple is "
                f"specified: {guided_options_request}")

        # 将 dict 转成 GuidedDecodingRequest 对象
        guided_options_request = GuidedDecodingRequest(
            **guided_options_request)

    # sampling_params 如果没传，则使用默认采样参数
    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()

    # 校验请求参数并把请求加入 engine 队列
    self._validate_and_add_requests(
        prompts=parsed_prompts,
        params=sampling_params,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        guided_options=guided_options_request,
        priority=priority
    )

    # 真正执行推理/生成（可能会内部做批处理）
    outputs = self._run_engine(use_tqdm=use_tqdm)

    # 对 outputs 做结构校验，并确保输出类型是 RequestOutput
    return self.engine_class.validate_outputs(outputs, RequestOutput)
```



###  _validate_and_add_requests()（vllm/entrypoints/llm.py 第 1268 行）

```python
def _validate_and_add_requests(
    self,
    # prompts：可以是单个 prompt，也可以是一组 prompt（批量）
    prompts: Union[PromptType, Sequence[PromptType]],

    # params：可以是 SamplingParams / PoolingParams，也可以是一组 params（和 prompts 一一对应）
    params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                  Sequence[PoolingParams]],

    # lora_request：可以是单个 LoRARequest，也可以是每个 prompt 一个（列表）
    lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],

    # prompt_adapter_request：Prompt Adapter（一般是全局共用一个）
    prompt_adapter_request: Optional[PromptAdapterRequest],

    # guided_options：旧 guided decoding 参数（已经过期/废弃中）
    guided_options: Optional[GuidedDecodingRequest] = None,

    # priority：请求优先级列表（用于优先级调度）
    priority: Optional[list[int]] = None,
) -> None:

    # 1) 如果仍然传了 guided_options，发出 DeprecationWarning
    #    现在推荐用 SamplingParams.guided_decoding 来代替
    if guided_options is not None:
        warnings.warn(
            "guided_options_request is deprecated, use "
            "SamplingParams.guided_decoding instead",
            DeprecationWarning,
            stacklevel=2,   # 让 warning 的定位指向调用栈上层（对用户更友好）
        )

    # 2) 如果 prompts 是单个 prompt（str 或 dict），统一转成 list
    #    这样后续逻辑统一按 batch 处理
    if isinstance(prompts, (str, dict)):
        # Convert a single prompt to a list.
        prompts = [prompts]

    # 3) 请求数量（batch size）
    num_requests = len(prompts)

    # 4) 如果 params 是列表，那么长度必须和 prompts 一样，否则无法一一对应
    if isinstance(params, list) and len(params) != num_requests:
        raise ValueError("The lengths of prompts and params "
                         "must be the same.")

    # 5) 如果 lora_request 是列表，也必须和 prompts 一样长
    #    否则无法对每条 prompt 绑定对应的 LoRA
    if isinstance(lora_request, list) and len(lora_request) != num_requests:
        raise ValueError("The lengths of prompts and lora_request "
                         "must be the same.")

    # 6) 遍历 params（如果不是 list，就包成单元素 tuple 统一处理）
    for sp in params if isinstance(params, list) else (params, ):
        # 这里只对 SamplingParams 做特殊处理（PoolingParams 不需要）
        if isinstance(sp, SamplingParams):
            # 将 guided_options 的配置“塞进” SamplingParams（旧兼容逻辑）
            self._add_guided_params(sp, guided_options)

            # 这里强制设置 output_kind 只输出最终结果
            # 不输出中间 token、logprobs 流式信息等（节省开销）
            sp.output_kind = RequestOutputKind.FINAL_ONLY

    # 7) 真正把请求逐条加入 engine（排队等待执行）
    for i, prompt in enumerate(prompts):
        self._add_request(
            prompt,

            # params 如果是 Sequence，就取第 i 个；
            # 否则所有请求共享同一个 params
            params[i] if isinstance(params, Sequence) else params,

            # lora_request 同理：如果是 Sequence，则每条请求一个 LoRA
            lora_request=lora_request[i] if isinstance(
                lora_request, Sequence) else lora_request,

            # Prompt Adapter 通常是所有请求共用一个
            prompt_adapter_request=prompt_adapter_request,

            # priority 如果传了列表，就取 priority[i]；否则默认 0
            priority=priority[i] if priority else 0,
        )
```



### _add_request()（vllm/entrypoints/llm.py 第 1317 行）

```python
def _add_request(
    self,
    # prompt：单条输入（可以是 str，也可以是 dict 格式的 PromptType）
    prompt: PromptType,

    # params：该请求对应的参数（生成用 SamplingParams，或池化/embedding 用 PoolingParams）
    params: Union[SamplingParams, PoolingParams],

    # lora_request：可选，为该请求指定 LoRA（如果启用 LoRA 微调适配）
    lora_request: Optional[LoRARequest] = None,

    # prompt_adapter_request：可选，为该请求指定 Prompt Adapter
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,

    # priority：请求优先级，数值越大通常代表越优先（取决于调度策略实现）
    priority: int = 0,
) -> None:

    # 1) 生成一个唯一的 request_id
    # request_counter 是一个计数器（一般是 itertools.count），每次 next() 递增
    request_id = str(next(self.request_counter))

    # 2) 把请求提交给底层 llm_engine（真正的推理引擎队列）
    # 引擎会负责排队、batching、调度、执行推理等
    self.llm_engine.add_request(
        request_id,                 # 请求唯一 ID
        prompt,                     # prompt 内容
        params,                     # 推理参数（采样/池化）
        lora_request=lora_request,  # 是否挂载 LoRA
        prompt_adapter_request=prompt_adapter_request,  # 是否挂载 Prompt Adapter
        priority=priority,          # 请求优先级
    )
```



## 2）请求预处理

在 v0 的 **LLMEngine** 里，**`add_request()`**（`vllm/engine/llm_engine.py` 第 699 行）：把 (request_id, prompt, params) 加入引擎，并做校验与预处理。内部先调 **`self.input_preprocessor.preprocess()`**（`vllm/engine/llm_engine.py` 第 784 行）：对 prompt 做 tokenize、多模态预处理等，得到 `preprocessed_inputs`。再调 **`self.input_processor()`**（`vllm/engine/llm_engine.py` 第 789 行）：对预处理结果做 registry 里配置的输入处理（如 encoder-decoder 拆分等），得到 `processed_inputs`。最后调 **`_add_processed_request()`**（`vllm/engine/llm_engine.py` 第 791 行）：用已处理好的输入把请求正式加入引擎内部结构。

### add_request()（vllm/engine/llm_engine.py 第699行）

```python
@deprecate_kwargs(
    "inputs",
    additional_message="Please use the 'prompt' parameter instead.",
)
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
        inputs: Optional[PromptType] = None,  # 已废弃参数（DEPRECATED）
) -> None:
    """添加一个请求到引擎的请求池中。

    请求会被加入 request pool，并在不断调用 `engine.step()` 时由 scheduler 逐步执行。
    具体怎么调度（例如公平调度/优先级调度）由 scheduler 的策略决定。

    Args:
        request_id: 请求的唯一 ID
        prompt: LLM 的输入 prompt（支持多种格式，见 PromptType）
        params: 推理参数，可能是采样参数或 pooling 参数
            SamplingParams -> 文本生成
            PoolingParams -> embedding / pooling
        arrival_time: 请求到达时间，如果不传则使用当前时间
        lora_request: LoRA 配置请求（如果开启 LoRA 才能用）
        trace_headers: OpenTelemetry 的 trace header（用于链路追踪）
        prompt_adapter_request: prompt adapter 请求（如用于 prompt 微调/适配）
        priority: 请求优先级（只有 priority 调度策略下才生效）

    Details:
        - 如果 arrival_time 为 None，则设为当前时间
        - 如果 prompt_token_ids 为 None，则把 prompt 编码成 token ids
        - 创建 n 个 Sequence 对象（对应 SamplingParams.n）
        - 用这些 Sequence 构造一个 SequenceGroup
        - 把 SequenceGroup 加入 scheduler

    Example:
        (省略)
    """

    # 如果用户还传了旧参数 inputs，则把它当作 prompt 用
    if inputs is not None:
        prompt = inputs

    # 必须保证 prompt 和 params 都不为空（否则无法运行）
    assert prompt is not None and params is not None

    # 如果请求带了 lora_request，但引擎没有启用 LoRA，则报错
    if lora_request is not None and not self.lora_config:
        raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                         "not enabled!")

    # 如果 priority 不为 0，说明用户想使用优先级调度
    # 但 scheduler 没配置 priority 策略，则报错
    if priority != 0 and not self.scheduler_config.policy == "priority":
        raise ValueError(f"Got priority {priority} but "
                         "Priority scheduling is not enabled.")

    # 如果是 SamplingParams，并且启用了 guided decoding 或 logits processors
    # 同时 scheduler 采用多步解码(num_scheduler_steps > 1)
    # 则不支持，直接报错
    if isinstance(params, SamplingParams) \
        and (params.guided_decoding or params.logits_processors) \
        and self.scheduler_config.num_scheduler_steps > 1:
        raise ValueError(
            "Guided decoding and logits processors are not supported "
            "in multi-step decoding")

    # 如果 arrival_time 没传入，则用当前系统时间戳
    if arrival_time is None:
        arrival_time = time.time()

    # tokenizer 存在时，对 prompt 做合法性检查/校验
    if self.tokenizer is not None:
        self._validate_token_prompt(
            prompt,
            tokenizer=self.get_tokenizer(lora_request=lora_request))

    # 预处理输入（例如把 prompt 结构化、处理多模态输入、处理 adapter/LoRA 等）
    preprocessed_inputs = self.input_preprocessor.preprocess(
        prompt,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
    )

    # 进一步处理输入（例如转换成内部统一格式、编码成 token ids 等）
    processed_inputs = self.input_processor(preprocessed_inputs)

    # 把处理后的请求正式加入调度器队列
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
```



### _add_processed_request()（vllm/engine/llm_engine.py 第578行 ）

```python
def _add_processed_request(
    self,
    request_id: str,
    processed_inputs: ProcessorInputs,
    params: Union[SamplingParams, PoolingParams],
    arrival_time: float,
    lora_request: Optional[LoRARequest],
    prompt_adapter_request: Optional[PromptAdapterRequest],
    trace_headers: Optional[Mapping[str, str]] = None,
    priority: int = 0,
) -> Optional[SequenceGroup]:
    """Add a processed request to the engine's request pool.
    return the created sequence group.
    """
    # 如果是 SamplingParams 且 params.n > 1，说明要并行采样生成多个候选结果
    if isinstance(params, SamplingParams) and params.n > 1:
        # 使用 ParallelSampleSequenceGroup 专门处理这种并行采样请求
        ParallelSampleSequenceGroup.add_request(
            request_id,
            self,
            params,
            processed_inputs=processed_inputs,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )
        # 这里返回 None，因为并行采样的 SequenceGroup 由 ParallelSampleSequenceGroup 内部管理
        return None

    # 校验 processed_inputs 是否符合模型输入要求
    self._validate_model_inputs(processed_inputs, lora_request)

    # ------------------------
    # 创建 Sequence（序列对象）
    # ------------------------

    # KV cache 的 block size（决定缓存如何分块管理）
    block_size = self.cache_config.block_size

    # 生成一个新的 seq_id（全局递增序列 ID）
    seq_id = next(self.seq_counter)

    # 获取 EOS token id（结束符 token），不同 LoRA 可能对应不同 tokenizer
    eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

    # 判断输入是否为 encoder-decoder 架构（例如 T5 / BART）
    if is_encoder_decoder_inputs(processed_inputs):
        # encoder-decoder 模型会拆成 decoder 输入与 encoder 输入两部分
        decoder_inputs = processed_inputs["decoder"]
        encoder_inputs = processed_inputs["encoder"]
    else:
        # 普通 decoder-only 模型只有 decoder_inputs
        decoder_inputs = processed_inputs
        encoder_inputs = None

    # 创建 decoder 侧的 Sequence（主生成序列）
    seq = Sequence(
        seq_id,
        decoder_inputs,
        block_size,
        eos_token_id,
        lora_request,
        prompt_adapter_request
    )

    # 如果存在 encoder 输入，则也创建一个 encoder_seq；否则为 None
    encoder_seq = (
        None if encoder_inputs is None else Sequence(
            seq_id,
            encoder_inputs,
            block_size,
            eos_token_id,
            lora_request,
            prompt_adapter_request
        )
    )

    # ------------------------------------------------
    # 根据 params 类型创建对应的 SequenceGroup
    # ------------------------------------------------

    # 如果是 SamplingParams（文本生成模式）
    if isinstance(params, SamplingParams):
        seq_group = self._create_sequence_group_with_sampling(
            request_id,
            seq,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority
        )

    # 如果是 PoolingParams（embedding/pooling 模式）
    elif isinstance(params, PoolingParams):
        seq_group = self._create_sequence_group_with_pooling(
            request_id,
            seq,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=encoder_seq,
            priority=priority
        )

    # 既不是 SamplingParams 也不是 PoolingParams，则参数非法
    else:
        raise ValueError(
            "Either SamplingParams or PoolingParams must be provided."
        )

    # ------------------------------------------------
    # 把 SequenceGroup 添加到 scheduler 中
    # 选择当前 unfinished seq groups 最少的 scheduler（负载均衡）
    # ------------------------------------------------

    # 计算每个 scheduler 当前未完成的 seq_group 数量，作为调度成本(cost)
    costs = [
        scheduler.get_num_unfinished_seq_groups()
        for scheduler in self.scheduler
    ]

    # 选择 cost 最小（最空闲）的 scheduler
    min_cost_scheduler = self.scheduler[costs.index(min(costs))]

    # 将 seq_group 加入该 scheduler
    min_cost_scheduler.add_seq_group(seq_group)

    # 返回创建好的 seq_group
    return seq_group
```







在 **`_add_processed_request()`**（`vllm/engine/llm_engine.py` 第 578 行）：根据 processed_inputs 和 params 创建 Sequence/SequenceGroup，并选一个 Scheduler 加入。内部用 **`Sequence()`**（`vllm/engine/llm_engine.py` 第 619、622 行）：用 decoder_inputs、block_size、eos_token_id 等构造一条（或 encoder+decoder 两条）Sequence。用 **`_create_sequence_group_with_sampling()`**（`vllm/engine/llm_engine.py` 第 628 行）：把 Sequence 和 SamplingParams 包成 SequenceGroup（或对 pooling 用 `_create_sequence_group_with_pooling()`）。用 **`min_cost_scheduler.add_seq_group(seq_group)`**（`vllm/engine/llm_engine.py` 第 658 行）：把该 SequenceGroup 加入当前负载最小的 Scheduler 的等待队列。

1. 主循环里每次调用 **`step()`**（`vllm/engine/llm_engine.py` 第 1295 行）：执行一轮“调度 → 执行模型 → 后处理”，并返回本步新产生的 RequestOutput 列表。内部先调 **`self.scheduler[virtual_engine].schedule()`**（`vllm/engine/llm_engine.py` 第 1378 行）：从等待/运行中的 SequenceGroup 里选出本步要跑的序列、以及 KV 块的 swap_in/swap_out/copy 等调度结果，得到 `seq_group_metadata_list` 和 `scheduler_outputs`。
2. 再调 **`self.model_executor.execute_model()`**（`vllm/engine/llm_engine.py` 第 1434 行）：根据 `ExecuteModelRequest`（含 seq_group_metadata_list、blocks_to_swap_in/out/copy 等）在 GPU 上跑一次模型，得到本步的 logits/sampler 输出。然后把本步的 outputs、seq_group_metadata_list、scheduler_outputs 等通过 **`ctx.append_output()`**（`vllm/engine/llm_engine.py` 第 1480 行）：塞进当前虚拟引擎的 output_queue，供后处理消费。
3. 接着调 **`_process_model_outputs()`**（`vllm/engine/llm_engine.py` 第 1016、1497 行）：从 output_queue 取出一批输出，按 seq_group 更新序列状态并做 detokenize/stop 检测、生成 RequestOutput。内部用 **`self.output_processor.process_prompt_logprob()`**（`vllm/engine/llm_engine.py` 第 1143 行）：处理 prompt 部分的 logprobs（若有）。用 **`self.output_processor.process_outputs()`**（`vllm/engine/llm_engine.py` 第 1145 行）：对生成步的 token 做 detokenize、stop 检测、logprobs 处理，并更新各 Sequence 的产出与完成状态。
4. 对已结束的 seq_group，用 **`RequestOutputFactory.create()`**（`vllm/engine/llm_engine.py` 第 1159 行）：根据 SequenceGroup 和 seq_id_to_seq_group 拼出对外的 **RequestOutput**（含 prompt、outputs、finished 等）。再 **`ctx.request_outputs.append(request_output)`**（`vllm/engine/llm_engine.py` 第 1164 行）：把本步新完成的 RequestOutput 加入当次 step 的返回列表。最后 **`step()` 返回 `ctx.request_outputs`**（`vllm/engine/llm_engine.py` 第 1522 行）：把本步产生的所有已完成请求的 RequestOutput 返回给调用方。
8. 多进程和异步是在上述单进程流程外再包一层：**MQLLMEngine**（`vllm/engine/multiprocessing/engine.py`）在子进程里创建并持有同一个 **LLMEngine**，在循环里从 ZMQ 收请求、调 `add_request`/`step()`、把结果通过 ZMQ 发回。**MQLLMEngineClient**（`vllm/engine/multiprocessing/client.py`）在父进程里通过 ZMQ 发 RPC 请求、收 RequestOutput 流，相当于远程调用子进程的 `add_request`/`step()`。**AsyncLLMEngine**（`vllm/engine/async_llm_engine.py`）用 asyncio 后台循环不断调同一套 `add_request`/`step()`，并把 `step()` 得到的 RequestOutput 通过 callback/queue 送给各请求的等待端；核心逻辑仍是单进程里的 LLMEngine。
