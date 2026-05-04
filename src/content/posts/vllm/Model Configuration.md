---
title: "Model Configuration"
published: 2026-04-30
description: "Model Configuration"
image: ""
tags: ["vllm","Model Configuration"]
category: vllm
draft: false
lang: ""
createdAt: "2026-04-30T20:45:41.762.433517500Z"
---



# Model Configuration

`create_engine_config()` 作用是：**把 CLI / EngineArgs 里的各种参数，整理成一个最终的 `VllmConfig` 对象**。这个 `VllmConfig` 里面再分成很多子配置：`ModelConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig`、`LoadConfig`、`CompilationConfig` 等





## 1.函数入口

```python
def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
```

-   **UsageContext** (vllm/usage/usage_lib.py)

    `UsageContext` 是 vLLM 用来标记“当前 engine 是从哪里启动/使用的”的枚举。通常用于 usage telemetry / 统计上报 / 日志区分

    ```python
    class UsageContext(str, Enum):
        UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT" # default
        LLM_CLASS = "LLM_CLASS" # Python API LLM()
        API_SERVER = "API_SERVER"  # API serve
        OPENAI_API_SERVER = "OPENAI_API_SERVER" 
        OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
        ENGINE_CONTEXT = "ENGINE_CONTEXT" # Use Engine
    ```

-   **headless**

    headless 只会加载执行模型, 没有 HTTP 服务入口。

    ```python
    Node 0:
      API server + engine
      接收用户 HTTP 请求
      调度请求到各个 engine
    Node 1:
      headless engine
      不接 HTTP 请求
      只加载模型，执行推理
    Node 2:
      headless engine
      不接 HTTP 请求
      只加载模型，执行推理
    ```

    

## 2. 平台和设备初始化

### 1)vLLM 检测当前注册的平台

```
current_platform.pre_register_and_update()
```

访问current_platform触发 `vllm/platforms/__init__.py` 里的 lazy initialization.

```py
1. 先解析 current_platform
2. 如果 current_platform 还没初始化，触发 __getattr__("current_platform")
3. 调用 resolve_current_platform_cls_qualname() # Resolve the qualified name of the current platform class
4. 检测平台：CUDA / ROCm / XPU / CPU / TPU ...
5. 得到平台类名，比如 "vllm.platforms.cuda.CudaPlatform"
6. resolve_obj_by_qualname(...) 找到这个类
7. 实例化：CudaPlatform()
8. 把实例保存到 _current_platform
9. 再调用这个实例的 pre_register_and_update()
```

```
# device should be treated as Device (设备类型), not just a normal string (普通字符串).
device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

# vllm/config/device.py
Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]
```



### 2)检测当前环境变量是否合法

```
envs.validate_environ(self.fail_on_environ_validation)
```

如果是 `True`，环境变量检查失败时直接报错；如果是 `False`，可能只 warning。



## 3.speculator 检测

```
if not is_cloud_storage(self.model):
```

如果模型不是 S3、GCS 这种云存储路径，就尝试用 Hugging Face 方式读取 config。

```
(self.model, self.tokenizer, self.speculative_config) = (
    maybe_override_with_speculators(...)
)
```

这一步检查当前模型是不是 speculative decoding 的 speculator 模型。

传进去的参数：

```
model=self.model
```

当前模型名或路径。

```
tokenizer=self.tokenizer
```

当前 tokenizer 名或路径。

```
revision=self.revision
```

模型 revision。

```
trust_remote_code=self.trust_remote_code
```

是否允许执行远程模型代码。

```
vllm_speculative_config=self.speculative_config
```

用户显式传入的 speculative decoding 配置。

```
hf_token=self.hf_token
```





## 4.创建 `ModelConfig`

```
model_config = self.create_model_config()
```

```
ModelConfig(  # 创建模型配置对象，后续 engine 会基于它加载模型、tokenizer、HF config 等
    model=self.model,  # 模型名或模型路径，比如 HF repo、本地路径、S3/GCS 路径
    model_weights=self.model_weights,  # 单独指定权重路径；不指定时通常和 model 一致
    hf_config_path=self.hf_config_path,  # 单独指定 HuggingFace config 的路径
    runner=self.runner,  # 模型运行模式，例如 generate、pooling、draft 等
    convert=self.convert,  # 是否/如何转换模型结构或任务类型
    tokenizer=self.tokenizer,  # tokenizer 路径或名称；不指定时通常默认用 model
    tokenizer_mode=self.tokenizer_mode,  # tokenizer 加载模式，例如 auto、slow、mistral 等
    trust_remote_code=self.trust_remote_code,  # 是否允许执行 HF repo 里的自定义 Python 代码
    allowed_local_media_path=self.allowed_local_media_path,  # 多模态输入允许读取的本地媒体路径
    allowed_media_domains=self.allowed_media_domains,  # 多模态 URL 输入允许访问的域名白名单
    dtype=self.dtype,  # 模型权重/计算 dtype，例如 auto、float16、bfloat16
    seed=self.seed,  # 随机种子，用于可复现性
    revision=self.revision,  # HF 模型 revision，比如 branch、tag、commit hash
    code_revision=self.code_revision,  # HF 自定义代码的 revision，可和权重 revision 分开
    hf_token=self.hf_token,  # 访问私有 HF repo 的 token
    hf_overrides=self.hf_overrides,  # 覆盖 HF config 里的字段
    tokenizer_revision=self.tokenizer_revision,  # tokenizer 使用的 HF revision
    max_model_len=self.max_model_len,  # 模型最大上下文长度
    quantization=self.quantization,  # 量化方式，例如 awq、gptq、fp8 等
    quantization_config=self.quantization_config,  # 额外的量化配置
    allow_deprecated_quantization=self.allow_deprecated_quantization,  # 是否允许已废弃的量化格式
    enforce_eager=self.enforce_eager,  # 是否强制 eager 执行，禁用 CUDA graph 等优化
    enable_return_routed_experts=self.enable_return_routed_experts,  # MoE 模型是否返回 routed experts 信息
    max_logprobs=self.max_logprobs,  # 最多允许返回多少个 logprobs
    logprobs_mode=self.logprobs_mode,  # logprobs 的计算/返回模式
    disable_sliding_window=self.disable_sliding_window,  # 是否禁用 sliding window attention
    disable_cascade_attn=self.disable_cascade_attn,  # 是否禁用 cascade attention 优化
    skip_tokenizer_init=self.skip_tokenizer_init,  # 是否跳过 tokenizer 初始化
    enable_prompt_embeds=self.enable_prompt_embeds,  # 是否允许直接传 prompt embeddings
    served_model_name=self.served_model_name,  # 对外服务时暴露的模型名
    language_model_only=self.language_model_only,  # 是否只加载语言模型部分，忽略多模态组件
    limit_mm_per_prompt=self.limit_mm_per_prompt,  # 每个 prompt 允许的多模态输入数量限制
    enable_mm_embeds=self.enable_mm_embeds,  # 是否允许多模态 embeddings 输入
    interleave_mm_strings=self.interleave_mm_strings,  # 是否把多模态占位符和文本字符串交错处理
    media_io_kwargs=self.media_io_kwargs,  # 媒体读取/解码相关额外参数
    skip_mm_profiling=self.skip_mm_profiling,  # 是否跳过多模态 profiling
    config_format=self.config_format,  # config 格式，例如 auto、hf、mistral 等
    mm_processor_kwargs=self.mm_processor_kwargs,  # 多模态 processor 的额外参数
    mm_processor_cache_gb=self.mm_processor_cache_gb,  # 多模态 processor 缓存大小，单位 GB
    mm_processor_cache_type=self.mm_processor_cache_type,  # 多模态 processor 缓存类型
    mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,  # 共享内存缓存单对象大小上限
    mm_encoder_only=self.mm_encoder_only,  # 是否只使用多模态 encoder
    mm_encoder_tp_mode=self.mm_encoder_tp_mode,  # 多模态 encoder 的 tensor parallel 模式
    mm_encoder_attn_backend=self.mm_encoder_attn_backend,  # 多模态 encoder attention 后端
    pooler_config=self.pooler_config,  # pooling 模型相关配置
    generation_config=self.generation_config,  # generation config 来源或路径
    override_generation_config=self.override_generation_config,  # 覆盖 generation config 的字段
    enable_sleep_mode=self.enable_sleep_mode,  # 是否启用 sleep mode，用于释放/恢复 GPU 内存
    model_impl=self.model_impl,  # 使用哪个模型实现，例如 auto、vllm、transformers
    override_attention_dtype=self.override_attention_dtype,  # 覆盖 attention 使用的 dtype
    logits_processors=self.logits_processors,  # 自定义 logits processor 列表
    video_pruning_rate=self.video_pruning_rate,  # 视频输入的 pruning 比例
    mm_tensor_ipc=self.mm_tensor_ipc,  # 是否使用 IPC 传递多模态 tensor
    io_processor_plugin=self.io_processor_plugin,  # 自定义 IO processor 插件
    renderer_num_workers=self.renderer_num_workers,  # renderer 使用的 worker 数量
)
```





## 5.Function supports checking and setting default parameters

### 1)`_check_feature_supported` 

#### Partial Prefill

Partial Prefill is an optimization that splits a long prompt into smaller chunks during the prefill stage, instead of processing the entire prompt at once.

Without Partial Prefill:

Request A: long prompt prefill ─────────────────────
Request B: short request decode                      waiting
Request C: short request decode                      waiting

A long prompt may occupy the engine for a long time during prefill.
Other requests have to wait until the full prefill finishes.


With Partial Prefill:

Request A: prefill chunk 1 ── prefill chunk 2 ── prefill chunk 3 ──
Request B:                  decode ─ decode ─ decode
Request C:                           decode ─ decode

By splitting the long prefill into smaller chunks, the scheduler can run other requests between chunks. This reduces blocking and improves latency.



#### Pipeline Parallelism

Pipeline Parallelism is a model parallelism technique that splits different layers of a model across multiple devices.

For example:

GPU 0: Layer 1 ─ Layer 2
GPU 1: Layer 3 ─ Layer 4
GPU 2: Layer 5 ─ Layer 6

Requests then move through these devices like a pipeline:

Request A: GPU 0 ── GPU 1 ── GPU 2
Request B:          GPU 0 ── GPU 1 ── GPU 2
Request C:                   GPU 0 ── GPU 1 ── GPU 2

This helps large models fit across multiple devices and allows different pipeline stages to work in parallel.



```
_check_feature_supported
├── Purpose
│   └── Check early whether the features enabled in EngineArgs are supported
│       └── If unsupported, raise via _raise_unsupported_error(...)
│
├── Check 1: Concurrent Partial Prefill
│   ├── Condition
│   │   ├── self.max_num_partial_prefills != default value
│   │   └── or self.max_long_partial_prefills != default value
│   ├── Meaning
│   │   └── The user is trying to enable/change concurrent partial prefill
│   └── Result
│       └── Currently unsupported, so it raises an error
│
└── Check 2: Pipeline Parallelism
    ├── Trigger
    │   └── self.pipeline_parallel_size > 1
    ├── Meaning
    │   └── The user enabled pipeline parallelism across multiple workers/GPUs
    ├── supports_pp
    │   └── Reads supports_pp from distributed_executor_backend
    │       └── Defaults to False if the attribute does not exist
    ├── Allowed backends
    │   ├── default distributed_executor_backend
    │   ├── "ray"
    │   ├── "mp"
    │   └── "external_launcher"
    └── Error condition
        └── The backend does not declare supports_pp and is not in the allowed list
```



### `_set_default_chunked_prefill_and_prefix_caching_args`

#### prefix_caching

Prefix Caching is an optimization that reuses the KV cache of previously processed prompt prefixes.

For example, if multiple requests share the same beginning:

Request A: System prompt + User question A
Request B: System prompt + User question B
Request C: System prompt + User question C

Without Prefix Caching:

Request A: prefill System prompt ── prefill question A
Request B: prefill System prompt ── prefill question B
Request C: prefill System prompt ── prefill question C

The shared system prompt is processed repeatedly.


With Prefix Caching:

Request A: prefill System prompt ── prefill question A
Request B: reuse cached prefix ──── prefill question B
Request C: reuse cached prefix ──── prefill question C

The shared prefix is reused, reducing repeated computation and improving throughput.



```
_set_default_chunked_prefill_and_prefix_caching_args
├── Purpose
│   └── Set default values for chunked prefill and prefix caching
│       └── Warn if the user manually chooses an unsupported/risky setting
│
├── Read model defaults
│   ├── default_chunked_prefill
│   │   └── model_config.is_chunked_prefill_supported
│   └── default_prefix_caching
│       └── model_config.is_prefix_caching_supported
│
├── Chunked prefill
│   ├── If self.enable_chunked_prefill is None
│   │   └── User did not explicitly set it
│   │       └── Use default_chunked_prefill
│   ├── Else if generate model + user disables it + model supports it
│   │   └── Warn: disabling chunked prefill may crash or produce wrong outputs
│   └── Else if pooling model + user enables it + model does not support it
│       └── Warn: enabling chunked prefill may crash or produce wrong outputs
│
├── Prefix caching
│   ├── If self.enable_prefix_caching is None
│   │   └── User did not explicitly set it
│   │       └── Use default_prefix_caching
│   └── Else if pooling model + user enables it + model does not support it
│       └── Warn: enabling prefix caching may crash or produce wrong outputs
│
└── RISC-V CPU special case (RISC-Five)
    ├── If current platform is CPU and architecture is RISC-V
    │   ├── Force self.enable_chunked_prefill = False
    │   └── Force self.enable_prefix_caching = False
    └── Reason
        └── V1 backend does not support these features on RISC-V CPUs
```



### `_set_default_reasoning_config_args`

 if the user passed `--reasoning-parser`, this function makes sure `self.reasoning_config` exists and stores that parser inside it.

```
_set_default_reasoning_config_args
├── Purpose
│   └── Sync the legacy/simple reasoning_parser argument into reasoning_config
│
├── If self.reasoning_parser is empty
│   └── Return immediately
│       └── No reasoning parser was requested
│
├── If self.reasoning_config is None
│   └── Create a default ReasoningConfig()
│       └── Ensures there is a config object to write into
│
└── Set parser
    └── self.reasoning_config.reasoning_parser = self.reasoning_parser
        └── Copies the user-provided parser name into the structured config
        
```





## 6.sliding window/KV cache dtype/prefix cache

```
create_engine_config
├── Handle sliding-window cache config
│   ├── Start with sliding_window = None
│   ├── If the model is not interleaved
│   │   └── Read one global sliding_window value from model_config
│   └── If the model is interleaved
│       └── Keep sliding_window = None
│           └── Avoid overriding per-layer/global layer settings inside the model
│
├── Resolve KV cache dtype
│   └── resolved_cache_dtype = resolve_kv_cache_dtype_string(...)
│       └── Converts "auto" into the actual KV cache dtype using model_config
│
└── Validate prefix caching default was already resolved
    └── assert self.enable_prefix_caching is not None
        └── At this point it must be True or False, not unset
```

