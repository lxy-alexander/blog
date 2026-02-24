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



# Async LLM Streaming[¶](https://docs.vllm.ai/en/latest/examples/offline_inference/async_llm_streaming/#async-llm-streaming)

## AsyncEngineArgs：配置模型参数

-   使用 **dataclass** 自动生成构造函数等方法
-   **继承 `EngineArgs`**（说明它是在原有引擎参数基础上的扩展）
-   专门给 **异步 vLLM engine** 使用

```
@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    enable_log_requests: bool = False # 开启请求日志记录

```





## enforce_eager=True 

关闭延迟编译(lazy compilation)，加快示例启动速度(speed up startup)。强制 vLLM 使用 **PyTorch eager 模式**，
 不使用 **Torch Compile / CUDA Graph / Triton 等编译优化**。

```
 # Configure sampling parameters for streaming
    sampling_params = SamplingParams(
        max_tokens=100, # 最大生成长度
        temperature=0.8, # 随机性强度
        top_p=0.95, # nucleus sampling 截断
        seed=42,  # For reproducible results
        output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
    )

```







## AsyncLLM.from_engine_args()：创建异步推理引擎

```python
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )
```





## current_platform.pre_register_and_update()

qualified name（全限定名）

```
vllm.platforms.cuda.CudaPlatform
```





## UsageContext 枚举

| 值                  | 含义                                                      |
| :------------------ | :-------------------------------------------------------- |
| ENGINE_CONTEXT      | 直接使用引擎（如 AsyncLLM、LLMEngine），没有上层 API 服务 |
| LLM_CLASS           | 通过 LLM 类使用                                           |
| OPENAI_API_SERVER   | 通过 OpenAI 兼容 API 服务使用                             |
| API_SERVER          | 通过通用 API 服务使用                                     |
| OPENAI_BATCH_RUNNER | 通过 OpenAI Batch Runner 使用                             |
| UNKNOWN_CONTEXT     | 未知场景                                                  |







## RequestOutputKind

| 模式       | 每次返回                  | 类比               |
| :--------- | :------------------------ | :----------------- |
| DELTA      | 只返回新生成的 token/文本 | 只发“这次新打的字” |
| CUMULATIVE | 返回到目前为止的完整输出  | 发“整段话从头到尾” |

假设模型依次生成："Hello" → " world" → "!"

| 迭代 | DELTA 模式 | CUMULATIVE 模式 |
| :--- | :--------- | :-------------- |
| 1    | "Hello"    | "Hello"         |
| 2    | " world"   | "Hello world"   |
| 3    | "!"        | "Hello world!"  |







