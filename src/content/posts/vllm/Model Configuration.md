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

```py

```



```
device_config = DeviceConfig(device=cast(Device, current_platform.device_type))
```



