---
title: vLLM
published: 2026-01-22
description: "vLLM"
image: "./cover.jpeg"
tags: ["Blogging","vLLM"]
category: Guides
draft: false
---

## vLLM

```apl
vllm/ 
├── assets/                 # 多模态资源处理（音频、图像、视频） 

├── attention/              # 注意力机制实现
│   ├── backends/           # 不同后端的注意力实现
│   ├── layers/             # 注意力层
│   ├── ops/                # 注意力操作
│   └── utils/              # 工具函数

├── benchmarks/             # 性能基准测试

├── compilation/            # 编译相关（CUDA graphs, fusion等）

├── config/                 # 配置管理

├── device_allocator/       # 设备内存分配器

├── distributed/            # 分布式相关
│   ├── device_communicators/  # 设备通信
│   ├── ec_transfer/        # EC 传输
│   ├── eplb/               # EP 负载均衡
│   └── kv_transfer/        # KV 缓存传输

├── engine/                 # LLM 引擎核心
│   ├── llm_engine.py       # 主引擎
│   ├── async_llm_engine.py # 异步引擎
│   ├── arg_utils.py        # 参数工具
│   └── protocol.py         # 协议定义

├── entrypoints/            # 入口点
│   ├── anthropic/          # Anthropic API
│   ├── cli/                # 命令行接口
│   ├── openai/             # OpenAI 兼容 API
│   ├── pooling/            # 池化操作
│   ├── sagemaker/          # AWS SageMaker
│   └── serve/              # 服务相关

├── inputs/                 # 输入处理

├── logging_utils/          # 日志工具

├── lora/                   # LoRA 适配器支持
│   ├── layers/             # LoRA 层
│   ├── ops/                # LoRA 操作（Triton, Torch, IPEX, XLA）
│   └── punica_wrapper/     # Punica 包装器

├── model_executor/         # 模型执行器
│   ├── layers/             # 模型层（FLA, MoE, Mamba等）
│   ├── model_loader/       # 模型加载器
│   ├── models/             # 支持的模型实现
│   └── warmup/             # 预热相关

├── multimodal/             # 多模态支持

├── platforms/              # 平台支持（CUDA, ROCM, CPU, TPU, XPU）

├── plugins/                # 插件系统
│   ├── io_processors/      # IO 处理器
│   └── lora_resolvers/     # LoRA 解析器

├── profiler/               # 性能分析工具

├── ray/                    # Ray 集成

├── reasoning/              # 推理相关

├── tokenizers/             # 分词器

├── tool_parsers/           # 工具解析器

├── transformers_utils/      # Transformers 工具
│   ├── chat_templates/     # 聊天模板
│   ├── configs/            # 配置解析
│   └── processors/         # 处理器

├── triton_utils/           # Triton 工具

├── utils/                  # 通用工具

├── usage/                  # 使用示例

└── v1/                     # v1 API（新版本）
    ├── attention/          # v1 注意力
    ├── core/               # v1 核心
    ├── engine/             # v1 引擎
    ├── executor/           # v1 执行器
    ├── kv_offload/         # KV 卸载
    ├── metrics/            # 指标
    ├── sample/             # 采样
    ├── spec_decode/        # 推测解码
    ├── structured_output/  # 结构化输出
    └── worker/             # 工作器
```

