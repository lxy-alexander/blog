---
title: "Offline Inference"
published: 2026-03-10
description: "Offline Inference"
image: ""
tags: ["vllm","Offline Inference"]
category: vllm
draft: false
lang: ""
---



------

**1. Async LLM Streaming** 异步流式推理。用 `asyncio` 非阻塞地逐 token 输出结果，适合 Web 服务实时返回响应，避免等待整个序列生成完才返回。

**2. Audio Language** 音频+语言多模态推理。将音频输入（语音、声音）与文本结合送入模型，支持语音问答等场景。

**3. Automatic Prefix Caching (APC)** 自动前缀缓存。对相同前缀的请求复用已计算的 KV Cache，避免重复计算 system prompt，显著降低首 token 延迟。

**4. Batch LLM Inference** 离线批量推理。一次性处理大量请求，最大化 GPU 吞吐，适合数据处理、评估等非实时场景。

**5. Chat With Tools** 工具调用推理。模型在对话中调用外部函数/API（Function Calling），实现搜索、计算等能力扩展。

**6. Context Extension** 上下文窗口扩展。通过 RoPE 缩放等技术让模型处理超出训练长度的输入，支持长文档场景。

**7. Data Parallel** 数据并行推理。多个 GPU/进程各自持有完整模型副本，并行处理不同请求，横向扩展吞吐量。

**8. Disaggregated Prefill V1** 预填充分离 V1 版。将计算密集的 prefill 阶段和内存密集的 decode 阶段分配到不同实例，各自优化资源利用。

**9. Disaggregated Prefill** 预填充分离基础版。同上，是早期/基础实现版本，概念相同但架构实现有差异。

**10. Encoder Decoder Multimodal** 编码器-解码器多模态。使用 encoder-decoder 架构（如 T5、Whisper 类结构）处理多模态任务，与纯 decoder 架构不同。

**11. Extract Hidden States** 提取隐藏层状态。获取模型中间层的向量表示，用于 embedding、分类、RAG 检索等下游任务。

**12. KV Load Failure Recovery Test** KV Cache 加载失败恢复测试。验证当 KV Cache 读取失败时系统能否优雅降级、自动重算，保证服务稳定性。

**13. LLM Engine Example** LLM 引擎基础示例。直接使用 vLLM 底层 `LLMEngine` API，展示最基本的请求提交与结果获取流程。

**14. LLM Engine Reset KV** 引擎运行时重置 KV Cache。演示如何在不重启服务的情况下清空缓存，用于状态管理和内存回收。

**15. Load Sharded State** 加载分片模型权重。将大模型按层/张量切分成多个文件分片加载，解决单文件过大或显存不足问题。

**16. Custom Logits Processors** 自定义 logits 处理器。在采样前对模型输出的原始分数进行自定义修改，实现关键词屏蔽、格式约束等控制。

**17. LoRA With Quantization Inference** LoRA + 量化联合推理。在量化（INT4/INT8）的基础模型上叠加 LoRA 适配器，兼顾显存节省和个性化能力。

**18. Metrics** 推理性能指标监控。暴露 throughput、latency、TTFT（首 token 时间）等指标，对接 Prometheus/Grafana 用于生产监控。

**19. Mistral-Small** Mistral Small 模型推理示例。展示如何加载和运行 Mistral 系列小参数量模型，适合资源受限场景。

**20. MLPSpeculator** MLP 投机解码器。用一个轻量 MLP 网络预测后续多个 token，再由主模型并行验证，加速生成速度 2~3 倍。

**21. MultiLoRA Inference** 多 LoRA 并发推理。单个 vLLM 实例同时加载多个 LoRA 适配器，按请求动态切换，服务多租户场景。

**22. New Weight Syncing** 新权重同步机制。在训练-推理协同场景（如在线 RLHF）中，将最新训练好的权重热更新到推理引擎，无需重启。

**23. Offline Inference with OpenAI Batch File Format** 兼容 OpenAI Batch API 格式的离线推理。读取 `.jsonl` 格式批量请求文件，输出兼容 OpenAI 格式的结果，便于迁移。

**24. Pause Resume** 推理暂停与恢复。演示如何中断正在进行的推理任务并在之后恢复，用于资源调度和优先级抢占场景。

**25. Prefix Caching** 手动前缀缓存示例。与 APC 对应，展示显式控制缓存行为的方式，更灵活地管理共享前缀的缓存策略。

**26. Prompt Embed Inference** 直接输入 embedding 推理。跳过 tokenizer，直接传入预计算的向量作为模型输入，用于特殊的 embedding 注入场景。

**27. Qwen2.5-Omni Offline Inference** Qwen2.5-Omni 离线推理示例。展示阿里 Qwen 全模态模型（文本+图像+音频+视频）的本地批量推理用法。

**28. Qwen3 Omni** Qwen3 全模态推理示例。Qwen3 版本的多模态推理，支持更强的模态理解和更长上下文。

**29. Qwen 1M** Qwen 百万上下文推理。演示如何运行支持 100 万 token 超长上下文的 Qwen 模型，处理超长文档。

**30. Reproducibility** 推理结果可复现性。通过固定随机种子、采样参数等手段，确保相同输入每次产生相同输出，用于测试和调试。

**31. RLHF** 基于人类反馈的强化学习训练示例。展示 vLLM 作为 rollout 引擎参与 PPO 等 RLHF 训练流程的基本用法。

**32. RLHF Colocate** 训练推理共置的 RLHF。将训练器和推理引擎放在同一组 GPU 上交替运行，节省跨机通信开销和硬件成本。

**33. RLHF Online Quant** 在线量化 RLHF。训练过程中动态对模型量化后推理，降低 rollout 阶段显存占用，提升训练效率。

**34. RLHF Utils** RLHF 工具函数库。提供 reward 计算、数据格式转换、采样策略等 RLHF 流程中复用的通用工具。

**35. Run One Batch** 单批次运行示例。最简化的批量推理演示，用于快速验证环境配置和模型加载是否正常。

**36. Save Sharded State** 保存分片模型权重。将运行中的模型权重按分片格式导出保存，配合 Load Sharded State 使用，加速下次启动。

**37. Simple Profiling** 简单性能分析。使用 PyTorch Profiler 等工具记录推理各阶段耗时，定位性能瓶颈。

**38. Skip Loading Weights In Engine Init** 初始化时跳过权重加载。在测试或调试引擎逻辑时跳过耗时的权重加载步骤，加快启动速度。

**39. Spec Decode** 投机解码通用示例。用 draft 模型生成候选 token，主模型并行验证，显著提升生成吞吐，适合延迟敏感场景。

**40. Structured Outputs** 结构化输出约束。强制模型输出符合 JSON Schema 或正则表达式的格式，保证下游程序可靠解析。

**41. Torchrun DP Example** torchrun 数据并行示例。用 `torchrun` 启动多进程数据并行推理，展示标准 PyTorch 分布式启动方式。

**42. Torchrun Example** torchrun 基础示例。展示用 `torchrun` 启动 vLLM 的最基本分布式配置，不限于数据并行。

**43. Vision Language** 视觉语言单图推理。将单张图片与文本一起输入模型，支持图像问答、图像描述等 VQA 任务。

**44. Vision Language Multi Image** 视觉语言多图推理。支持在单次对话中传入多张图片，处理图像对比、多帧理解等更复杂的视觉任务。
