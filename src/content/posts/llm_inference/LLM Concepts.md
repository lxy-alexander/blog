---
title: "LLM Concepts"
published: 2026-08-10
description: "LLM Concepts"
image: ""
tags: ["llm_inference","LLM Concepts"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-10T22:10:34.234.410929590Z"
---

## MHA | MQA | GQA

1）**MHA — Multi-Head Attention**

Each Query head has its own corresponding Key and Value heads.

2）**MQA — Multi-Query Attention**

Multiple Query heads share a single Key head and a single Value head.

3）**GQA — Grouped-Query Attention**

Query heads are divided into groups, and the Query heads within each group share the same Key and Value heads.



## Slice Runtime

```python
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm serve repro/kimi-k3-slice \
  --trust-remote-code \
  --skip-tokenizer-init \
  --load-format dummy \
  --dtype bfloat16 \
  --max-model-len 512 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 128 \
  --gpu-memory-utilization 0.1 \
  --enforce-eager \
  --attention-backend TRITON_MLA \
  --spec-method dspark \
  --spec-model repro/kimi-k3-dspark-slice \
  --spec-tokens 3 \
  --port 8000
```





## emplace_back

vector `push_back` copies or moves an existing object into the vector, while `emplace_back` constructs the object directly in place. `push_back` 将已有对象拷贝或移动进 vector，而 `emplace_back` 直接在 vector 尾部原地构造对象。







## 量化

### 分组量化

By splitting the weights into groups and letting each group have its own scale and offset, you can compress large floating-point tensors into lower-bit representations. This reduces the storage and bandwidth requirements, speeding up computation. 

量化是把原始的浮点数映射成较少位数的整数，这本质上是对精度的压缩。当你把一组数字共用一个缩放尺度时，组内的不同值会被“近似”成有限的离散整数。这种近似会带来一些精度损失，尤其是当数值跨度很大或者细微变化很重要时。不过，合理的分组和动态调整缩放尺度，可以在保持大部分精度的同时，显著减少模型的存储与计算量。

量化就是把一把很长的尺子变成一小段一小段的尺子。因为不同区间的数字范围不一样，用同一把尺子量会不精确。分组后，每个小组都用适合自己的尺子，量起来精度就更好。现代量化技术就是尽量让这种分组更智能、更灵活，让模型在减少计算量的同时，仍然保持不错的表现。

常见的量化技术有几类。首先有定点量化，比如用整数来表示原来的浮点数。其次有对称量化和非对称量化，对称量化让数字围绕零对称，而非对称量化可以针对偏移的数值范围。还有像 GPTQ 和 AWQ 这样的感知量化方法，它们在量化时考虑了模型的误差敏感度。此外还有分组量化，也就是我们刚谈到的那种分批共享尺度的方法。不同技术各有权衡，目标都是在压缩的同时尽量保持模型的准确性。





shared_ptr 的核心逻辑——通过引用计数自动管理对象的生命周期，一旦没人再需要那个对象，它就会被自动清理掉！



etcd



minikube



File discovery 是一种简单的服务发现方式，本质上就是用文件来记录服务的位置。这种方式很简单，适合单机或小规模环境，但它只做服务列表，不提供自动调度、监控、容错等能力。

Kubernetes 是一个完整的容器编排系统，除了服务发现，它还能自动调度工作负载、管理资源、监控健康状态，并在故障时自动恢复。简单说，file discovery 是一个小工具，而 Kubernetes 是一个强大的全面平台，用来管理大规模分布式系统。





Kubernetes 在 llm-d 的底层，主要负责资源编排和容器管理。它托管 llm-d 的各个组件（比如 EPP 和模型服务），并且负责调度、扩缩容、服务发现等基础设施层面的工作。简单来说，Kubernetes 是 llm-d 运行的环境管理者，而 llm-d 则负责模型推理请求的路由和调度。
