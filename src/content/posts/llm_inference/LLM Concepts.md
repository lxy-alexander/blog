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





