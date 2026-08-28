---
title: "vllm Speculative decoding"
published: 2026-08-28
description: "vllm Speculative decoding"
image: ""
tags: ["llm_inference","vllm Speculative decoding"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-28T22:02:18.856.599912040Z"
---

对，但要区分“有没有独立 drafter”以及“这个 drafter 是不是一个完整的小 LLM”。

1）**n-gram：没有独立 draft model**

```text
target model
    +
历史 token 匹配算法
```

它只是从已有 token 中找重复片段来猜后续 token，所以没有额外权重，也不需要加载另一个模型。

2）**draft model：有一个独立的小模型**

这是最典型的“双模型” speculative decoding：

```text
Target model: Qwen3-8B
Draft model:  Qwen3-0.6B
```

例如 vLLM 官方示例就是：

```python
model="Qwen/Qwen3-8B"

speculative_config={
    "model": "Qwen/Qwen3-0.6B",
    "method": "draft_model",
}
```

这个 `Qwen3-0.6B` 是真正独立加载的一套模型权重。([GitHub](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/draft_model.md?utm_source=chatgpt.com))

3）**EAGLE：通常有独立的 drafter 权重，但不是一个普通小 LLM**

例如：

```text
Target:
Llama-3-8B
       │
       ├── hidden states
       ↓
EAGLE drafter
       ↓
draft tokens
```

EAGLE drafter 会利用 target model 的 hidden states，因此它和 target model 是配套训练的。

比如官方示例：

```python
model="meta-llama/Meta-Llama-3-8B-Instruct"

speculative_config={
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "method": "eagle",
}
```

这里 `EAGLE-LLaMA3-Instruct-8B` 就是额外的 drafter 权重。([GitHub](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/eagle.md?utm_source=chatgpt.com))

所以 EAGLE 更像：

```text
主模型 + 专门训练的轻量 speculative head/model
```

而不是：

```text
8B target + 另一个普通 1B causal LM
```

4）**MTP：通常不是额外指定一个独立模型，而是 target model 自带的 MTP 模块**

这是它和 EAGLE/draft model 很重要的区别。

比如：

```text
Qwen3.5
├── 主 transformer
├── LM head
└── MTP layers   ← 模型 checkpoint 自己带
```

所以配置时可以只写：

```bash
--speculative-config \
'{"method":"mtp","num_speculative_tokens":7}'
```

你没有额外写：

```text
"model": "xxx-drafter"
```

因为 vLLM 可以从 target model 自己解析 MTP 模块。当前相关文档也明确区分了：EAGLE、draft model 等通常需要指定 `model`，而 `mtp` 会复用/从 target model 自动解析。([GitHub](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/speculative_decoding.md?utm_source=chatgpt.com))

所以你可以这样记：

| 方法        | 有没有额外权重       | 是不是独立普通 LLM           |
| ----------- | -------------------- | ---------------------------- |
| n-gram      | 没有                 | 不是                         |
| draft model | 有                   | **通常是**                   |
| EAGLE       | 有                   | 不是普通 LLM，是专门 drafter |
| MTP         | 通常 checkpoint 自带 | 通常不是独立模型             |

最关键的区别可以画成：

```text
n-gram
Target
  +
token matching


Draft Model
Target        Small LLM
  │              │
  │          draft tokens
  │              │
  └──── verify ──┘


EAGLE
Target
  │
hidden states
  │
  ↓
EAGLE drafter（额外权重）
  │
draft tokens
  ↓
Target verify


MTP
Target checkpoint
├── normal layers
└── MTP layers
       │
       ↓
   draft tokens
       │
       ↓
 Target verify
```

所以如果你问“**MTP 的 drafter 是不是像 EAGLE 一样单独下载一个专属模型？**”，通常答案是：**不是，MTP 往往就是主模型 checkpoint 自带的专用预测层；EAGLE 则通常有独立的、与 target 配套的 drafter checkpoint。**
