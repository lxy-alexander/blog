---
title: "RLHF"
published: 2026-03-10
description: "RLHF"
image: ""
tags: ["interview","llm","RLHF"]
category: interview / llm
draft: false
lang: ""
---



## RLHF 是什么？

RLHF 全称是 **基于人类反馈的强化学习**，是让大模型对齐人类偏好的核心技术。

简单说就是三步：

1.  **SFT**：用人工写的高质量问答让模型学会基本对话
2.  **RM**：让人类对模型生成的多个回答进行排序，训练一个能够自动判别回答质量的打分器
3.  **RL (PPO)**：用这个打分器指导模型优化，同时用 KL 惩罚防止模型跑偏





## **SFT 会导致知识遗忘吗？**

-   会存在**灾难性遗忘（Catastrophic Forgetting）**风险。因此在微调时，通常会保留一小部分预训练数据进行混合训练，以维持模型的通用能力。

## **SFT 和 RLHF 的界限在哪里？**

-   SFT 是给模型定**下限**（确保它会说话、懂格式）；RLHF 是拔高模型的**上限**（让它说话更得体、更安全、更聪明）。
