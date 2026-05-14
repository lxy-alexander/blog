---
title: "MultiModal Large Models"
published: 2026-05-12
description: "MultiModal Large Models"
image: ""
tags: ["llm","MultiModal Large Models"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-12T23:51:37.211.870003062Z"
---



# Multimodal Large Language Models (MM-LLM) Architecture Evolution

Multimodal Large Language Models（多模态大语言模型）can process text, images, audio, and video in one framework.
The evolution is: **from modular connection → unified tokens → native end-to-end learning.**

<br>

## 1. The Classic Modular Architecture (2023-2024)

![image-20260512213131469](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512213131469)

![image-20260512205500146](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512205500146)

The classic modular architecture connects pretrained unimodal experts with trainable projectors. Most modules are frozen, so it is cheap to train.

**Each piece in the diagram:**

The **Modality Encoder（模态编码器, coral）** is a pretrained expert, such as ViT or CLIP for images and HuBERT for audio. It turns raw input $I_X$ into features $F_X$. The snowflake means frozen weights（权重冻结）.

The **Input Projector（输入投影器, teal）** is the translator. It maps multimodal features into the LLM’s text token space（文本词元空间）, producing $P_X$. Common choices are MLP, cross-attention, and Q-Former（查询变换器）. This is usually trained.

The **LLM Backbone（LLM 骨干, purple）** is the reasoning center. It takes text features $F_T$ and projected features $P_X$, then outputs text and signal tokens $S_X$（信号词元）.

The **Output Projector（输出投影器, teal）** maps signal tokens into conditioning vectors $H_X$（条件向量）for generation.

The **Modality Generator（模态生成器, coral）** is usually a frozen pretrained generator, such as Stable Diffusion, Zeroscope, or AudioLDM.

The **key insight**: only the projectors are trained, around 1% of parameters. So this design is cheap and practical. Examples: LLaVA, BLIP-2, Qwen-VL, MiniGPT-4.

<br>

## 2. Early Fusion Architecture (2024)

![image-20260512205338193](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512205338193)

Early fusion（早期融合）turns every modality into discrete tokens first, then uses one Transformer to process them together.

**Each piece in the diagram:**

The **BPE tokenizer（text, blue）** splits text into subword tokens.

The **VQ-VAE quantizer（image, coral）** maps image patches into a learned codebook（码本）, so images become discrete image tokens.

The **Unified token stream（purple）** is the core idea. Text tokens and image tokens, like `[v1][v7]`, are placed in the same sequence. To the model, they are all token IDs.

The **Single unified transformer（teal）** applies self-attention（自注意力）over the whole sequence. Cross-modal reasoning starts from the first layer. Training uses next-token prediction（下一词元预测）.

The **Output stream** is generated autoregressively. Boundary tokens like `<image_start>` and `<image_end>` mark modality switches.

**Representatives**: Meta Chameleon, Google Gemini, GPT-4o.
**Trade-off**: stronger cross-modal reasoning, but much higher training cost.

<br>

## 3. Native End-to-End Architecture (2026)

![image-20260512205330233](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512205330233)

Native end-to-end models remove even the vision tokenizer. They learn representations directly from raw pixels, text, and audio.

**Each piece in the diagram:**

The **Raw inputs（gray）** go almost directly into the model. There is no separate vision encoder or VAE tokenizer. The idea is to learn from near-lossless input（近无损输入）.

The **Mixture-of-Transformer（MoT, 混合变换器, purple container）** routes different modalities to different experts. Vision tokens use the vision expert, text tokens use the text expert. Self-attention is shared, while feed-forward layers（前馈层）are specialized.

The **Vision expert（coral）** uses **pixel flow matching（像素流匹配）**, learning how to transform noise into pixels. This avoids VQ-VAE information loss.

The **Text expert（teal）** uses autoregressive cross-entropy（自回归交叉熵）, the standard LLM training objective.

The **Audio expert（pink）** applies a similar flow-matching idea to audio spectrograms.

**The big idea**: the model thinks across modalities natively, instead of translating between spaces.

**Representatives**: NEO-unify, Bagel, Emu3.
 **Trade-off**: most powerful, but extremely expensive to train.

<br>

## 4. Three-Generation Comparison

![](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512205126412-20260512205311776-20260512205319668)

The evolution is easy to remember: **modular → early fusion → native end-to-end.**

**Reading the comparison:**

The architecture becomes more unified over time. Modular systems have many separate components. Early fusion uses unified tokens and one Transformer. Native end-to-end removes tokenizers and learns directly from raw modalities.

The trade-off is also clear: the model becomes more integrated, but training cost increases.

**Why all three coexist today:** modular is cheap and common in open-source MM-LLMs. Early fusion is used in frontier products like Gemini and GPT-4o. Native end-to-end is still mainly a research frontier.

<br>
