---
title: "LLM"
published: 2026-05-10
description: "LLM"
image: ""
tags: ["llm","LLM"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-10T21:47:47.895.930639854Z"
---

# LLM

LLM (大语言模型) is a neural network model (神经网络模型) trained on massive text data to understand, generate, and reason with natural language (自然语言).

## 1. Core Idea

LLM (大语言模型) predicts the next token (下一个词元) based on the previous context (上下文), so text generation (文本生成) is essentially repeated next-token prediction (下一个词元预测).
$$
P(x_t \mid x_1,x_2,\ldots,x_{t-1})
$$

## 2. Token

Token (词元) is the basic input unit (基本输入单位) of an LLM (大语言模型), and it can be a word, subword, character, or symbol.

```py
text = "I love AI"

tokens = text.split()

print(tokens)

# Output:
# ['I', 'love', 'AI']
```

<br>

## 3. Embedding

Embedding (嵌入) converts each token (词元) into a dense vector (稠密向量) so the model can process language mathematically.

```py
import numpy as np

tokens = ["I", "love", "AI"]

np.random.seed(0)
embedding_table = {
    "I": np.random.randn(4),
    "love": np.random.randn(4),
    "AI": np.random.randn(4)
}

embeddings = np.array([embedding_table[token] for token in tokens])

print(embeddings)

# Output:
# [[ 1.76405235  0.40015721  0.97873798  2.2408932 ]
#  [ 1.86755799 -0.97727788  0.95008842 -0.15135721]
#  [-0.10321885  0.4105985   0.14404357  1.45427351]]
```

<br>

## 4. Transformer

Transformer (Transformer架构) is the main architecture (架构) behind modern LLMs (大语言模型), and it uses attention mechanism (注意力机制) to model relationships between tokens (词元).

<br>

## 5. Self-Attention

Self-Attention (自注意力机制) lets each token (词元) look at other tokens in the same sequence (序列) to build a context-aware representation (上下文感知表示).
$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

```py
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

Q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0],
              [0.0, 1.0]])
V = np.array([[10.0, 0.0],
              [0.0, 20.0]])

scores = Q @ K.T / np.sqrt(K.shape[-1])
weights = softmax(scores)
output = weights @ V

print(weights)
print(output)

# Output:
# [[0.66976155 0.33023845]]
# [[6.69761549 6.60476903]]
```

<br>

## 6. Feed Forward Network

![image-20260512145149501](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512145149501)

![image-20260512145217415](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260512145217415)

Feed Forward Network (前馈神经网络) processes each token (词元) independently after Self-Attention (自注意力机制), helping the model transform simple features (特征) into more useful representations (表示).

```py
import numpy as np

# Feed Forward Network numeric example
# Input token representation: x = [1, 2]

x = np.array([[1.0, 2.0]])

# First linear layer
W1 = np.array([
    [1.0, 0.5, -1.0],
    [0.5, 1.0,  1.0]
])
b1 = np.array([0.0, 0.0, 0.0])

# Second linear layer
W2 = np.array([
    [1.0],
    [0.5],
    [2.0]
])
b2 = np.array([0.0])

# Step 1: Linear transformation
hidden_before_relu = x @ W1 + b1

# Step 2: ReLU activation
hidden_after_relu = np.maximum(0, hidden_before_relu)

# Step 3: Output projection
output = hidden_after_relu @ W2 + b2

print("Input:")
print(x)

print("Hidden before ReLU:")
print(hidden_before_relu)

print("Hidden after ReLU:")
print(hidden_after_relu)

print("Output:")
print(output)

# Output:
# Input:
# [[1. 2.]]
# Hidden before ReLU:
# [[2.  2.5 1. ]]
# Hidden after ReLU:
# [[2.  2.5 1. ]]
# Output:
# [[5.25]]
```





<br>

## 7. Pretraining

Pretraining (预训练) trains an LLM (大语言模型) on large-scale text data (大规模文本数据) so it learns general language patterns (语言模式).

<br>

## 8. Fine-Tuning

Fine-Tuning (微调) further trains a pretrained model (预训练模型) on task-specific data (任务数据) so it performs better on a target task (目标任务).

<br>

## 9. Inference

Inference (推理) is the process where the trained model (已训练模型) generates output tokens (输出词元) from an input prompt (提示词).

```
vocab = ["cat", "dog", "AI"]
probabilities = [0.1, 0.2, 0.7]

next_token = vocab[probabilities.index(max(probabilities))]

print(next_token)

# Output:
# AI
```

<br>

## 10. Temperature

Temperature (温度系数) controls randomness (随机性) during generation (生成), where lower values make outputs more deterministic (确定性) and higher values make outputs more diverse (多样化).
$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

```
import numpy as np

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    exp_values = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_values / np.sum(exp_values)

logits = np.array([2.0, 1.0, 0.1])

print(softmax_with_temperature(logits, 0.5))
print(softmax_with_temperature(logits, 2.0))

# Output:
# [0.86377712 0.11689952 0.01932336]
# [0.50168776 0.30428901 0.19402324]
```

<br>

## 11. Context Window

Context Window (上下文窗口) is the maximum number of tokens (最大词元数) the model can read at one time.

<br>

## 12. Hallucination

Hallucination (幻觉) means an LLM (大语言模型) generates fluent but incorrect or unsupported information (不可靠信息).

<br>

## 13. RAG

RAG (检索增强生成) combines retrieval (检索) with generation (生成), so the model can answer using external knowledge (外部知识).

<br>

## 14. Main Limitation

LLM (大语言模型) does not truly know facts (事实) like a database (数据库), because it generates likely text based on learned patterns (学习到的模式).

<br> <br>
