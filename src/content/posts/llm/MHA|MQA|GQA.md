---
title: "MHA|MQA|GQA"
published: 2026-05-10
description: "MHA|MQA|GQA"
image: ""
tags: ["llm","MHA|MQA|GQA"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-10T21:48:34.555.253317149Z"
---

# MHA, MQA, and GQA

MHA (多头注意力), MQA (多查询注意力), and GQA (分组查询注意力) are Attention Mechanism (注意力机制) variants that trade off model quality, inference speed, and KV Cache (键值缓存) memory usage.

The core difference is how many Key (键) and Value (值) heads are shared by Query (查询) heads.
$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 2. MHA

MHA (多头注意力) gives each Query Head (查询头) its own Key Head (键头) and Value Head (值头), so every head learns independent attention patterns.

MHA (多头注意力) is like every interviewer (面试官) having their own notebook (笔记本), so everyone can record detailed information but the total storage cost is high.

```
# MHA: each query head has its own K/V head

num_query_heads = 8
num_kv_heads = 8

print(f"MHA query heads: {num_query_heads}")
print(f"MHA KV heads: {num_kv_heads}")

# Output:
# MHA query heads: 8
# MHA KV heads: 8
```

<br>

## 3. MQA

MQA (多查询注意力) lets all Query Heads (查询头) share one Key Head (键头) and one Value Head (值头), which greatly reduces KV Cache (键值缓存) memory.

### 1) Analogy

MQA (多查询注意力) is like all interviewers (面试官) sharing one common notebook (公共笔记本), so memory cost is very low but some detail may be lost.

```
# MQA: all query heads share one K/V head

num_query_heads = 8
num_kv_heads = 1

print(f"MQA query heads: {num_query_heads}")
print(f"MQA KV heads: {num_kv_heads}")

# Output:
# MQA query heads: 8
# MQA KV heads: 1
```

<br>

## 4. GQA

GQA (分组查询注意力) groups Query Heads (查询头), and each group shares one Key Head (键头) and one Value Head (值头), balancing quality and memory efficiency.

### 1) Analogy

GQA (分组查询注意力) is like interviewers (面试官) being split into teams, where each team shares one notebook (笔记本), so it is cheaper than MHA (多头注意力) and more expressive than MQA (多查询注意力).

```
# GQA: query heads are divided into groups, and each group shares one K/V head

num_query_heads = 8
num_kv_heads = 2
group_size = num_query_heads // num_kv_heads

print(f"GQA query heads: {num_query_heads}")
print(f"GQA KV heads: {num_kv_heads}")
print(f"Each KV head is shared by {group_size} query heads")

# Output:
# GQA query heads: 8
# GQA KV heads: 2
# Each KV head is shared by 4 query heads
```

<br>

## 5. Comparison

MHA (多头注意力) has the best expressiveness (表达能力), MQA (多查询注意力) has the lowest KV Cache (键值缓存) memory cost, and GQA (分组查询注意力) is a practical middle ground.

| Method (方法)        | Query Heads (查询头) | KV Heads (键值头) | KV Cache Memory (键值缓存内存) | Quality (质量) | Speed (速度) |
| -------------------- | -------------------- | ----------------- | ------------------------------ | -------------- | ------------ |
| MHA (多头注意力)     | Many                 | Many              | High                           | High           | Slower       |
| MQA (多查询注意力)   | Many                 | 1                 | Low                            | Lower          | Faster       |
| GQA (分组查询注意力) | Many                 | Few Groups        | Medium                         | Good           | Fast         |

<br>

## 6. KV Cache Memory

KV Cache (键值缓存) memory is mainly proportional to the number of KV Heads (键值头), so reducing KV Heads (键值头) directly reduces inference memory usage.
$$
KV\ Cache\ Memory \propto Layers \times SequenceLength \times KVHeads \times HeadDim
$$

## 7. Simple Selection Rule

Use MHA (多头注意力) when quality is the priority, MQA (多查询注意力) when memory and speed are the priority, and GQA (分组查询注意力) when you want a balanced production choice.

<br><br>
