---
title: "TP"
published: 2026-06-05
description: "TP"
image: ""
tags: ["llm","TP"]
category: llm
draft: false
lang: ""
createdAt: "2026-06-05T23:42:53.365.303398240Z"
---

# Tensor Parallelism

==Tensor Parallelism (张量并行) splits tensors inside each Transformer layer so multiple GPUs jointly run one model replica.==

<br>

## 1. Core Idea

TP does not split a model by layers. It splits tensors inside each layer.

Wrong idea:

```text
GPU0 runs layer 0
GPU1 runs layer 1
GPU2 runs layer 2
```

Correct idea:

```text
GPU0, GPU1, GPU2, GPU3 all run the same layer,
but each GPU owns different weight slices.
```

Example:

```python
# This example shows TP rank ownership.
# It can run independently.

tp_size = 4

for rank in range(tp_size):
    print(f"TP rank {rank}: owns 1/{tp_size} tensor slice")

# Output:
# TP rank 0: owns 1/4 tensor slice
# TP rank 1: owns 1/4 tensor slice
# TP rank 2: owns 1/4 tensor slice
# TP rank 3: owns 1/4 tensor slice
```

<br>

## 2. TP作用点总览

TP mainly affects these parts:

| Area              | Modules                      | TP Role                    |
| ----------------- | ---------------------------- | -------------------------- |
| Attention         | `q_proj`, `k_proj`, `v_proj` | Column Parallel            |
| Attention output  | `o_proj`                     | Row Parallel + all-reduce  |
| Attention heads   | Q/K/V heads                  | Split heads across GPUs    |
| Position encoding | Rotary embedding             | Local per TP rank          |
| KV Cache          | K cache, V cache             | Split by KV heads          |
| MLP               | `gate_proj`, `up_proj`       | Column Parallel            |
| MLP output        | `down_proj`                  | Row Parallel + all-reduce  |
| Embedding         | token embedding              | May use vocab parallel     |
| Output head       | `lm_head`                    | May use vocab parallel     |
| Sampling          | logits, top-k, top-p, argmax | Needs global communication |

<br>

## 3. Linear Layer Basics

A linear layer computes:

$$ Y = XW $$

Where:

```text
X: input activation
W: weight matrix
Y: output activation
```

TP splits `W` across GPUs.

There are two common TP linear patterns:

```text
Column Parallel Linear:
  split W by columns

Row Parallel Linear:
  split W by rows
```

<br>

## 4. Column Parallel Linear

Column Parallel Linear (列并行线性层) splits the output dimension.

Full weight:

```text
W: [input_dim, output_dim]
```

With TP = 2:

```text
W = [ W0 | W1 ]
```

Each GPU receives full `X`:

```text
GPU0: Y0 = X @ W0
GPU1: Y1 = X @ W1
```

Then:

```text
Y = concat(Y0, Y1)
```

Shape example:

```text
X:  [batch, 4096]
W:  [4096, 8192]

GPU0: W0 [4096, 4096]
GPU1: W1 [4096, 4096]

GPU0: Y0 [batch, 4096]
GPU1: Y1 [batch, 4096]

Y: [batch, 8192]
```

Used by:

```text
q_proj
k_proj
v_proj
gate_proj
up_proj
some lm_head cases
```

Example:

```python
# This example simulates Column Parallel Linear.
# It can run independently.

import numpy as np

X = np.array([[1, 2]])

W = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
])

Y_full = X @ W

W0 = W[:, :2]
W1 = W[:, 2:]

Y0 = X @ W0
Y1 = X @ W1

Y_tp = np.concatenate([Y0, Y1], axis=-1)

print("Y_full =", Y_full)
print("Y_tp   =", Y_tp)

# Output:
# Y_full = [[11 14 17 20]]
# Y_tp   = [[11 14 17 20]]
```

Slice meaning:

```python
W0 = W[:, :2]
Take all rows and the first two columns.
W1 = W[:, 2:]
Take all rows and columns from index 2 to the end.
```

<br>

## 5. Row Parallel Linear

Row Parallel Linear (行并行线性层) splits the input dimension.

Full weight:

```text
W: [input_dim, output_dim]
```

With TP = 2:

```text
    [ W0 ]
W = [ W1 ]
```

Input `X` is also split:

```text
X = [ X0 | X1 ]
```

Each GPU computes a partial result:

```text
GPU0: P0 = X0 @ W0
GPU1: P1 = X1 @ W1
```

Then all partial results are summed:

```text
Y = P0 + P1
```

This sum is all-reduce (全规约).

Shape example:

```text
X: [batch, 8192]
W: [8192, 4096]

GPU0:
  X0 [batch, 4096]
  W0 [4096, 4096]
  P0 [batch, 4096]

GPU1:
  X1 [batch, 4096]
  W1 [4096, 4096]
  P1 [batch, 4096]

Y = P0 + P1
```

Used by:

```text
o_proj
down_proj
```

Note:

```text
In a real layer, X is already sharded
because it comes from a previous Column Parallel output.
So no extra split of X is needed.
```

Example:

```python
# This example simulates Row Parallel Linear.
# It can run independently.

import numpy as np

X = np.array([[1, 2]])

W = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
])

Y_full = X @ W

X0 = X[:, :1]
X1 = X[:, 1:]

W0 = W[:1, :]
W1 = W[1:, :]

P0 = X0 @ W0
P1 = X1 @ W1

Y_tp = P0 + P1

print("P0 =", P0)
print("P1 =", P1)
print("Y_full =", Y_full)
print("Y_tp   =", Y_tp)

# Output:
# P0 = [[1 2 3 4]]
# P1 = [[10 12 14 16]]
# Y_full = [[11 14 17 20]]
# Y_tp   = [[11 14 17 20]]
```

<br>

## 6. TP in Attention

Attention is split mainly by heads.

Assume:

```text
hidden_size = 4096
num_heads = 32
head_dim = 128
TP = 4
```

Then:

```text
heads_per_rank = 32 / 4 = 8
```

Head ownership:

```text
GPU0: heads 0-7
GPU1: heads 8-15
GPU2: heads 16-23
GPU3: heads 24-31
```

Attention TP mapping:

```text
q_proj:
  Column Parallel

k_proj:
  Column Parallel

v_proj:
  Column Parallel

rotary embedding:
  local per rank

paged attention:
  local per rank

o_proj:
  Row Parallel + all-reduce
```

Example:

```python
# This example shows attention head assignment.
# It can run independently.

num_heads = 32
tp_size = 4

heads_per_rank = num_heads // tp_size

for rank in range(tp_size):
    start = rank * heads_per_rank
    end = start + heads_per_rank - 1
    print(f"GPU{rank}: heads {start}-{end}")

# Output:
# GPU0: heads 0-7
# GPU1: heads 8-15
# GPU2: heads 16-23
# GPU3: heads 24-31
```

<br>

## 7. TP in q_proj, k_proj, v_proj

`q_proj`, `k_proj`, and `v_proj` usually use Column Parallel.

Full weight example:

```text
Wq: [4096, 4096]
TP = 4
```

Split by columns:

```text
GPU0: Wq_0 [4096, 1024]
GPU1: Wq_1 [4096, 1024]
GPU2: Wq_2 [4096, 1024]
GPU3: Wq_3 [4096, 1024]
```

Each GPU computes local heads:

```text
GPU0: Q0 / K0 / V0 -> heads 0-7
GPU1: Q1 / K1 / V1 -> heads 8-15
GPU2: Q2 / K2 / V2 -> heads 16-23
GPU3: Q3 / K3 / V3 -> heads 24-31
```

Why Column Parallel works here:

```text
Each attention head can be computed independently before o_proj.
```

<br>

## 8. TP in Rotary Embedding

Rotary embedding (旋转位置编码) applies position information to Q and K.

Since each TP rank owns local Q/K heads, rotary embedding is applied locally.

```text
GPU0: apply rotary to local Q/K heads 0-7
GPU1: apply rotary to local Q/K heads 8-15
GPU2: apply rotary to local Q/K heads 16-23
GPU3: apply rotary to local Q/K heads 24-31
```

No full Q/K gather is needed before rotary embedding.

<br>

## 9. TP in Paged Attention

Paged attention (分页注意力) reads and writes KV cache in blocks.

With TP, each rank only handles local heads and local KV cache.

During prefill:

```text
local K/V are written into local KV cache blocks
```

During decode:

```text
local Q reads local K/V cache
local attention output is computed
```

Flow:

```text
local Q
  -> local K cache
  -> local V cache
  -> local attention output
  -> o_proj
```

TP affects paged attention through:

```text
head ownership
KV cache ownership
KV block allocation
prefill attention
decode attention
GPU memory usage
```

<br>

## 10. TP in o_proj

`o_proj` combines local attention outputs and projects them back to hidden size.

Before `o_proj`:

```text
GPU0: A0 from heads 0-7
GPU1: A1 from heads 8-15
GPU2: A2 from heads 16-23
GPU3: A3 from heads 24-31
```

Logically:

```text
A = [ A0 | A1 | A2 | A3 ]
```

`o_proj` uses Row Parallel:

```text
GPU0: P0 = A0 @ Wo_0
GPU1: P1 = A1 @ Wo_1
GPU2: P2 = A2 @ Wo_2
GPU3: P3 = A3 @ Wo_3
```

Final output:

```text
Y = P0 + P1 + P2 + P3
```

This requires all-reduce.

Example:

```python
# This example simulates o_proj Row Parallel.
# It can run independently.

import numpy as np

A = np.array([[1, 2, 3, 4]])

Wo = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 1],
])

Y_full = A @ Wo

A0 = A[:, :2]
A1 = A[:, 2:]

Wo0 = Wo[:2, :]
Wo1 = Wo[2:, :]

P0 = A0 @ Wo0
P1 = A1 @ Wo1

Y_tp = P0 + P1

print("Y_full =", Y_full)
print("Y_tp   =", Y_tp)

# Output:
# Y_full = [[12  9]]
# Y_tp   = [[12  9]]
```

<br>

## 11. TP in MLP

LLaMA-style MLP contains:

```text
gate_proj
up_proj
activation
elementwise multiply
down_proj
```

Computation:

```text
gate = X @ W_gate
up   = X @ W_up
mid  = silu(gate) * up
out  = mid @ W_down
```

TP mapping:

```text
gate_proj:
  Column Parallel

up_proj:
  Column Parallel

activation:
  local

elementwise multiply:
  local

down_proj:
  Row Parallel + all-reduce
```

Shape example:

```text
hidden_size = 4096
intermediate_size = 11008
TP = 4
```

Then:

```text
gate_proj:
  full weight: [4096, 11008]
  each rank:  [4096, 2752]

up_proj:
  full weight: [4096, 11008]
  each rank:  [4096, 2752]

down_proj:
  full weight: [11008, 4096]
  each rank:  [2752, 4096]
```

Flow:

```text
X
  -> local gate_i
  -> local up_i
  -> local mid_i = silu(gate_i) * up_i
  -> local partial_i = mid_i @ W_down_i
  -> all-reduce
  -> full MLP output
```

Example:

```python
# This example simulates TP-style MLP splitting.
# It can run independently.

import numpy as np

def silu(x):
    return x / (1.0 + np.exp(-x))

X = np.array([[1.0, 2.0]])

W_gate = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
])

W_up = np.array([
    [2.0, 1.0, 2.0, 1.0],
    [1.0, 2.0, 1.0, 2.0],
])

W_down = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
])

gate = X @ W_gate
up = X @ W_up
mid = silu(gate) * up
out_full = mid @ W_down

W_gate_0, W_gate_1 = W_gate[:, :2], W_gate[:, 2:]
W_up_0, W_up_1 = W_up[:, :2], W_up[:, 2:]
W_down_0, W_down_1 = W_down[:2, :], W_down[2:, :]

mid0 = silu(X @ W_gate_0) * (X @ W_up_0)
mid1 = silu(X @ W_gate_1) * (X @ W_up_1)

partial0 = mid0 @ W_down_0
partial1 = mid1 @ W_down_1

out_tp = partial0 + partial1

print("out_full =", np.round(out_full, 4))
print("out_tp   =", np.round(out_tp, 4))

# Output:
# out_full = [[49.793  32.9256]]
# out_tp   = [[49.793  32.9256]]
```

<br>

## 12. TP in KV Cache

KV cache (键值缓存) stores past K/V tensors for decoding.

TP splits KV cache by KV heads.

Standard MHA case (num_kv_heads equals num_heads):

```text
num_kv_heads = 32   # MHA: kv heads == attention heads
TP = 4
```

Then:

```text
GPU0: KV heads 0-7
GPU1: KV heads 8-15
GPU2: KV heads 16-23
GPU3: KV heads 24-31
```

Each rank stores:

```text
local K cache
local V cache
```

For GQA (Grouped Query Attention, 分组查询注意力):

```text
num_heads = 32
num_kv_heads = 8    # GQA: fewer kv heads than attention heads
TP = 4
```

Then:

```text
GPU0: KV heads 0-1
GPU1: KV heads 2-3
GPU2: KV heads 4-5
GPU3: KV heads 6-7
```

TP affects KV cache in:

```text
prefill attention
decode attention
paged KV block allocation
prefix cache
GPU memory usage
```

Example:

```python
# This example shows KV cache head ownership (GQA case).
# It can run independently.

num_kv_heads = 8
tp_size = 4

kv_heads_per_rank = num_kv_heads // tp_size

for rank in range(tp_size):
    start = rank * kv_heads_per_rank
    end = start + kv_heads_per_rank - 1
    print(f"GPU{rank}: KV heads {start}-{end}")

# Output:
# GPU0: KV heads 0-1
# GPU1: KV heads 2-3
# GPU2: KV heads 4-5
# GPU3: KV heads 6-7
```

<br>

## 13. TP in Prefill and Decode

Prefill (预填充) processes prompt tokens. Decode (解码) generates new tokens step by step.

During prefill:

```text
prompt tokens
  -> q/k/v projection
  -> local attention
  -> write local K/V cache
  -> o_proj all-reduce
  -> MLP TP
```

During decode:

```text
new token
  -> q/k/v projection
  -> append local K/V cache
  -> read local K/V cache
  -> local attention
  -> o_proj all-reduce
  -> MLP TP
  -> lm_head
  -> sampling
```

Important TP communication points:

```text
o_proj:
  all-reduce

down_proj:
  all-reduce

lm_head / sampling:
  global communication
```

<br>

## 14. TP in Token Embedding

Token embedding (词嵌入) maps token IDs to hidden vectors.

It can use vocabulary parallelism (词表并行).

Assume:

```text
vocab_size = 128000
hidden_size = 4096
TP = 4
```

Embedding table:

```text
embedding: [128000, 4096]
```

Split by vocabulary rows:

```text
GPU0: vocab 0-31999
GPU1: vocab 32000-63999
GPU2: vocab 64000-95999
GPU3: vocab 96000-127999
```

Embedding lookup (mask + all-reduce):

```text
token id
  -> mask tokens not in local vocab shard
  -> local lookup (out-of-range tokens give zero)
  -> all-reduce sum across ranks
```

The trick: each rank zeros out tokens it does not own, so summing all ranks recovers the correct vector.

Example:

```python
# This example shows vocab parallel embedding: mask + all-reduce.
# It can run independently.

import numpy as np

vocab_size = 8
hidden = 3
tp_size = 2
shard = vocab_size // tp_size   # 4 rows per rank

# full embedding table (for reference only)
emb = np.arange(vocab_size * hidden, dtype=float).reshape(vocab_size, hidden)

token_id = 5   # this token belongs to GPU1 (owns vocab 4-7)

def local_lookup(rank, tid):
    start = rank * shard
    end = start + shard
    if start <= tid < end:
        return emb[tid].copy()   # owned -> real vector
    return np.zeros(hidden)      # not owned -> zero (masked)

out0 = local_lookup(0, token_id)
out1 = local_lookup(1, token_id)

result = out0 + out1   # all-reduce sum

print("GPU0 local:", out0)
print("GPU1 local:", out1)
print("all-reduce :", result)
print("ground truth:", emb[token_id])

# Output:
# GPU0 local: [0. 0. 0.]
# GPU1 local: [15. 16. 17.]
# all-reduce : [15. 16. 17.]
# ground truth: [15. 16. 17.]
```

<br>

## 15. TP in LM Head and Logits

`lm_head` maps hidden states to vocabulary logits.

It can also use vocabulary parallelism.

Assume:

```text
hidden_states: [batch, hidden_size]
lm_head: [hidden_size, vocab_size]
vocab_size = 128000
TP = 4
```

Each rank computes logits for its own vocab shard:

```text
GPU0: logits for vocab 0-31999
GPU1: logits for vocab 32000-63999
GPU2: logits for vocab 64000-95999
GPU3: logits for vocab 96000-127999
```

The full logits are logically:

```text
logits = concat(local_logits_0, local_logits_1, local_logits_2, local_logits_3)
```

But the best token can be on any rank, so sampling needs global communication.

Example:

```python
# This example shows why logits need global comparison.
# It can run independently.

import numpy as np

logits_gpu0 = np.array([1.0, 2.0, 3.0])
logits_gpu1 = np.array([0.5, 4.5, 1.5])

local_max_gpu0 = logits_gpu0.max()
local_max_gpu1 = logits_gpu1.max()

global_max = max(local_max_gpu0, local_max_gpu1)

print("local_max_gpu0 =", local_max_gpu0)
print("local_max_gpu1 =", local_max_gpu1)
print("global_max =", global_max)

# Output:
# local_max_gpu0 = 3.0
# local_max_gpu1 = 4.5
# global_max = 4.5
```

<br>

## 16. TP in Sampling

Sampling (采样) selects the next token from logits.

When logits are split by vocab, sampling must be globally correct.

Sampling may need:

```text
global argmax
global top-k
global top-p
global probability normalization
```

Greedy decoding:

```text
each rank finds local max
all ranks compare local max
global best token is selected
```

Top-k sampling:

```text
each rank finds local top-k
all ranks merge candidates
global top-k is selected
```

Top-p sampling:

```text
probability mass must be computed across vocab shards
```

<br>

## 17. TP Communication Summary

TP saves GPU memory by splitting tensors, but it introduces communication.

Main communication patterns:

```text
Column Parallel:
  output may stay sharded
  all-gather / concat may be needed

Row Parallel:
  partial outputs must be summed
  all-reduce is needed

Embedding:
  lookup results need all-reduce sum

LM Head:
  logits may be sharded

Sampling:
  token decision must be global
```

Most important all-reduce points:

```text
o_proj
down_proj
```

<br>

## 18. TP with DP

TP can be combined with Data Parallelism (数据并行).

```text
TP:
  splits one model replica across multiple GPUs

DP:
  creates multiple model replicas for higher throughput
```

Example:

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --data-parallel-size 4
```

Meaning:

```text
Each replica uses 2 GPUs.
There are 4 replicas.
Total GPUs = 2 * 4 = 8.
```

Layout:

```text
Replica 0: GPU0, GPU1
Replica 1: GPU2, GPU3
Replica 2: GPU4, GPU5
Replica 3: GPU6, GPU7
```

<br>

## 19. vLLM Commands

Use TP when one model replica needs multiple GPUs.

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4
```

Meaning:

```text
One model replica is split across 4 GPUs.
```

Use TP + DP when both memory scaling and throughput scaling are needed.

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 2 \
  --data-parallel-size 4
```

Meaning:

```text
Each replica uses 2 GPUs.
There are 4 replicas.
Total GPUs = 8.
```

<br>

## 20. Final Summary Table

| Module               | TP Method           | Communication                 |
| -------------------- | ------------------- | ----------------------------- |
| token embedding      | Vocabulary Parallel | all-reduce sum of lookup      |
| q_proj               | Column Parallel     | usually local                 |
| k_proj               | Column Parallel     | usually local                 |
| v_proj               | Column Parallel     | usually local                 |
| rotary embedding     | Local per rank      | none                          |
| paged attention      | Local per rank      | none before o_proj            |
| attention heads      | Head Parallel       | local per rank                |
| o_proj               | Row Parallel        | all-reduce                    |
| gate_proj            | Column Parallel     | local                         |
| up_proj              | Column Parallel     | local                         |
| activation           | Local per rank      | none                          |
| elementwise multiply | Local per rank      | none                          |
| down_proj            | Row Parallel        | all-reduce                    |
| KV cache             | KV Head Parallel    | local cache access            |
| lm_head              | Vocabulary Parallel | global sampling communication |
| logits               | Vocabulary Parallel | global coordination           |
| sampling             | Global decision     | cross-rank communication      |

<br>

## 21. Key Rules

```text
q_proj, k_proj, v_proj:
  Column Parallel

o_proj:
  Row Parallel + all-reduce

gate_proj, up_proj:
  Column Parallel

activation and elementwise multiply:
  local per rank

down_proj:
  Row Parallel + all-reduce

attention heads:
  split across TP ranks

rotary embedding:
  local to each rank's Q/K heads

paged attention:
  local to each rank's KV cache

KV cache:
  split by KV heads

token embedding:
  may use vocab parallel (mask + all-reduce)

lm_head:
  may use vocab parallel

logits and sampling:
  need global communication
```

Shortest version:

```text
TP splits one model replica inside each Transformer layer:
q/k/v and gate/up use Column Parallel,
o_proj and down_proj use Row Parallel,
attention and KV cache are split by heads,
embedding and lm_head can be split by vocab,
and sampling needs global communication.
```

<br> <br>
