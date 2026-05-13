---
title: "Parallelism Strategy"
published: 2026-05-05
description: "Parallelism Strategy"
image: ""
tags: ["llm","Parallelism Strategy"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-05T04:07:13.124.473064193Z"
---

# Parallelism Strategy

## 1. Data Parallelism (数据并行) — Solves "Training is Too Slow"

**Core problem:** The model fits on a single GPU, but the dataset is so large that single-GPU training would take months.

### 1) Numerical example

Suppose you want to fine-tune **LLaMA-2 7B** (14 GB of weights in FP16):

-   Single A100 80GB: fits comfortably (weights 14 GB + gradients 14 GB + Adam optimizer states 28 GB ≈ 56 GB)
-   Training data: **10 billion tokens**
-   Single-GPU throughput: ~**3000 tokens/sec**
-   Single-GPU training time: 10B / 3000 ≈ **38 days**

Way too slow. Solution? **Replicate the model on 8 GPUs, each handling 1/8 of the data.**

### 2) GPU configuration

```
8 × A100 80GB (one DGX A100 server)

GPU 0: Full LLaMA-2 7B replica + samples [0:32]
GPU 1: Full LLaMA-2 7B replica + samples [32:64]
GPU 2: Full LLaMA-2 7B replica + samples [64:96]
...
GPU 7: Full LLaMA-2 7B replica + samples [224:256]

End of each step: 8 GPUs all-reduce gradients (~14 GB of data, over NVLink)
```

### 3) Payoff

-   Throughput: 3000 → **~24,000 tokens/sec** (8×)
-   Training time: 38 days → **~5 days**

### 4) When it stops working

Switch to **LLaMA-2 70B** (140 GB) and DP **breaks immediately** — no single GPU can hold a full replica. DP's prerequisite is "every GPU can fit the whole model," and that no longer holds.

<br>

## 2. Pipeline Parallelism (流水线并行) — Solves "Model is Too Deep to Fit on One GPU"

**Core problem:** The model has so many layers that the total weights exceed a single GPU's memory.

### 1) Numerical example

**LLaMA-2 70B**: 80 layers, ~1.75 GB of weights per layer:

-   Total weights: 80 × 1.75 GB ≈ **140 GB**
-   Plus KV cache (during inference) + activations: another tens of GBs
-   **Single H100 80GB: doesn't fit**
-   **Single H200 141GB: barely fits the weights, no room for KV cache**

Solution? **Cut the model into 4 segments by layer, one per GPU.**

### 2) GPU configuration

```
4 × H100 80GB

GPU 0: Layer 1-20    (~35 GB weights) + embed_tokens
GPU 1: Layer 21-40   (~35 GB weights)
GPU 2: Layer 41-60   (~35 GB weights)
GPU 3: Layer 61-80   (~35 GB weights) + norm + lm_head

Forward pass for one token:
  GPU 0 finishes → sends 16 KB hidden state → GPU 1 → GPU 2 → GPU 3 outputs
```

### 3) Payoff

-   Per-GPU memory pressure: 140 GB → **~35 GB** (1/4)
-   Communication: only ~48 KB per token (3 stage transitions), negligible
-   **70B model finally runs**

### 4) The cost: pipeline bubbles

```
Time →
GPU 0: [B1][B2][B3][B4][  ][  ][  ]   ← idle in cool-down
GPU 1:     [B1][B2][B3][B4][  ][  ]
GPU 2:         [B1][B2][B3][B4][  ]
GPU 3:             [B1][B2][B3][B4]   ← idle in warm-up
```

With only 4 micro-batches and 4 stages, bubble ratio can hit **~40%**. You squeeze it down to 5–10% by cramming in more micro-batches.

<br>

## 3. Tensor Parallelism (张量并行) — Solves "A Single Layer is Too Wide for One GPU"

**Core problem:** Even after cutting by layer, **a single layer's matrices** are too big for one GPU.

### 1) Numerical example

**LLaMA-2 70B FFN layer:**

-   `gate_proj`: `[8192, 28672]`, FP16 → **0.45 GB**
-   `up_proj`:   `[8192, 28672]`, **0.45 GB**
-   `down_proj`: `[28672, 8192]`, **0.45 GB**
-   Per-layer FFN weights: **1.35 GB**

Looks small? Now look at **Megatron-Turing NLG 530B**:

-   hidden_size = 20480, FFN intermediate size = 81920
-   One FFN matrix: `[20480, 81920]` × FP16 = **3.2 GB**
-   Per-layer FFN total: **~10 GB**

Add gradients (×2) and optimizer states (×4 for Adam) and **a single layer balloons to ~70 GB — too big for one GPU**.

PP can't save you here — PP cuts **between layers**, not **inside** a layer.

### 2) GPU configuration

**TP=8 splitting one LLaMA-2 70B layer:**

```
8 × H100 80GB (single node, NVLink at 900 GB/s)

Each GPU holds:
  qkv_proj slice:       [8192, 3072]   ≈ 50 MB
  o_proj slice:         [1024, 8192]   ≈ 17 MB
  gate_up_proj slice:   [8192, 7168]×2 ≈ 235 MB
  down_proj slice:      [3584, 8192]   ≈ 60 MB

  Per-layer load split across 8 GPUs: ~360 MB/GPU instead of 1.35 GB
```

Executing one layer:

1.  All 8 GPUs compute their own QKV slice in parallel
2.  All 8 GPUs compute their own 8 attention heads
3.  **All-reduce** after `o_proj` (sync #1)
4.  FFN column-wise computation
5.  **All-reduce** after `down_proj` (sync #2)

**8 GPUs cooperatively execute one layer.**

### 3) Payoff

-   Per-GPU memory pressure: 1.35 GB/layer → **~360 MB/layer** (1/8)
-   Solves the hard constraint of "one matrix doesn't fit"
-   All GPUs busy at the same time — **no bubble**

### 4) The cost: communication is brutal

Two all-reduces per layer. LLaMA-2 70B with 80 layers = **160 all-reduces per forward pass**. Each all-reduce moves a few MB of data across all 8 GPUs.

**Total communication is orders of magnitude higher than PP**, which is why TP demands NVLink — running TP across nodes over InfiniBand causes a 5–10× slowdown.

<br>

## 4. Quick Mental Model

| Strategy               | Problem solved             | Cuts what        | Best link needed |
| ---------------------- | -------------------------- | ---------------- | ---------------- |
| Data Parallel (DP)     | Training too slow          | The data batch   | Any              |
| Pipeline Parallel (PP) | Model too **deep** to fit  | Layers (depth)   | Cross-node OK    |
| Tensor Parallel (TP)   | Layers too **wide** to fit | Matrices (width) | NVLink only      |

>   **DP makes training fast. PP fits deep models. TP fits wide layers.** **Small models: DP is enough. Mid-size: DP + PP. Giant models: DP + PP + TP, all three.**
