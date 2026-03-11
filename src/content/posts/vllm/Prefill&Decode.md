---
title: "Prefill&Decode"
published: 2026-03-10
description: "Prefill&Decode"
image: ""
tags: ["vllm","Prefill&Decode"]
category: vllm
draft: false
lang: ""
---

# **I. Prefill (预填充) vs. Decode (解码)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Every LLM inference request (推理请求) passes through exactly two phases. <span style="color:#E8600A;font-weight:700">Prefill (预填充)</span> processes all input tokens at once to build the model’s context(tokens（文本序列）) and KV cache. <span style="color:#E8600A;font-weight:700">Decode (解码)</span> then generates the output tokens one by one using that cached context.. Understanding the difference between these two phases is essential for diagnosing latency (延迟), optimizing throughput (吞吐量), and making sense of every modern serving optimization. </div>

------

## 1. The Big Picture (整体流程)

**Interviewer:** *"Walk me through what happens when I send a prompt to an LLM."*

>   Sure. Take the prompt `"Translate to French: Hello world"`. The model first runs **Prefill** — it reads all 6 tokens simultaneously and builds the KV Cache. Then it enters **Decode** — it generates `"Bonjour"`, then `"monde"`, then `<EOS>`, one token at a time. Two completely different compute patterns, back to back.

```
User Prompt:  "Translate to French: Hello world"
               ↓
┌─────────────────────────────────────┐
│  PREFILL PHASE (预填充阶段)           │
│  Input:  [T1][T2][T3][T4][T5][T6]  │  ← all tokens in parallel (并行)
│  Output: KV Cache built ✅           │
└─────────────────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  DECODE PHASE (解码阶段)              │
│  Step 1: KV Cache → "Bonjour"       │  ← one token per step (逐步生成)
│  Step 2: KV Cache → "monde"         │
│  Step 3: KV Cache → <EOS> → DONE   │
└─────────────────────────────────────┘
```

------

## 2. Prefill Phase Deep Dive (预填充阶段详解)

### 1) What Happens (发生了什么)

**Interviewer:** *"What exactly is the model doing during Prefill?"*

>   It runs one single forward pass (前向传播) over all N prompt tokens **at the same time**. During **prefill**, the model reads all tokens, computes attention, and stores their key–value vectors in the KV cache.

$$ \text{Prefill Cost} \propto O(N^2) \quad \text{(due to full self-attention over N prompt tokens)} $$

### 2) Why It's Compute-Bound (为什么是计算密集型)

>   Because we're doing a massive matrix multiplication (矩阵乘法) over all N tokens in parallel. The GPU arithmetic units (算术单元) are hammered — this is the phase where GPU utilization (GPU利用率) peaks. A long prompt of 4096 tokens is far more expensive than a 64-token prompt.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Prefill time directly determines <span style="color:#E8600A;font-weight:700">TTFT (Time-To-First-Token, 首个令牌时间)</span> — the latency the user feels before they see any output. Long prompts = slow TTFT.</div>

### 3) KV Cache Construction (KV缓存构建)

>   During Prefill, for every Transformer layer (Transformer层), we compute and **store** the Key (K) and Value (V) matrices for every prompt token. These are saved so Decode steps never have to recompute them.

```python
# Simplified: what Prefill produces
def prefill(prompt_tokens, model):
    # One forward pass over all tokens simultaneously
    hidden_states = model.embed(prompt_tokens)          # [N, d_model]
    
    kv_cache = {}
    for layer_idx, layer in enumerate(model.layers):
        K = layer.W_k @ hidden_states                   # [N, d_k]
        V = layer.W_v @ hidden_states                   # [N, d_v]
        kv_cache[layer_idx] = (K, V)                    # ← saved for Decode
        hidden_states = layer.attention(hidden_states, K, V)
    
    first_token = model.lm_head(hidden_states[-1])      # predict next token
    return first_token, kv_cache                        # ← hand off to Decode
```

------

## 3. Decode Phase Deep Dive (解码阶段详解)

### 1) What Happens (发生了什么)

**Interviewer:** *"And during Decode?"*

>   During **decode**, it uses that cache to predict the next token, append it to the sequence, update the KV cache, and repeat until `<EOS>`.

### 2) Why It's Memory-Bound (为什么是内存密集型)

>   Each Decode step does very little compute — just one token — but it must **read the entire KV Cache** from GPU memory (显存) on every step. For a 7B model with a long context, that's gigabytes of data transferred per step. The GPU sits mostly waiting for memory, not computing.

$$ \text{Decode Cost per step} \propto O(N_{\text{ctx}}) \quad \text{(reading KV Cache of length } N_{\text{ctx}}\text{)} $$

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This is why <span style="color:#C0392B;font-weight:600">increasing batch size helps Decode more than Prefill</span> — batching amortizes the memory read cost across many sequences sharing the same GPU memory bandwidth (显存带宽).</div>

### 3) Autoregressive Loop (自回归循环)

```python
# Simplified: the Decode loop
def decode(first_token, kv_cache, model, max_new_tokens=200):
    generated = [first_token]
    current_token = first_token

    for _ in range(max_new_tokens):
        # Only ONE token as input — very small compute
        hidden = model.embed([current_token])             # [1, d_model]

        for layer_idx, layer in enumerate(model.layers):
            K_new = layer.W_k @ hidden                    # [1, d_k]
            V_new = layer.W_v @ hidden                    # [1, d_v]

            # Append to cache (缓存追加)
            K_full, V_full = kv_cache[layer_idx]
            K_full = concat(K_full, K_new)                # [N+t, d_k]
            V_full = concat(V_full, V_new)
            kv_cache[layer_idx] = (K_full, V_full)

            # Attend over FULL history — memory read dominates (内存读取主导)
            hidden = layer.attention(hidden, K_full, V_full)

        next_token = model.lm_head(hidden[0])
        if next_token == EOS_TOKEN:
            break
        generated.append(next_token)
        current_token = next_token

    return generated
```

------

## 4. Side-by-Side Comparison (对比总结)

| Dimension                   | Prefill (预填充)               | Decode (解码)                  |
| --------------------------- | ------------------------------ | ------------------------------ |
| Input size (输入规模)       | N prompt tokens (all at once)  | 1 token per step               |
| Parallelism (并行度)        | Full parallel (完全并行)       | Sequential (串行)              |
| Bottleneck (瓶颈)           | Compute / FLOPS (算力)         | Memory bandwidth (显存带宽)    |
| Complexity (复杂度)         | $O(N^2)$ attention             | $O(N_\text{ctx})$ cache read   |
| Latency impact (延迟影响)   | Determines TTFT (首个令牌时间) | Determines TPOT (每个令牌时间) |
| KV Cache role               | **Build** the cache            | **Read + Append** the cache    |
| GPU utilization (GPU利用率) | High (高)                      | Lower (较低)                   |

------

## 5. Key Metrics (关键指标)

### 1) TTFT — Time To First Token (首个令牌时间)

**Interviewer:** *"What's the main latency metric for Prefill?"*

>   <span style="color:#E8600A;font-weight:700">TTFT</span> is how long the user waits before seeing the first word of the response. It's dominated entirely by Prefill time. Long system prompts (系统提示词) or RAG context (检索增强生成上下文) bloat TTFT because the model must digest all those tokens before generating anything.

### 2) TPOT — Time Per Output Token (每个输出令牌时间)

>   <span style="color:#E8600A;font-weight:700">TPOT</span> is the per-step latency during Decode — how fast tokens stream out. Users perceive this as the "typing speed" of the model. It's bounded by memory bandwidth and batch size.

$$ \text{Total Latency} = \underbrace{\text{TTFT}}*{\text{Prefill}} + \underbrace{(L - 1) \times \text{TPOT}}*{\text{Decode, L output tokens}} $$

------

## 6. Optimization Strategies (优化策略)

### 1) For Prefill — Reduce TTFT

-   <span style="color:#2980B9">**Chunked Prefill (分块预填充)**</span>: Split a long prompt into chunks, interleaved with Decode steps — prevents one big Prefill from blocking ongoing requests.
-   <span style="color:#2980B9">**Prompt Caching (提示词缓存)**</span>: If the system prompt is reused across requests, cache its KV pairs once and skip re-computing Prefill for that portion.
-   <span style="color:#2980B9">**Prefix Caching (前缀缓存)**</span>: Same idea — hash prompt prefixes, reuse cached KV if seen before.

### 2) For Decode — Improve Throughput

-   <span style="color:#2980B9">**Continuous Batching (连续批处理)**</span>: Keep GPU slots full by swapping finished sequences out every step.
-   <span style="color:#2980B9">**Speculative Decoding (投机解码)**</span>: Use a small draft model to propose multiple tokens, verify with the main model in one pass — effectively generates >1 token per step.
-   <span style="color:#2980B9">**PagedAttention (分页注意力)**</span>: Eliminate KV Cache fragmentation (碎片化) so more sequences fit in memory simultaneously.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Common pitfall (常见误区)</span>: optimizing Prefill and Decode requires different strategies — don't conflate them. A trick that halves TTFT (e.g., chunked prefill) may have zero effect on TPOT, and vice versa.</div>

------

## 7. Chunked Prefill (分块预填充) — Best of Both Worlds

**Interviewer:** *"You mentioned Chunked Prefill — how does it work?"*

>   Standard Prefill is a monolithic operation — a 4096-token prompt monopolizes the GPU for tens of milliseconds, spiking latency for every other user. Chunked Prefill breaks the prompt into fixed-size chunks (e.g., 512 tokens) and interleaves them with Decode steps from other requests.

```
Without Chunked Prefill (没有分块预填充):
Step:  [  PREFILL 4096 tokens  ][D][D][D][D]...
        ↑ all other requests stall here (其他请求全部阻塞)

With Chunked Prefill (有分块预填充):
Step:  [PF-chunk1][D][D][PF-chunk2][D][D][PF-chunk3][D][D]...
        ↑ Decode requests keep flowing (解码请求持续推进)
```

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Prefill (预填充) is a <em>one-shot parallel compute burst</em> that builds the KV Cache and sets TTFT; Decode (解码) is a <em>memory-bound sequential loop</em> that streams tokens out one by one — every serving optimization ultimately targets one or both of these two fundamentally different bottlenecks.</div>
