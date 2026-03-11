---
title: "Chunked Prefill"
published: 2026-03-10
description: "Chunked Prefill"
image: ""
tags: ["vllm","Chunked Prefill"]
category: vllm
draft: false
lang: ""
---

# **I. Chunked Prefill (分块预填充)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">One-sentence definition:</span> Split a long prompt into small chunks, and interleave those chunks with ongoing Decode steps — so a big Prefill never monopolizes (独占) the GPU and blocks other users, which improves throughput.<br><br> <span style="color:#3B5BDB;font-weight:700">Why it matters:</span> Standard Prefill is a GPU-hogging monolith (独占性巨块). Chunked Prefill turns it into a polite, interruptible background task — keeping <span style="color:#E8600A;font-weight:700">TTFT (首个令牌时间，Time To First Token)</span> low for new requests while keeping <span style="color:#27AE60;font-weight:700">TPOT (每个输出令牌时间)</span> smooth for ongoing ones. </div>

------

## 1. The Problem It Solves (它解决什么问题)

### 1) Standard Prefill Blocks Everyone (标准预填充阻塞所有人)

<span style="color:#C0392B;font-weight:600">Imagine User A sends a 4096-token prompt.</span> The GPU must process all 4096 tokens in one forward pass — this takes tens of milliseconds. During that time, Users B, C are all frozen, waiting.

```
❌ Without Chunked Prefill (没有分块预填充):

GPU Timeline ──────────────────────────────────────────────▶
│◀────── PREFILL: 4096 tokens (40ms) ──────▶│DC│DC│DC│DC│DC│DC│DC│
                                               ↑
                         Users B, C stall here (全部阻塞)
                         TTFT for B, C = 40ms + queue wait
✅ With Chunked Prefill (有分块预填充):

GPU Timeline ──────────────────────────────────────────────▶
│PF-512│DC│DC│PF-512│DC│DC│PF-512│DC│DC│PF-512│DC│DC│PF-512│DC│DC│...│
   ↑         ↑         ↑         ↑         ↑
  chunk1   chunk2   chunk3   chunk4   chunk5   (8 chunks total)

Decode requests keep flowing every step ✅
New requests get slots within milliseconds ✅
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The chunk size (块大小) is a tunable hyperparameter — typically <strong>256 to 512 tokens</strong>. Smaller = smoother Decode, but more scheduling overhead. Larger = faster Prefill completion, but more jitter.</div>

------

## 2. Key Concepts (核心概念)

### 1) <span style="color:#8E44AD;font-weight:700">Chunk (块)</span> — The atomic unit

>   One piece of the prompt. A 2048-token prompt with chunk size 512 becomes **4 chunks**, each processed in a separate iteration.

$$ \text{Num Chunks} = \left\lceil \frac{\text{Prompt Length (提示词长度)}}{\text{Chunk Size (块大小)}} \right\rceil $$

### 2) <span style="color:#27AE60;font-weight:700">Interleaving (交错调度)</span> — The key insight

>   Between every two chunks, the scheduler runs one or more **Decode steps** for other active sequences. Prefill and Decode coexist in the same batch — they are no longer separate phases.

### 3) <span style="color:#E8600A;font-weight:700">Partial KV Cache (部分KV缓存)</span> — Built incrementally

>   After each chunk, the KV Cache grows by `chunk_size` entries. The full KV Cache is ready only after the **last chunk** — only then does the sequence enter Decode mode.

```
Prompt = [T1 T2 ... T2048],  chunk_size = 512

After chunk 1:  KV Cache = [T1 ... T512]    ← partial (部分)
After chunk 2:  KV Cache = [T1 ... T1024]   ← partial
After chunk 3:  KV Cache = [T1 ... T1536]   ← partial
After chunk 4:  KV Cache = [T1 ... T2048]   ← complete ✅ → enters Decode
```

------

## 3. Scheduler Logic (调度器逻辑)

### 1) Per-Iteration Decision (每次迭代的决策)

```python
# Simplified Chunked Prefill scheduler (简化调度器)
CHUNK_SIZE = 512

def schedule_iteration(prefill_queue, decode_batch, kv_manager):
    batch = []

    # 1. Add one chunk from each waiting prefill request (添加预填充块)
    for req in prefill_queue:
        chunk = req.next_chunk(CHUNK_SIZE)     # slice next 512 tokens
        batch.append(("prefill_chunk", req, chunk))
        if req.is_prefill_done():
            prefill_queue.remove(req)
            decode_batch.append(req)           # graduate to Decode (晋升至解码)

    # 2. Add all active decode sequences (添加解码序列)
    for req in decode_batch:
        batch.append(("decode", req, req.last_token))

    # 3. One forward pass — mixed batch (混合批次前向传播)
    return model_forward(batch)
```

<div style="background:#F5F5F5;border-left:4px solid #8E44AD;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#8E44AD;font-weight:700">Key insight: </span> The forward pass (前向传播) now contains a <strong>mixed batch (混合批次)</strong> — some tokens are Prefill chunks, some are Decode tokens. The Transformer handles them in the same kernel call, just with different attention masks (注意力掩码).</div>

### 2) Attention Mask Difference (注意力掩码的区别)

| Token Type                                                   | Can Attend To                     | Mask Pattern             |
| ------------------------------------------------------------ | --------------------------------- | ------------------------ |
| <span style="color:#3B5BDB;font-weight:700">Prefill chunk token (预填充块令牌)</span> | All previous tokens in same chunk | Causal mask within chunk |
| <span style="color:#27AE60;font-weight:700">Decode token (解码令牌)</span> | Full KV Cache history             | Full causal mask         |

------

## 4. Impact on Latency Metrics (对延迟指标的影响)

### 1) <span style="color:#E8600A;font-weight:700">TTFT (Time-To-First-Token, 首个令牌时间)</span>

<span style="color:#C0392B;font-weight:600">Standard Prefill:</span> TTFT ∝ full prompt length — a 4096-token prompt forces everyone to wait.

<span style="color:#27AE60;font-weight:700">Chunked Prefill:</span> TTFT is bounded by chunk size — new requests get scheduled within one chunk interval.

$$ \text{TTFT}*{\text{chunked}} \approx \frac{\text{Prompt Length}}{\text{Chunk Size}} \times T*{\text{iter}} \quad \ll \quad \text{TTFT}_{\text{standard}} $$

### 2) <span style="color:#27AE60;font-weight:700">TPOT (Time-Per-Output-Token, 每个输出令牌时间)</span>

>   Without Chunked Prefill, a large Prefill **spikes TPOT** for all Decode sequences — they get zero GPU time during that window. With Chunked Prefill, Decode sequences get at least one step per chunk interval. TPOT stays smooth.

```
TPOT without Chunked Prefill:
──────────────────────────────────────────────────▶ time
█████████████████░░░░░░░░░░░░░░░░░░░░░█████████████
      Decode tokens           PREFILL blocks ← spike!

TPOT with Chunked Prefill:
──────────────────────────────────────────────────▶ time
██░██░██░██░██░██░██░██░██░██░██░██░██░██░██░██░██
D P  D P  D P  D P  D P  D P  ...   (smooth interleaving ✅)
```

------

## 5. Comparison Table (对比总表)

|                             | <span style="color:#C0392B;font-weight:700">Standard Prefill</span> | <span style="color:#27AE60;font-weight:700">Chunked Prefill</span> |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Prefill unit (预填充单位)   | Entire prompt at once                                        | Fixed-size chunk per iteration                               |
| TTFT (首个令牌时间)         | High for long prompts                                        | Bounded by chunk size                                        |
| TPOT jitter (TPOT抖动)      | High — Prefill blocks Decode                                 | Low — Decode runs every step                                 |
| GPU utilization (GPU利用率) | Bursty                                                       | Smooth and high                                              |
| KV Cache build              | One shot                                                     | Incremental per chunk                                        |
| Scheduling complexity       | Simple                                                       | Moderate                                                     |
| Used in                     | Naive servers                                                | vLLM, SGLang, TGI                                            |

------

## 6. Configuration in vLLM (vLLM中的配置)

```bash
# Enable Chunked Prefill with custom chunk size
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512    # ← chunk size (块大小)
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_chunked_prefill=True,
    max_num_batched_tokens=512,     # chunk size
)
```

<div style="background:#F5F5F5;border-left:4px solid #27AE60;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#27AE60;font-weight:700">Tip: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">max_num_batched_tokens</code> controls the <strong>total tokens per iteration</strong> across all active sequences. Tuning this value balances Prefill throughput against Decode smoothness — start with <strong>512</strong> and profile.</div>

------

## 7. When to Use Chunked Prefill (什么时候用)

| Scenario (场景)                                  | Use Chunked Prefill?                                         | Reason                                       |
| ------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| Long system prompts / RAG context (长提示词/RAG) | <span style="color:#27AE60;font-weight:700">✅ Yes</span>     | Prevents TTFT spike for other users          |
| Interactive chatbot (交互式聊天)                 | <span style="color:#27AE60;font-weight:700">✅ Yes</span>     | Low TPOT jitter = smooth streaming           |
| Offline batch jobs (离线批处理)                  | <span style="color:#E8600A;font-weight:700">⚠️ Maybe</span>   | Throughput may drop slightly vs full Prefill |
| Short prompts < 512 tokens                       | <span style="color:#C0392B;font-weight:600">❌ Not needed</span> | Prompt fits in one chunk anyway              |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#F0FFF4 50%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Chunked Prefill = <span style="color:#8E44AD;font-weight:700">cut the prompt into small pieces</span> + <span style="color:#27AE60;font-weight:700">interleave with Decode every step</span> → no single request can hog the GPU, TTFT stays low, TPOT stays smooth — <em>best of both worlds (两全其美)</em>.</div>

