---
title: "Continuous Batching"
published: 2026-03-09
description: "Continuous Batching"
image: ""
tags: ["llm","Continuous Batching"]
category: llm
draft: false
lang: ""
---

# I. Continuous Batching (连续批处理)

![image-20260415031834502](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260415031834502)

## 1. Background

Traditional LLM serving uses **static batching (静态批处理)**: a fixed group of requests is loaded together, the GPU runs until *every* request in the batch finishes, then the next batch starts. Two problems arise:

-   **Padding waste (填充浪费)**: short requests must be padded to match the longest request in the batch, wasting compute.
-   **Head-of-line blocking (队头阻塞)**: new requests wait for the entire current batch to complete, even if most slots are already idle.

Continuous Batching (also called *iteration-level scheduling*, 迭代级调度) solves both by scheduling at the granularity of a **single iteration** rather than a whole batch.

------

## 2. Core Idea

>   As soon as one request finishes, its GPU slot is freed and a new request is admitted — **within the same next iteration**.

The scheduler runs once per forward pass (前向传播). It looks at:

1.  Which running requests still need decode steps.
2.  Whether any free capacity exists to admit a new prefill request.

This means the batch composition changes **every iteration**, hence "continuous."

------

## 3. Algorithm

### 1) Iteration-Level Scheduler (迭代级调度器)

```python
# continuous_batching_demo.py
# Standalone simulation of continuous batching.
# No external dependencies required.

from collections import deque

class Request:
    """One inference request (推理请求)."""
    def __init__(self, req_id: str, prompt_len: int, max_new_tokens: int):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.generated = 0          # tokens generated so far
        self.prefill_done = False

    def is_done(self) -> bool:
        return self.prefill_done and self.generated >= self.max_new_tokens

    def step(self):
        """Simulate one decode step (解码步骤)."""
        if not self.prefill_done:
            self.prefill_done = True    # single-step prefill (simplified)
        else:
            self.generated += 1

def run_continuous_batching(
    all_requests: list[Request],
    max_batch_size: int = 3,
    max_iterations: int = 20,
):
    waiting_queue = deque(all_requests)   # requests not yet admitted
    running: list[Request] = []           # currently active requests
    finished: list[Request] = []

    for iteration in range(1, max_iterations + 1):
        # ── 1. Evict finished requests (移除完成的请求) ──────────────────
        done = [r for r in running if r.is_done()]
        for r in done:
            running.remove(r)
            finished.append(r)
            print(f"  [Done]    {r.req_id} finished at iteration {iteration}")

        # ── 2. Admit new requests to fill freed slots (补充新请求) ────────
        while waiting_queue and len(running) < max_batch_size:
            new_req = waiting_queue.popleft()
            running.append(new_req)
            print(f"  [Admit]   {new_req.req_id} admitted at iteration {iteration}")

        if not running:
            print(f"Iteration {iteration}: all done.")
            break

        # ── 3. Step every running request (执行一步) ──────────────────────
        print(f"\n--- Iteration {iteration} | batch={[r.req_id for r in running]} ---")
        for r in running:
            r.step()

    print("\n=== Summary ===")
    for r in finished:
        print(f"  {r.req_id}: prompt={r.prompt_len}, generated={r.generated}")

if __name__ == "__main__":
    requests = [
        Request("R1", prompt_len=10, max_new_tokens=2),
        Request("R2", prompt_len=8,  max_new_tokens=5),
        Request("R3", prompt_len=12, max_new_tokens=3),
        Request("R4", prompt_len=6,  max_new_tokens=2),
        Request("R5", prompt_len=9,  max_new_tokens=4),
    ]
    run_continuous_batching(requests, max_batch_size=3)
```

**Expected output (abridged):**

```
--- Iteration 1 | batch=['R1', 'R2', 'R3'] ---
  [Admit]   R1 admitted at iteration 1
  ...
--- Iteration 3 | batch=['R1', 'R2', 'R3'] ---
  [Done]    R1 finished at iteration 4
  [Admit]   R4 admitted at iteration 4    ← slot freed, new request in immediately
--- Iteration 4 | batch=['R2', 'R3', 'R4'] ---
...
```

------

### 2) Key Invariant (关键不变式)

At every iteration $t$:

$$ |\text{running}*t| \leq B*{\max} $$

where $B_{\max}$ is the maximum batch size (最大批大小), constrained by GPU memory (显存) available for KV Cache (键值缓存).

------

## 4. No Padding Needed

In static batching, all sequences in a batch must share the same length tensor, requiring padding tokens (填充令牌):

```
Static:  [A A A A _ _]   ← _ = wasted padding
         [B B _ _ _ _]
         [C C C C C C]
```

In continuous batching with **PagedAttention (分页注意力)**, each request owns its own KV cache pages. The forward pass uses variable-length attention — no padding:

```
Continuous:  [A A A A]   ← exact length, no padding
             [B B]
             [C C C C C C]
```

------

## 5. Comparison Table

| Property                               | Static Batching (静态批处理)     | Continuous Batching (连续批处理) |
| -------------------------------------- | -------------------------------- | -------------------------------- |
| Scheduling granularity (调度粒度)      | Per-batch                        | Per-iteration                    |
| Padding (填充)                         | Required                         | Not needed                       |
| GPU idle time (GPU空闲时间)            | High (slot waits for stragglers) | Near zero                        |
| Latency for new requests (新请求延迟)  | Full batch wait                  | At most one iteration            |
| Implementation complexity (实现复杂度) | Simple                           | Moderate                         |
| Typical throughput gain (吞吐量提升)   | Baseline                         | 2–5× higher                      |

------

## 6. Key Metrics (关键指标)

$$ \text{Throughput (吞吐量)} = \frac{\text{Total output tokens}}{\text{Wall-clock time}} $$

$$ \text{TTFT (首个令牌时间)} = t_{\text{first token}} - t_{\text{request arrival}} $$

$$ \text{ITL (令牌间延迟)} = \frac{t_{\text{last token}} - t_{\text{first token}}}{\text{output tokens} - 1} $$

Continuous batching primarily improves **throughput** and reduces **queuing latency (排队延迟)** for newly arriving requests.

------

## 7. Related Concepts (相关概念)

-   **Chunked Prefill (分块预填充)** — splits long prefills into chunks; a scheduling strategy that sits *on top of* continuous batching.
-   **PagedAttention (分页注意力)** — enables variable-length KV cache per request; prerequisite for efficient continuous batching.
-   **Preemption (抢占)** — when KV cache is full, the scheduler may evict a low-priority request and recompute later.
-   **vLLM** — the open-source serving framework that popularized continuous batching + PagedAttention.
