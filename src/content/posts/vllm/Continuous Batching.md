---
title: "Continuous Batching"
published: 2026-03-09
description: "Continuous Batching"
image: ""
tags: ["vllm","Continuous Batching"]
category: vllm
draft: false
lang: ""
---

# **I. Continuous Batching (连续批处理)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<span style="color:#E8600A;font-weight:700">Continuous Batching (连续批处理)</span> is an optimization technique for LLM inference that <span style="color:#2980B9">dynamically adds new requests</span> to an ongoing batch as soon as previous ones finish. Unlike traditional static batching that waits for all sequences to complete, continuous batching keeps <span style="color:#E8600A;font-weight:700">GPU utilization (GPU利用率)</span> near 100% and forms the foundation of high-performance inference engines like <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">vLLM</code>, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">TGI</code>, and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">TensorRT-LLM</code>.
</div>

## 1. The Problem: Static Batching (静态批处理)

Traditional inference servers use <span style="color:#2980B9">static batching</span> – collecting requests until a batch is full, then processing them together.

```python
# Conceptual example of static batching
def static_batch_inference(requests):
    # Wait until we have 4 requests
    batch = wait_for_batch(size=4, requests=requests)
    
    # Process all 4 together
    results = model.generate(batch)
    
    # All requests finish at the SAME time
    # Even if some were much shorter!
    return results
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">The Restaurant Analogy: </span>Static batching is like a restaurant that <span style="color:#C0392B;font-weight:600">only seats new customers after EVERYONE at a table has finished eating</span>. Fast eaters wait idly while slow eaters finish – terrible utilization!</div>

### 1) Why Static Batching Wastes GPU Resources

```python
# Processing times vary wildly in real workloads
request_times = [2, 2, 10, 10]  # seconds

# Static batching timeline:
# Time 0-2:  2 short requests DONE, 2 long requests still running
# Time 2-10: GPU only working on 2 requests (50% utilization)
# Time 10:   All done, can start next batch

# Utilization calculation:
total_compute = 2 + 2 + 10 + 10 = 24 seconds of work
elapsed_time = 10 seconds
ideal_utilization = 24 / (4 * 10) = 60%  # Only 60%!
# GPU sits idle 40% of the time!
```

## 2. The Solution: Continuous Batching

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<span style="color:#E8600A;font-weight:700">Core idea: </span>Treat each request independently. When a request finishes, <span style="color:#2980B9">immediately evict it</span> and <span style="color:#2980B9">insert a new waiting request</span> into its slot. The batch is <span style="color:#E8600A;font-weight:700">continuously refilled</span>.
</div>

```python
# Conceptual example of continuous batching
def continuous_batching(scheduler, generator):
    # Initialize batch with some requests
    batch = scheduler.get_initial_batch()
    
    while has_pending_requests():
        # Run ONE iteration for ALL requests in batch
        for request in batch:
            if not request.finished:
                next_token = generator.generate_next_token(request)
                request.append_token(next_token)
        
        # Check for finished requests
        finished = [r for r in batch if r.finished]
        
        # Remove finished, add new ones IMMEDIATELY 
        for request in finished:
            batch.remove(request) # Dynamic Eviction (动态剔除)
            if scheduler.has_waiting():
                new_request = scheduler.get_next_waiting()
                batch.add(new_request)
        
        # Batch size stays constant – full utilization!
```

### 1) Iteration-Level Scheduling

```python
# Pseudo-code showing how continuous batching works at each step
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size=4):
        self.max_batch_size = max_batch_size
        self.active_requests = []  # Currently running
        self.waiting_queue = []     # Waiting to start
        
    def step(self):
        # Generate next token for all active requests
        for req in self.active_requests:
            if not req.finished:
                token = self.model.generate_next_token(req)
                req.tokens.append(token)
        
        # Remove finished requests
        self.active_requests = [r for r in self.active_requests if not r.finished]
        
        # Add new requests up to max_batch_size
        slots_available = self.max_batch_size - len(self.active_requests)
        for _ in range(min(slots_available, len(self.waiting_queue))):
            new_req = self.waiting_queue.pop(0)
            self.active_requests.append(new_req)
            
        # Batch is always full – maximum GPU utilization!
```

## 3. Performance Comparison

| Metric                                                       | Static Batching | Continuous Batching                                          | Improvement   |
| ------------------------------------------------------------ | --------------- | ------------------------------------------------------------ | ------------- |
| <span style="color:#2980B9">GPU Utilization (GPU利用率)</span> | 60-70%          | <span style="color:#E8600A;font-weight:700">95-99%</span>    | +40%          |
| <span style="color:#2980B9">Latency (延迟)</span>            | High variance   | <span style="color:#2980B9">More consistent</span>           | Better UX     |
| <span style="color:#2980B9">Throughput (吞吐量)</span>       | Baseline        | <span style="color:#E8600A;font-weight:700">2-3x higher</span> | 200-300%      |
| <span style="color:#2980B9">Queue time (排队时间)</span>     | Unpredictable   | <span style="color:#2980B9">Shorter</span>                   | 50% reduction |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Real-world numbers: </span>vLLM's implementation of continuous batching achieves <span style="color:#E8600A;font-weight:700">up to 23x higher throughput</span> than HuggingFace Transformers and <span style="color:#E8600A;font-weight:700">2.7x higher</span> than traditional static batching systems.</div>

## 4. Key Implementation Techniques

### 1) PagedAttention (vLLM)

```python
# Traditional KV cache: contiguous memory per request
# Problem: Memory fragmentation, can't easily add/remove

# PagedAttention: Like virtual memory for LLMs
class PagedAttention:
    def __init__(self):
        # KV cache divided into fixed-size blocks (pages)
        self.kv_blocks = []  # Physical blocks
        self.block_table = {}  # Maps request -> list of blocks
        
    def allocate_for_request(self, request):
        # Allocate physical blocks as needed
        blocks_needed = estimate_blocks(request)
        blocks = self.get_free_blocks(blocks_needed)
        self.block_table[request.id] = blocks
        
    def free_request(self, request):
        # Immediately free blocks when request finishes
        blocks = self.block_table.pop(request.id)
        self.free_blocks(blocks)  # Ready for new requests!
```

### 2) Iteration-Level Scheduling

```python
import asyncio
from typing import List, Optional

class Request:
    def __init__(self, prompt: str, max_tokens: int = 100):
        self.prompt = prompt
        self.generated_tokens: List[str] = []
        self.max_tokens = max_tokens
        self.finished = False
        
    def step(self, model) -> Optional[str]:
        """Generate one token, return None if finished"""
        if len(self.generated_tokens) >= self.max_tokens:
            self.finished = True
            return None
        
        next_token = model.generate_next(self.prompt, self.generated_tokens)
        self.generated_tokens.append(next_token)
        return next_token

class ContinuousBatchServer:
    def __init__(self, model, batch_size: int = 8):
        self.model = model
        self.batch_size = batch_size
        self.active: List[Request] = []
        self.queue: List[Request] = []
        
    async def infer_loop(self):
        """Main inference loop – runs continuously"""
        while True:
            # 1. Run one step for all active requests
            for req in self.active[:]:  # Copy list to avoid modification issues
                token = req.step(self.model)
                if token:
                    print(f"Generated: {token}")
            
            # 2. Remove finished requests
            self.active = [req for req in self.active if not req.finished]
            
            # 3. Add new requests up to batch_size
            slots = self.batch_size - len(self.active)
            for _ in range(min(slots, len(self.queue))):
                self.active.append(self.queue.pop(0))
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.01)
    
    async def submit_request(self, prompt: str) -> str:
        """API endpoint to submit a new request"""
        req = Request(prompt)
        self.queue.append(req)
        
        # Wait for completion (simplified)
        while not req.finished:
            await asyncio.sleep(0.1)
            
        return ''.join(req.generated_tokens)
```

## 5. Production Systems Using Continuous Batching

| System                                                       | Company      | Key Feature                               |
| ------------------------------------------------------------ | ------------ | ----------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vLLM</code> | UC Berkeley  | PagedAttention + Continuous Batching      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TGI (Text Generation Inference)</code> | Hugging Face | Production-ready, used by HuggingChat     |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TensorRT-LLM</code> | NVIDIA       | Optimized for NVIDIA GPUs                 |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Llama.cpp</code> | Community    | Continuous batching for consumer hardware |

## 6. Visual Timeline Comparison

```
Static Batching (Batch Size = 4):
Time ─────────────────────────────────────────────►
0    │ Request A (short)  │
     │ Request B (short)  │
     │ Request C (long)   │
     │ Request D (long)   │
2    │ Still waiting for C and D (2 requests idle)
     │ GPU only 50% utilized!
10   │ All done → Next batch starts
     │
     │ Utilization: ████████░░░░ 60%

Continuous Batching (Batch Size = 4):
Time ─────────────────────────────────────────────►
0    │ A │ B │ C │ D │
2    │ E │ F │ C │ D │  (A,B done, E,F start)
4    │ G │ H │ C │ D │  (E,F done, G,H start)
6    │ I │ J │ C │ D │  (G,H done, I,J start)
8    │ K │ L │ C │ D │  (I,J done, K,L start)
10   │ C,D finally done!
     │
     │ Utilization: ████████████ 98%
     
Throughput: 12 requests vs 4 requests in same time!
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Why it matters: </span>LLM inference is <span style="color:#2980B9">memory-bandwidth bound</span>, not compute-bound. Continuous batching keeps the memory pipeline full, extracting maximum throughput from expensive GPU hardware. For a $30,000 A100, <span style="color:#E8600A;font-weight:700">40% better utilization = $12,000 annual savings per GPU!</span></div>

## 7. Implementation Challenges

### 1) Memory Management

```python
# Challenge: KV cache grows with sequence length
class KVCacheManager:
    def __init__(self, total_memory: int, block_size: int = 16):
        self.total_memory = total_memory
        self.block_size = block_size
        self.free_blocks = list(range(total_memory // block_size))
        self.allocated = {}  # request_id -> list of blocks
        
    def allocate(self, request_id: str, num_tokens: int):
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < blocks_needed:
            # OOM! Need to preempt or wait
            raise MemoryError("Out of KV cache memory")
            
        blocks = self.free_blocks[:blocks_needed]
        self.free_blocks = self.free_blocks[blocks_needed:]
        self.allocated[request_id] = blocks
        return blocks
        
    def free(self, request_id: str):
        blocks = self.allocated.pop(request_id, [])
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()  # Optional, helps fragmentation
```

### 2) Scheduling Policy

```python
from enum import Enum
import heapq

class SchedulingPolicy(Enum):
    FCFS = "first_come_first_served"
    SJF = "shortest_job_first"  # If you can predict length
    FAIR = "fair_scheduling"

class Scheduler:
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.FCFS):
        self.policy = policy
        self.waiting = []  # Priority queue based on policy
        
    def add_request(self, request):
        if self.policy == SchedulingPolicy.FCFS:
            # Simple list, append at end
            self.waiting.append(request)
        elif self.policy == SchedulingPolicy.SJF:
            # Estimate based on prompt length
            priority = -len(request.prompt)  # Negative for min-heap
            heapq.heappush(self.waiting, (priority, request))
            
    def get_next(self):
        if self.policy == SchedulingPolicy.FCFS:
            return self.waiting.pop(0) if self.waiting else None
        elif self.policy == SchedulingPolicy.SJF:
            return heapq.heappop(self.waiting)[1] if self.waiting else None
```

## 8. Advanced Topics

### 1) Preemption and Migration

```python
# When memory is full, preempt longest-running request
class PreemptiveScheduler(ContinuousBatchScheduler):
    def handle_oom(self, new_request):
        # Find request with most generated tokens (largest KV cache)
        longest = max(self.active_requests, 
                     key=lambda r: len(r.generated_tokens))
        
        # Swap out – save state to CPU memory
        self.swapped[longest.id] = {
            'prompt': longest.prompt,
            'tokens': longest.generated_tokens,
            'position': len(longest.generated_tokens)
        }
        
        # Free GPU memory
        self.active_requests.remove(longest)
        self.kv_cache.free(longest.id)
        
        # Add new request
        self.active_requests.append(new_request)
        
    def resume_swapped(self):
        # When slots available, resume preempted requests
        if self.swapped and len(self.active_requests) < self.max_batch_size:
            req_id, state = self.swapped.popitem()
            # Restore from CPU to GPU
            restored = Request(state['prompt'])
            restored.generated_tokens = state['tokens']
            self.active_requests.append(restored)
```

### 2) Dynamic Batching with Speculative Decoding

```python
# Combine continuous batching with speculative execution
class SpeculativeContinuousBatch:
    def __init__(self, target_model, draft_model):
        self.target = target_model
        self.draft = draft_model  # Smaller, faster model
        self.batch = []
        
    def step(self):
        # Draft model proposes next 5 tokens for all requests
        proposals = [self.draft.speculate(req, k=5) for req in self.batch]
        
        # Target model verifies all proposals in parallel
        verified = self.target.verify(proposals)
        
        # Update each request with verified tokens
        for req, ver in zip(self.batch, verified):
            for token in ver.accepted_tokens:
                req.generated_tokens.append(token)
                
        # Continuous batching still applies!
        self.refill_batch()
```

## 9. Performance Tuning Guidelines

```python
# Key parameters to tune
class ContinuousBatchConfig:
    def __init__(self):
        # Max batch size – larger = higher throughput, higher latency
        self.max_batch_size: int = 8  # Start here, increase while memory allows
        
        # Block size for KV cache – smaller = more flexible, more overhead
        self.kv_block_size: int = 16  # 16 tokens per block
        
        # Queue size – balance memory vs. throughput
        self.max_queue_size: int = 1000
        
        # Scheduling policy – affects fairness and latency
        self.scheduling_policy: str = "fcfs"  # or "sjf", "fair"
        
        # Preemption – enable for high load
        self.enable_preemption: bool = True
        
        # Swap space – CPU memory for preempted requests
        self.swap_space_gb: float = 10.0
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br><span style="color:#E8600A;font-weight:700">Continuous Batching (连续批处理)</span> eliminates GPU idle time by <span style="color:#2980B9">dynamically replacing finished requests with new ones</span> – like a fast-food counter instead of a sit-down restaurant, delivering <span style="color:#E8600A;font-weight:700">2-3x higher throughput</span> for LLM serving.</div>
