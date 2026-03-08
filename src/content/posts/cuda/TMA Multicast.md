---
title: "TMA Multicast"
published: 2026-02-20
description: "TMA Multicast"
image: ""
tags: ["cuda","TMA Multicast"]
category: cuda
draft: false
lang: ""
---

# **I. Hopper TMA Multicast (跨 SM 广播加载)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
NVIDIA Hopper introduces the concept of a <span style="color:#E8600A;font-weight:700">Thread Block Cluster (线程块集群)</span> — multiple thread blocks across different SMs that can cooperate. Combined with <span style="color:#E8600A;font-weight:700">TMA Multicast (张量内存加速广播)</span>, a single global memory load can be broadcast to multiple SMs simultaneously, dramatically improving <span style="color:#2980B9">Bandwidth Utilization (带宽利用率)</span>.
</div>

---

## <span style="color:#E8600A">1.</span> **The Problem: Redundant Loads (问题：重复加载)**

Without TMA Multicast, each SM independently fetches the same data:

```
SM0 needs tile A → reads from L2
SM1 needs tile A → reads from L2 again
```

<span style="color:#C0392B;font-weight:600">Costs two L2 bandwidth reads for the same data.</span> With many SMs needing the same tile, this multiplies:

- <span style="color:#C0392B;font-weight:600">Wasted L2 bandwidth (浪费 L2 带宽)</span>
- <span style="color:#C0392B;font-weight:600">Increased latency (增加延迟)</span>

---

## <span style="color:#E8600A">2.</span> **Hopper's Solution: TMA Multicast (解决方案)**

TMA Multicast loads a tile **once** and broadcasts it to all SMs in the cluster:

```
Global Memory
↓
L2 Cache (read once)
↓ (broadcast / 广播)
SM0 Shared Memory
SM1 Shared Memory
SM2 Shared Memory
...
```

### 1) Key properties (核心特性)

- <span style="color:#E8600A;font-weight:700">Load once, deliver to many (一次加载，多处送达)</span>
- Destination is directly <span style="color:#2980B9">Shared Memory (共享内存)</span> — bypasses registers
- Enabled by the <span style="color:#E8600A;font-weight:700">Thread Block Cluster (线程块集群)</span> abstraction introduced in Hopper

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><span style="color:#E8600A;font-weight:700">TMA (Tensor Memory Accelerator, 张量内存加速器)</span> is a hardware unit in Hopper that handles async, bulk data movement between global memory and shared memory — offloading this work from CUDA cores entirely.
</div>

---

## <span style="color:#E8600A">3.</span> **Normal vs Multicast (普通加载 vs 广播对比)**

| | Normal Load (普通加载) | TMA Multicast (广播加载) |
|---|---|---|
| L2 reads for N SMs | N times | <span style="color:#E8600A;font-weight:700">1 time</span> |
| Bandwidth usage | <span style="color:#C0392B;font-weight:600">N × tile size</span> | <span style="color:#E8600A;font-weight:700">1 × tile size</span> |
| Latency | High (repeated) | <span style="color:#E8600A;font-weight:700">Low (single fetch)</span> |
| Requires | Any GPU | <span style="color:#2980B9">Hopper (sm_90) + Cluster</span> |

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<span style="color:#E8600A;font-weight:700">TMA Multicast (广播加载)</span> reads a tile from global memory <span style="color:#C0392B;font-weight:600">once</span> via L2 and simultaneously delivers it to multiple SMs' shared memory — turning N redundant reads into 1.
</div>