---
title: "Memory Hierarchy"
published: 2026-02-23
description: "Memory Hierarchy"
image: ""
tags: ["cuda","Memory Hierarchy"]
category: cuda
draft: false
lang: ""
---


# **I. GPU Architecture (GPU 架构层级与容量)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
A GPU consists of multiple <span style="color:#E8600A;font-weight:700">Streaming Multiprocessors (流式多处理器, SM)</span>, each with its own on-chip memory (Register, Shared Memory, L1, Constant Cache). Outside the SMs sit a device-wide <span style="color:#E8600A;font-weight:700">L2 Cache (二级缓存)</span> and <span style="color:#E8600A;font-weight:700">DRAM / Global Memory (全局内存)</span>. Capacity shrinks dramatically as you move closer to the compute cores — but speed increases just as dramatically.
</div>

---

## <span style="color:#E8600A">1.</span> **Hierarchy Overview (层级总览)**

```
GPU
├── L2 Cache (device-wide, hardware-managed)
│   └── Size: 4MB – 96MB
│
├── DRAM / Global Memory (device-wide)
│   ├── Size: 8GB – 80GB+
│   └── Constant Memory region (64KB, read-only)
│
└── Multiple SMs (Streaming Multiprocessors)
    ├── Compute Units
    │   ├── CUDA Cores
    │   └── Tensor Cores
    │
    ├── On-chip Memory (SM-local)
    │   ├── Register File   (per-thread)
    │   ├── Shared Memory   (per-block)
    │   └── L1 Cache        (hardware-managed)
    │
    └── Constant Cache (SM-local, read-only)
```

---

## <span style="color:#E8600A">2.</span> **SM (Streaming Multiprocessor, 流式多处理器)**

- A GPU is composed of <span style="color:#2980B9">many independent SMs</span>
- Each SM independently executes <span style="color:#E8600A;font-weight:700">Thread Blocks (线程块)</span>
- Each SM is a combination of compute units + on-chip storage

Typical SM config (Ampere architecture):

| Memory | Typical Size |
|---|---|
| Register File | **256 KB / SM** |
| Shared Memory | **64 KB – 164 KB / SM** |
| L1 Cache | **24 KB – 128 KB / SM** |
| Constant Cache | **8 KB / SM** |

---

## <span style="color:#E8600A">3.</span> **On-chip Memory Detail (SM 内部存储详解)**

### 1) Register File (寄存器文件) — 256 KB / SM

<span style="color:#2980B9">Private per-thread (每线程私有)</span>. Fastest memory on the GPU.

```
256 KB = 262,144 bytes → int: 65,536 / float: 65,536
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Max <span style="color:#C0392B;font-weight:600">255 registers per thread</span>. All threads in the SM physically share this 256 KB pool — more threads = fewer registers each.
</div>

### 2) Shared Memory (共享内存) — ~100 KB / SM

<span style="color:#2980B9">Shared within a thread block (块内共享)</span>. Programmer-managed.

```
100 KB = 102,400 bytes → int: 25,600 / float: 25,600
164 KB = 167,936 bytes → int: 41,984 / float: 41,984
```

### 3) L1 Cache (一级缓存) — ~128 KB / SM

Hardware-managed (硬件自动管理). Caches global memory accesses.

```
128 KB = 131,072 bytes → int: 32,768 / float: 32,768
```

### 4) Constant Cache (常量缓存) — 8 KB / SM

Read-only (只读). Accelerates access to the 64 KB Constant Memory region.

```
8 KB = 8,192 bytes → int: 2,048 / float: 2,048
```

---

## <span style="color:#E8600A">4.</span> **Off-chip Memory (SM 外部存储)**

### 1) L2 Cache (二级缓存) — 6 MB – 40 MB, device-wide

```
 6 MB =  6,291,456 bytes → int:  1,572,864
40 MB = 41,943,040 bytes → int: 10,485,760
```

### 2) DRAM / Global Memory (全局内存) — 8 GB – 80 GB

```
16 GB = 17,179,869,184 bytes → int:  4,294,967,296
80 GB = 85,899,345,920 bytes → int: 21,474,836,480
```

### 3) Constant Memory Region (常量内存区域) — fixed 64 KB

```
64 KB = 65,536 bytes → int: 16,384 / float: 16,384
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Only <span style="color:#C0392B;font-weight:600">64 KB total</span> for the entire GPU. Accessed via the 8 KB per-SM Constant Cache — fast if all threads read the same address (broadcast, 广播访问).
</div>

---

## <span style="color:#E8600A">5.</span> **Memory Access Path (内存层级访问路径)**

### 1) Accessing Global Memory (访问全局内存)

```
Register (256 KB / SM)
↓
L1 Cache (≤128 KB / SM)
↓
L2 Cache (MB-level, GPU-wide)
↓
DRAM (GB-level)
```

### 2) Accessing Constant Memory (访问常量内存)

```
Register
↓
Constant Cache (8 KB / SM)
↓
L2 Cache
↓
64 KB Constant Region (in DRAM)
```

---

## <span style="color:#E8600A">6.</span> **Capacity Comparison (容量数量级对比)**

| Memory Level | Typical Size | int/float Count |
|---|---|---|
| Register (per SM) | 256 KB | ~65K |
| Shared Memory | 100 KB | ~25K |
| L1 Cache | 128 KB | ~32K |
| Constant Cache | 8 KB | ~2K |
| Constant Memory | 64 KB | ~16K |
| L2 Cache | 6 – 40 MB | 百万级 |
| DRAM | 8 – 80 GB | 十亿级 |

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Each SM has ~256 KB Register, ~100 KB Shared Memory, ~128 KB L1; outside the SM sit MB-level L2 and GB-level DRAM; <span style="color:#E8600A;font-weight:700">Constant Memory (常量内存)</span> is fixed at 64 KB GPU-wide and accelerated by an 8 KB per-SM <span style="color:#E8600A;font-weight:700">Constant Cache (常量缓存)</span>.
</div>