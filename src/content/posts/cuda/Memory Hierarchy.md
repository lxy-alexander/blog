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



# GPU Architecture（层级关系 + 容量）

![image-20260223222603530](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260223222603530)

```
GPU
├── L2 Cache (device-wide, hardware-managed)
│   └── Size: 4MB – 96MB (架构相关)
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
    │   ├── Register (per-thread)
    │   ├── Shared Memory (per-block)
    │   └── L1 Cache (hardware-managed)
    │
    └── Constant Cache (SM-local)
```

------

# 分层容量说明（从内到外）

------

## SM（Streaming Multiprocessor）

-   GPU 由多个 SM 组成 

*   每个 SM 独立执行 thread blocks 
*   SM 是“计算 + 片上存储”的组合体



一个现代 SM（Ampere）典型配置：

-   Register File：**256 KB / SM**
-   Shared Memory：**64 KB – 164 KB / SM**
-   L1 Cache：**24 KB – 128 KB / SM**
-   Constant Cache：**8 KB / SM**

------

# SM 内部存储容量

------

## Register（每 SM 256 KB）

每线程私有。

```
256 KB = 262,144 bytes
```

可存储：

```
int   : 65,536
float : 65,536
```

⚠️ 注意：

-   每线程最多 255 registers
-   物理上所有线程共享这 256KB



------

## Shared Memory（典型 100 KB）

以 100 KB 举例：

```
100 KB = 102,400 bytes
```

可存储：

```
int   : 25,600
float : 25,600
```

如果是 164 KB：

```
164 KB = 167,936 bytes
```

可存储：

```
int   : 41,984
float : 41,984
```

------

## L1 Cache（典型 128 KB）

```
128 KB = 131,072 bytes
```

可缓存：

```
int   : 32,768
float : 32,768
```

（自动管理）

------

## Constant Cache（每 SM 8 KB）

每 SM 有 * 专门缓存 constant memory * 只读

```
8 KB = 8,192 bytes
```

可缓存：

```
int   : 2,048
float : 2,048
```

------

# 3️⃣ SM 外部结构

------

## L2 Cache（假设 6 MB）

```
6 MB = 6,291,456 bytes
```

可缓存：

```
int   : 1,572,864
float : 1,572,864
```

如果是 40 MB：

```
40 MB = 41,943,040 bytes
```

可缓存：

```
int   : 10,485,760
float : 10,485,760
```

------

## DRAM / Global Memory

举例 16 GB GPU：

```
16 GB = 17,179,869,184 bytes
```

可存储：

```
int   : 4,294,967,296
float : 4,294,967,296
```

80 GB GPU：

```
80 GB ≈ 85,899,345,920 bytes
```

可存储：

```
int   : 21,474,836,480
float : 21,474,836,480
```

------

## Constant Memory Region（固定 64 KB）

```
64 KB = 65,536 bytes
```

可存储：

```
int   : 16,384
float : 16,384
```

⚠️ 注意：

-   整个 GPU 只有 64KB
-   通过 Constant Cache 加速

------

# 内存层级路径（容量视角）

访问 global memory：

```
Register (256 KB / SM)
↓
L1 (≤128 KB / SM)
↓
L2 (MB 级，GPU-wide)
↓
DRAM (GB 级)
```

访问 constant memory：

```
Register
↓
Constant Cache (8 KB / SM)
↓
L2
↓
64 KB Constant Region (in DRAM)
```

------

# 容量级别对比（数量级）

| 层级              | 典型大小 | 可存 int/float 数量 |
| ----------------- | -------- | ------------------- |
| Register (per SM) | 256 KB   | 65K                 |
| Shared Memory     | 100 KB   | 25K                 |
| L1 Cache          | 128 KB   | 32K                 |
| Constant Cache    | 8 KB     | 2K                  |
| Constant Memory   | 64 KB    | 16K                 |
| L2 Cache          | 6–40 MB  | 百万级              |
| DRAM              | 8–80 GB  | 十亿级              |

------

# 最终层级关系一句话总结（含容量）

>   GPU 由多个 SM 组成；每个 SM 约有 256KB Register、~100KB Shared Memory、~128KB L1；SM 外有 MB 级 L2 和 GB 级 DRAM；Constant Memory 固定 64KB，可存 16K 个 int/float，通过 8KB Constant Cache 加速访问。
