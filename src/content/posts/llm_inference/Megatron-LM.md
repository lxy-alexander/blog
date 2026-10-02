---
title: "Megatron-LM"
published: 2026-09-24
description: "Megatron-LM"
image: ""
tags: ["llm_inference","Megatron-LM"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-24T16:15:28.857.497815647Z"
---



# CUDA_DEVICE_MAX_CONNECTIONS

>`CUDA_DEVICE_MAX_CONNECTIONS` 控制的是“CPU/CUDA 驱动向单张 GPU 派发计算和拷贝任务的硬件队列数量”。
>
>### 软件层：CUDA Stream
>
>程序可以创建很多 CUDA stream：
>
>```
>Stream A：矩阵计算
>Stream B：NCCL 通信
>Stream C：内存复制
>```
>
>它们只是程序表达任务顺序的逻辑概念。
>
>### 硬件层：Work Queue
>
>CUDA Driver 会把这些 stream 中的命令，映射到 GPU 能直接接收的工作队列：
>
>```
>CUDA Streams（软件逻辑）
>        │
>        ▼
>Work Queues（驱动和 GPU 执行前端维护）
>        │
>        ▼
>GPU 执行 kernel / 内存复制
>```
>
>工作队列里放的是命令，例如：
>
>```
>启动矩阵乘法 kernel
>启动 NCCL all-reduce kernel
>复制一块内存
>等待某个 CUDA event
>```

`CUDA_DEVICE_MAX_CONNECTIONS` 控制的是：一个 CUDA context 可以使用多少条 GPU 硬件工作队列，用来提交计算 kernel 和内存拷贝。取值 1–32，默认通常是 8。[NVIDIA CUDA 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)

它不是 GPU 数量，也不是 GPU 之间的 NCCL/NVLink 连接数。

| 模式                        | 设置     | 原因                                                         |
| --------------------------- | -------- | ------------------------------------------------------------ |
| A40 上 TP/CP、非 FSDP       | `1`      | 强制通信和计算按提交顺序调度，让异步 all-reduce 先启动，再与 GEMM 重叠 |
| Torch-FSDP2 / Megatron-FSDP | 取消设置 | 需要多条独立队列，让参数 all-gather、梯度 reduce-scatter 和计算真正并发 |

在普通 TP 中，Megatron 会先提交通信，再提交计算：

```
通信 all-reduce ──────────┐
计算 GEMM          ──────────
                         重叠
```

设为 `1` 后，共用一条工作队列，可以保证通信先被调度。Megatron 源码明确说明：这是为了性能，不是正确性。[TP 实现 (line 943)](/data/home/xli49/lxy/Megatron-LM/megatron/core/tensor_parallel/layers.py:943)

FSDP 的工作方式不同。它不断执行：

```
参数 all-gather → 当前层计算 → 梯度 reduce-scatter
```

并尝试在不同 CUDA streams 上把下一层通信和当前层计算重叠。如果设置为 `1`，这些 streams 会挤到同一条硬件队列，产生不必要的顺序依赖，通信和计算可能被串行化。因此 FSDP 需要多条独立队列。

所以：

```
unset CUDA_DEVICE_MAX_CONNECTIONS
```

表示删除这个限制，让 CUDA 使用默认队列数量。Megatron 甚至会直接拒绝 FSDP + `CUDA_DEVICE_MAX_CONNECTIONS=1`：

-   [Torch-FSDP2 检查 (line 1061)](/data/home/xli49/lxy/Megatron-LM/megatron/training/arguments.py:1061)
-   [Megatron-FSDP 检查 (line 1185)](/data/home/xli49/lxy/Megatron-LM/megatron/training/arguments.py:1185)

你的当前 A40 + TP=2 示例使用：

```
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun ...
```

以后使用 `--use-torch-fsdp2` 或 `--use-megatron-fsdp` 时则执行：

```
unset CUDA_DEVICE_MAX_CONNECTIONS
```

### 原理

可以把底层分成 5 层：

```
PyTorch / NCCL 操作
        ↓ 提交命令
CUDA Stream（逻辑命令流）
        ↓ 驱动映射
Connection / 硬件工作队列
        ↓ GPU 前端调度
SM 计算单元 / Copy Engine
```

### 1. CUDA Stream 是逻辑队列

例如 Megatron 发出：

```
通信 Stream：NCCL all-reduce
计算 Stream：GEMM
```

同一个 Stream 内严格按顺序执行；不同 Stream 默认可以并行。真正存在依赖时，通过 CUDA Event 表示：

```
all-gather(W2) ──event──> GEMM2
GEMM1          ─────────> GEMM2
```

所以虽然第二层计算依赖第一层输出，但“提前取得第二层参数”不依赖第一层计算，可以与第一层计算重叠。

### 2. Connection 是什么

这里的 `connection` 不是：

-   GPU 间的 NVLink/PCIe 连接
-   NCCL 网络连接
-   进程数量
-   物理电线

它更接近“CUDA 驱动向 GPU 提交工作的硬件工作队列通道”。

多个逻辑 CUDA Stream 最终需要映射到有限数量的硬件工作队列：

```
Stream A ─┐
Stream B ─┼──> Connection 0 ──> GPU
Stream C ─┘

Stream D ─────> Connection 1 ──> GPU
```

GPU 前端再将计算 kernel 分派给 SM，将复制任务分派给 Copy Engine。其精确物理实现属于 NVIDIA 内部细节，不能简单理解成“一条 connection 就是一根物理 FIFO”。

### 3. `CUDA_DEVICE_MAX_CONNECTIONS=1` 做了什么

它把设备侧可用的计算、复制工作队列数量限制为 1：

```
通信 Stream ─┐
计算 Stream ─┼──> 唯一工作队列 ──> GPU
复制 Stream ─┘
```

这会强化不同 Stream 之间的提交/调度顺序，使 GPU 更接近按照 CPU 发出命令的先后顺序接收工作。

但它不等于：

```
CUDA_LAUNCH_BLOCKING=1
```

CPU 提交仍然是异步的；也不意味着 GPU 上的通信和计算一定完全串行。一旦前面的 NCCL kernel 已经被调度进去，后面的 GEMM 仍可能利用不同资源与它重叠。它主要限制的是任务进入 GPU 的调度通道。

### 4. 为什么 Megatron TP 喜欢设为 1

Megatron 的 Tensor Parallel 经常按照下面的顺序提交：

```
先提交：NCCL 通信
再提交：GEMM 计算
```

它希望：

```
NCCL 先启动
    ↘ 与 GEMM 重叠
      GEMM 后启动
```

如果存在多个硬件队列，通信和计算处于不同 Stream/队列，GPU 可能先调度 GEMM，导致 NCCL 启动太晚，通信无法很好地隐藏在计算后面。

设置为 1 后：

```
NCCL → GEMM
```

更容易保证通信先进入执行阶段。因此 Megatron 源码明确说明：这是为了获得 TP 通信重叠的性能收益，不是为了保证计算正确性：[layers.py (line 943)](/data/home/xli49/lxy/Megatron-LM/megatron/core/tensor_parallel/layers.py:943)。

### 5. 为什么 FSDP 要取消

FSDP 的设计恰好依赖多个独立 Stream 并行：

```
当前层计算：       GEMM layer 1
下一层参数预取：   all-gather weights layer 2
梯度通信：         reduce-scatter gradients
```

理想状态：

```
时间 ─────────────────────────>

计算：    [ GEMM layer 1 ][ GEMM layer 2 ]
通信：       [AG weights 2]   [RS grads 1]
```

如果设成一个工作队列，本来没有依赖的操作也可能因为共享队列出现“假依赖”：

```
GEMM1 → AG(W2) → GEMM2
```

从而破坏 FSDP 的预取和通信/计算流水线。因此 FSDP 要：

```
unset CUDA_DEVICE_MAX_CONNECTIONS
```

恢复多个队列，让真正的依赖由 CUDA Event 控制，其他工作尽量并行。Megatron 对 Torch-FSDP2 和 Megatron-FSDP 都明确拒绝使用值 `1`：[arguments.py (line 1061)](/data/home/xli49/lxy/Megatron-LM/megatron/training/arguments.py:1061)。

一句话总结：

>   `CUDA_DEVICE_MAX_CONNECTIONS=1` 不是限制 GPU 之间的连接，而是限制 CUDA Stream 映射到 GPU 的工作提交队列；TP 用它强化“通信先于计算”的调度顺序，FSDP 则需要多个队列实现参数预取、计算和通信并行。

CUDA 官方也说明，这个变量控制计算和复制引擎的并发连接数；默认通常为 8，可设置为 1–32：[CUDA 环境变量文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#cuda-device-max-connections)。
