---
title: warp
published: 2025-12-18
description: "warp"
image: "./cover.jpeg"
tags: ["Blogging", "warp"]
category: Guides
draft: false

---

# Warp Group

## 1. 定义与基本概念

Warp group 是 NVIDIA Hopper 架构中引入的一种硬件执行单元。
它不是 CUDA 语言层面的语法概念。
Warp group 由固定数量的 warp 组成。
一个 warp group 恰好包含四个 warp。
每个 warp 包含 32 个线程。
因此，一个 warp group 一共包含 128 个线程。
Warp group 的大小是硬件固定的，不能改变。

A warp group is a hardware execution unit introduced in the NVIDIA Hopper architecture.
It is not a CUDA language-level concept.
A warp group consists of a fixed number of warps.
Each warp group contains exactly four warps.
Each warp has 32 threads.
Therefore, a warp group has a total of 128 threads.
The size of a warp group is fixed by hardware.

------

## 2. 设计动机与存在原因

传统的 warp 级执行粒度不适合更大的矩阵计算。
Hopper 引入了新的大规模矩阵指令。
这些指令一次计算很大的矩阵 tile。
典型的例子是 64x64 的矩阵计算。
单个 warp 无法提供足够的线程资源。
多个 warp 必须协同完成一次指令。
Warp group 提供了这种协同执行的最小硬件单位。
它的目标是更高的 Tensor Core 利用率。

Traditional warp-level execution is not suitable for large matrix computations.
Hopper introduces new large-scale matrix instructions.
These instructions compute large matrix tiles in one operation.
A typical example is a 64x64 matrix tile.
A single warp does not provide enough threads.
Multiple warps must cooperate to execute one instruction.
Warp groups provide the minimal hardware unit for this cooperation.
The goal is higher Tensor Core utilization.

------

## 3. 与 Thread、Warp 和 Block 的关系

Thread 是 CUDA 中最小的执行单元。
Warp 是由 32 个线程组成的调度单元。
Warp 内的线程以锁步方式执行。
Warp group 是由四个 warp 组成的执行单元。
Thread block 是程序员定义的线程集合。
一个 block 可以包含一个或多个 warp group。
Warp group 不能跨越 block 边界。
在很多 Hopper kernel 中，一个 block 对应一个 warp group。

A thread is the smallest execution unit in CUDA.
A warp is a scheduling unit composed of 32 threads.
Threads inside a warp execute in lockstep.
A warp group is an execution unit composed of four warps.
A thread block is a programmer-defined group of threads.
A block may contain one or more warp groups.
A warp group cannot cross block boundaries.
In many Hopper kernels, one block maps to one warp group.

------

## 4. 实际使用方式与代码示例

Warp group 主要用于执行 WGMMA 指令。
WGMMA 指令要求完整的 warp group。
使用 WGMMA 时，block 的线程数必须是 128 的倍数。
最简单的方式是一个 block 使用一个 warp group。
这种方式最容易保证正确性。
Warp group 计算需要显式同步指令。
这些指令由内联 PTX 汇编提供。

Warp groups are mainly used to execute WGMMA instructions.
WGMMA instructions require a full warp group.
When using WGMMA, the block size must be a multiple of 128 threads.
The simplest approach is one warp group per block.
This approach is the easiest to make correct.
Warp group computation requires explicit synchronization.
These instructions are provided through inline PTX assembly.

示例：Block 配置方式

```cpp
constexpr int NUM_THREADS = 128;
kernel<<<gridDim, NUM_THREADS>>>(...);
```

示例：Warp group 执行指令顺序

```cpp
asm volatile("wgmma.fence.sync.aligned;");
asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 ...;");
asm volatile("wgmma.commit_group.sync.aligned;");
asm volatile("wgmma.wait_group.sync.aligned 0;");
```

示例：Warp group 指令顺序含义

Fence 用于同步 warp group 内的所有 warp。
MMA 指令执行矩阵乘加计算。
Commit 表示一组 WGMMA 指令结束。
Wait 确保计算完成后再使用结果。

Example: Meaning of the warp group instruction sequence

Fence synchronizes all warps in the warp group.
The MMA instruction performs matrix multiply-accumulate.
Commit marks the end of a WGMMA instruction batch.
Wait ensures computation completes before using results.



