---
title: "Performance Analysis"
published: 2026-04-27
description: "Performance Analysis"
image: ""
tags: ["cuda","Performance Analysis"]
category: cuda
draft: false
lang: ""
createdAt: "2026-04-28T03:22:52.726.325971030Z"
---

# Profiling Metrics and Their Tools (性能指标与对应工具命令)

Goal: know exactly **which command** measures **which metric** so you can profile (分析) without guessing.

## 1. Quick Reference Table (快速查表)

This table maps each metric (指标) to its primary measurement command (主要测量命令).

| Metric (指标)                            | Tool (工具)                  | Command Snippet (命令)                                       |
| ---------------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| Kernel Time (耗时)                       | CUDA Events / Nsight Systems | `nsys profile --stats=true ./app`                            |
| Memory Throughput (内存吞吐)             | Nsight Compute               | `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./app` |
| Achieved Occupancy (实际占用率)          | Nsight Compute               | `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app` |
| Warp Execution Efficiency (Warp 效率)    | Nsight Compute               | `ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./app` |
| Global Load Efficiency (全局加载效率)    | Nsight Compute               | `ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./app` |
| Bank Conflicts (Bank 冲突)               | Nsight Compute               | `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./app` |
| Register Pressure (寄存器压力)           | nvcc / Nsight Compute        | `nvcc -Xptxas -v kernel.cu`                                  |
| SM Utilization (SM 利用率)               | Nsight Compute               | `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./app` |
| Tensor Core Utilization (张量核心利用率) | Nsight Compute               | `ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./app` |

<br>

## 2. Detailed Commands (详细命令)

### 1) Kernel Time (内核耗时)

CUDA Events (CUDA 事件) measure inside code; Nsight Systems (系统级分析器) measures from outside.

```bash
# Method A: Nsight Systems — timeline summary (时间线汇总)
nsys profile --stats=true -o report ./app
# Output shows: kernel name | time | calls | avg | min | max

# Method B: Nsight Compute — per-kernel duration (每内核耗时)
ncu --metrics gpu__time_duration.sum ./app
// Method C: in-code CUDA Events (代码内计时)
cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s);
my_kernel<<<grid, block>>>(...);
cudaEventRecord(e); cudaEventSynchronize(e);
float ms; cudaEventElapsedTime(&ms, s, e);
printf("kernel time = %.3f ms\n", ms);
```

### 2) Memory Throughput (内存吞吐)

This measures DRAM bandwidth (DRAM 带宽) as a percentage of peak (峰值百分比).

```bash
# Achieved DRAM bandwidth (实际 DRAM 带宽)
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./app

# Absolute bytes read/written (读写绝对字节数)
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum ./app

# Quick preset (快捷预设): MemoryWorkloadAnalysis section
ncu --section MemoryWorkloadAnalysis ./app
```

### 3) Achieved Occupancy (实际占用率)

Achieved Occupancy (实际占用率) = active warps (活跃 warp) / max warps per SM (每 SM 最大 warp 数).

```bash
# Achieved occupancy (实际占用率)
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app

# Theoretical vs achieved (理论与实际对比)
ncu --section Occupancy ./app
```

### 4) Warp Execution Efficiency (Warp 执行效率)

Measures the average active lanes (活跃通道) per warp instruction — low value indicates divergence (分支发散).

```bash
# Average active threads per executed instruction (每条指令的平均活跃线程数)
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./app

# Branch divergence detail (分支发散细节)
ncu --section SchedulerStats --section WarpStateStats ./app
```

### 5) Global Load Efficiency (全局加载效率)

Measures the ratio of useful bytes (有效字节) to actually transferred bytes (实际传输字节) — low value means uncoalesced access (非合并访问).

```bash
# Bytes per sector for global loads (全局加载每扇区字节)
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./app

# L1/TEX hit rate and global access pattern (L1/TEX 命中率)
ncu --section MemoryWorkloadAnalysis_Tables ./app
```

### 6) Shared Memory Bank Conflicts (共享内存 Bank 冲突)

Counts the number of bank-conflict (Bank 冲突) events on shared-memory loads/stores (共享内存读写).

```bash
# Bank conflicts on shared loads (共享内存读 bank 冲突)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./app

# Bank conflicts on shared stores (共享内存写 bank 冲突)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./app

# Both at once (同时查看读写)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./app
```

### 7) Register Pressure (寄存器压力)

Compile-time (编译期) check via `nvcc`; runtime (运行期) check via Nsight Compute (内核级分析器).

```bash
# Compile-time: registers per thread (每线程寄存器数)
nvcc -Xptxas -v kernel.cu -o app
# Output: "Used 32 registers, 0 bytes smem, ..." (输出寄存器/共享内存用量)

# Runtime: registers per thread + spills (寄存器 + 溢出)
ncu --metrics launch__registers_per_thread,\
smsp__sass_thread_inst_executed_op_local_ld.sum,\
smsp__sass_thread_inst_executed_op_local_st.sum ./app
```

### 8) SM Utilization (SM 利用率)

Measures how busy the streaming multiprocessor (流式多处理器) was — low value means GPU idle (GPU 空闲).

```bash
# SM throughput as % of peak (SM 吞吐占峰值百分比)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./app

# Compute pipeline utilization (计算管线利用率)
ncu --section ComputeWorkloadAnalysis ./app
```

### 9) Tensor Core Utilization (张量核心利用率)

Tensor Cores (张量核心) accelerate FP16/BF16/INT8 matrix multiply (矩阵乘法).

```bash
# Tensor core active cycles (张量核心活跃周期)
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./app

# All pipe utilizations (所有管线利用率)
ncu --metrics sm__inst_executed_pipe_tensor.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active ./app
```

<br>

## 3. One-Shot Full Profile (一键完整分析)

When you want everything at once (一次拿到所有指标), use Nsight Compute's `--set full` (完整集合).

```bash
# Full profile (full set, slowest but most complete 最完整)
ncu --set full -o report ./app

# Detailed sections only (仅详细分析章节)
ncu --set detailed ./app

# Target a specific kernel (只分析指定内核)
ncu -k reduce_warp --set full ./app

# Limit to first N launches (仅前 N 次启动,加速)
ncu -k reduce_warp -c 5 --set full ./app
```

<br>

## 4. Decision Flow (诊断流程)

Follow this order when analyzing a slow kernel (慢内核) — each step picks one or two metrics (指标) to check.

### 1) Step 1: Identify the Bottleneck Kernel (定位瓶颈内核)

```bash
nsys profile --stats=true ./app    # See which kernel dominates total time
```

### 2) Step 2: Memory-Bound or Compute-Bound? (内存受限还是计算受限?)

```bash
ncu --section SpeedOfLight ./app   # Compares Memory % vs Compute %
```

### 3) Step 3: If Memory-Bound, Check Coalescing and Bank Conflicts (内存受限则检查合并访问与 bank 冲突)

```bash
ncu --section MemoryWorkloadAnalysis ./app
```

### 4) Step 4: If Compute-Bound, Check Warp Efficiency and Divergence (计算受限则检查 warp 效率与发散)

```bash
ncu --section WarpStateStats --section SchedulerStats ./app
```

### 5) Step 5: Check Occupancy and Register Spills (检查占用率与寄存器溢出)

```bash
ncu --section Occupancy --section LaunchStats ./app
```

<br>

## 5. Stage Outcome (阶段成果)

You should now have a **fixed mental command list** (固定的命令清单) — given any metric (指标), you can immediately type the right `ncu` or `nsys` command without looking it up.

The key insight (核心洞察) is that profiling is a **lookup table problem** (查表问题): metric → command → number → next step. Memorize the table (记住表格), and CUDA performance debugging becomes mechanical (机械化) rather than mystical (玄学).

<br>
