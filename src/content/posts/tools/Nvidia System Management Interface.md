---
title: "Nvidia System Management Interface"
published: 2026-02-11
description: "Nvidia System Management Interface"
image: ""
tags: ["tools","Nvidia System Management Interface"]
category: tools
draft: false
lang: ""
---

# **I. `nvidia-smi` — NVIDIA System Management Interface**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">nvidia-smi</code> is the <strong>NVIDIA System Management Interface (系统管理接口)</strong>. It is used to monitor and manage GPU device status, including GPU memory usage, GPU utilization, temperature, and power consumption.
</div>

---

## 1. Show Overall GPU Status

```bash
nvidia-smi
```

Most commonly used to <span style="color:#E8600A;font-weight:700">quickly check whether GPUs are idle or busy</span>.

---

## 2. Monitor in Real Time

Refresh every second:

```bash
nvidia-smi -l 1
```

---

## 3. List All GPUs

```bash
nvidia-smi -L
```

---

## 4. Show Processes Using GPUs

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">pmon</code> = <span style="color:#2980B9">process monitor</span>

```bash
nvidia-smi pmon
```

Refresh every 2 seconds:

```bash
nvidia-smi pmon -d 2
```

### 1) `pmon` Column Reference

| Column | Full Name | Meaning |
| --- | --- | --- |
| <span style="color:#E8600A;font-weight:700">pid</span> | Process ID | The Linux process ID using the GPU |
| **type** | Process type | GPU workload type: **C** = Compute (CUDA), **G** = Graphics, **C+G** = Both |
| <span style="color:#E8600A;font-weight:700">sm</span> | Streaming Multiprocessor utilization | Percentage of **GPU compute cores** being used by the process |
| <span style="color:#E8600A;font-weight:700">mem</span> | Memory controller utilization | Percentage of **GPU memory bandwidth** used by the process |
| **enc** | Encoder utilization | Usage of the **NVENC video encoder** |
| **dec** | Decoder utilization | Usage of the **NVDEC video decoder** |
| **jpg** | JPEG engine utilization | Usage of the **hardware JPEG decoder/encoder** |
| **ofa** | Optical Flow Accelerator utilization | Usage of the **hardware optical-flow engine** (video/vision tasks) |
| <span style="color:#E8600A;font-weight:700">fb</span> | Frame Buffer memory | Amount of **GPU VRAM used** by the process (in MB) |
| **ccpm** | Compute & Copy Engine / Protected Memory | Internal GPU engine / protection state info; often **0** on most systems |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Key columns to watch: </span>
<strong>pid</strong> — shows which process is using the GPU.
<strong>sm</strong> — indicates whether GPU cores are actively computing.
<strong>fb</strong> — VRAM usage in MB; shows how much memory is consumed.
<strong>mem</strong> — memory bandwidth utilization; indicates I/O pressure on GPU memory.
</div>

### 2) Example Output

```
[xli49@ghpc008 ~]$ nvidia-smi pmon -i 0 -s um
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0          -     -      -      -      -      -      -      -      -      -    -
    0          -     -      -      -      -      -      -      -      -      -    -

[xli49@ghpc008 ~]$ nvidia-smi pmon -i 0
# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0          -     -      -      -      -      -      -      -    -
    0          -     -      -      -      -      -      -      -    -
```

All dashes (`-`) indicate <span style="color:#2980B9">GPU 0 is currently idle</span> — no processes are running on it.

---

## 5. Custom Query of GPU Information

```bash
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
```

Commonly used for scripts, logging, and automated monitoring.

---

## 6. Log GPU Status to a File

```bash
nvidia-smi -l 5 -f gpu.log
```

Records GPU information every <span style="color:#E8600A;font-weight:700">5 seconds</span> and appends it to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gpu.log</code>.

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nvidia-smi</code> for a quick snapshot, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nvidia-smi pmon</code> to watch per-process GPU activity in real time, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--query-gpu</code> for scriptable, structured output — focus on <strong>pid</strong>, <strong>sm</strong>, <strong>fb</strong>, and <strong>mem</strong> for the most actionable signals.</div>