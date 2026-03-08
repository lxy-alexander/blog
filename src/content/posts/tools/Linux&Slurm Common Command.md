---
title: "Linux&Slurm Common Command"
published: 2026-02-11
description: "Linux&Slurm Common Command"
image: ""
tags: ["tools","Linux&Slurm Common Command"]
category: tools
draft: false
lang: ""
---

# **I. HPC Resource Inspection Commands — Linux & SLURM**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> These are essential Linux and <strong>SLURM (Simple Linux Utility for Resource Management)</strong> commands used on HPC clusters to inspect <strong>memory, CPU, GPU resources, and job status</strong>.
</div>

---

## 1. `lscpu` — CPU Hardware Info

```bash
lscpu
```

Displays <span style="color:#E8600A;font-weight:700">CPU hardware information</span>: core count, clock frequency, NUMA topology, cache sizes, and more.

---

## 2. `scontrol show node ... | grep -i gres` — Node GPU Resources

```bash
scontrol show node ghpc008 | grep -i gres
```

Shows the node's <span style="color:#E8600A;font-weight:700">GPU resources</span> — the number and type of GPUs allocated or available on that node.

---

## 3. `top` — Real-time Process Monitor

```bash
top
```

Displays a live, continuously refreshed view of <span style="color:#E8600A;font-weight:700">running processes, CPU usage, and memory usage</span>.

---

## 4. `scontrol show job $SLURM_JOB_ID` — Current Job Details

```bash
scontrol show job $SLURM_JOB_ID
```

Displays <span style="color:#E8600A;font-weight:700">detailed information about the current SLURM job</span>, including assigned CPUs, GPUs, memory allocation, target node, and runtime status.

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">lscpu</code> for CPU specs, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scontrol show node ... | grep gres</code> for GPU inventory, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">top</code> for live process monitoring, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">scontrol show job</code> to inspect your running SLURM job's resource allocation.</div>


