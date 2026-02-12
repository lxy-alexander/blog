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

>   These are basic Linux and Slurm( Simple Linux Utility for Resource Management) commands used in HPC to check **memory, CPU, GPU resources, and job status**.



## lscpu

Display **CPU hardware information** (cores, frequency, NUMA layout, cache, etc.).



## scontrol show node ghpc008 | grep -i gres

**Purpose:** Show the node’s **GPU resources**(number and type of GPUs).

------



## top

Show real-time view of **processes, CPU usage, and memory usage**.

------



## scontrol show job $SLURM_JOB_ID

Display **detailed information about the current Slurm job**, including CPUs, GPUs, memory, assigned node, and runtime status.

------



