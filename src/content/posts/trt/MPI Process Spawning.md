---
title: "MPI Process Spawning"
published: 2026-05-23
description: "MPI Process Spawning"
image: ""
tags: ["trt","MPI Process Spawning"]
category: trt
draft: false
lang: ""
createdAt: "2026-05-24T00:22:54.039.304450910Z"
---

# MPI (Message Passing Interface, 消息传递接口)

MPI is a standard that lets many separate processes communicate with each other and work together on one job. (MPI 是一套标准,让多个独立进程互相通信、协同完成同一个任务。)

<br>

## 1. Core Concepts (核心概念)

MPI organizes processes into a group and gives each one an ID so they know who they are and who to talk to. (MPI 把进程组织成一个组,并给每个进程一个编号,让它们知道自己是谁、该和谁通信。)

### 1) Communicator (通信域)

A communicator is the "chat room" that holds a set of processes allowed to talk to each other. (通信域是一个"聊天室",装着一组允许互相通信的进程。)

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD       # The default room holding ALL processes (包含所有进程的默认通信域)
rank = comm.Get_rank()      # My ID inside this room (我在这个域里的编号)
size = comm.Get_size()      # Total members in this room (域内进程总数)

print(f"I am rank {rank} of {size}")
# Output (run with: mpirun -n 4 python script.py):
# I am rank 0 of 4
# I am rank 1 of 4
# I am rank 2 of 4
# I am rank 3 of 4
# (order may vary because processes run in parallel)
```

### 2) Rank (进程编号)

A rank is the unique ID of a process, where rank 0 is usually the leader that coordinates the rest. (rank 是进程的唯一编号,其中 rank 0 通常是协调其他进程的主进程。)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("I am the leader, I assign work")    # Only rank 0 does this
else:
    print(f"I am worker {rank}, I wait for tasks")
# Output (mpirun -n 3):
# I am the leader, I assign work
# I am worker 1, I wait for tasks
# I am worker 2, I wait for tasks
```

<br>

## 2. Point-to-Point Communication (点对点通信)

Point-to-point means one specific process sends data directly to one other specific process. (点对点指一个特定进程把数据直接发给另一个特定进程。)

The basic pair is `send` (发送) and `recv` (接收): one side ships the data, the other side waits to catch it. (基本搭配是 send 和 recv:一方发出数据,另一方等着接收。)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {"msg": "hello", "value": 42}
    comm.send(data, dest=1)              # Send to rank 1 (发给 1 号进程)
    print("Rank 0 sent the data")
elif rank == 1:
    data = comm.recv(source=0)           # Receive from rank 0 (从 0 号进程接收)
    print(f"Rank 1 received: {data}")

# Output (mpirun -n 2):
# Rank 0 sent the data
# Rank 1 received: {'msg': 'hello', 'value': 42}
```

<br>

## 3. Collective Communication (集合通信)

Collective communication is when ALL processes in the group participate in one shared operation at the same time. (集合通信指组内所有进程同时参与同一个共享操作。)

### 1) Broadcast (广播)

Broadcast sends one process's data to every other process in the group. (广播把一个进程的数据发送给组内所有其他进程。)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = "config_from_leader"          # Only rank 0 has the data first
else:
    data = None

data = comm.bcast(data, root=0)          # Everyone gets rank 0's data (所有进程拿到 0 号的数据)
print(f"Rank {rank} now has: {data}")

# Output (mpirun -n 3):
# Rank 0 now has: config_from_leader
# Rank 1 now has: config_from_leader
# Rank 2 now has: config_from_leader
```

### 2) All-Reduce (全规约)

All-reduce combines a value from every process (e.g. sums them) and gives the result back to all of them. (all-reduce 把每个进程的一个值合并起来(例如求和),再把结果发回给所有进程。)

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_value = rank + 1                   # rank 0->1, rank 1->2, rank 2->3
total = comm.allreduce(local_value, op=MPI.SUM)   # Sum across all (跨所有进程求和)
print(f"Rank {rank}: my value={local_value}, total={total}")

# Output (mpirun -n 3):
# Rank 0: my value=1, total=6
# Rank 1: my value=2, total=6
# Rank 2: my value=3, total=6
```

The math behind all-reduce with the SUM operation:

$$ \text{result} = \sum_{i=0}^{N-1} \text{value}_i $$

<br>

## 4. Two Ways to Enter an MPI Group (进入 MPI 组的两种方式)

A process gets into an MPI group either by being launched into one (static) or by creating one itself at runtime (spawn). (进程进入 MPI 组,要么是被启动时放进去(静态),要么是自己在运行时创建一个(派生)。) ==mpirun is created in advance from outside. Spawn is created during runtime within the program.==

Spawn is created during runtime within the program.

### 1) Static Launch with mpirun (用 mpirun 静态启动)

`mpirun` builds the group first and then launches your program already inside it. (mpirun 先建好组,再把你的程序作为组内成员启动。)

```bash
# mpirun creates the group, then starts python as rank 0 inside it
mpirun -n 1 python3 script.py
# The process is "born" inside an MPI group — no spawn needed
```

### 2) Dynamic Spawn (动态派生)

Spawn means a running process asks MPI to create brand-new worker processes from scratch. (派生指一个正在运行的进程请求 MPI 凭空创建全新的 worker 进程。)

```python
from mpi4py import MPI

# A running process tries to create 2 new workers at runtime
comm = MPI.COMM_SELF.Spawn(
    "python", args=["worker.py"], maxprocs=2
)
# Output in a normal cluster: 2 workers start
# Output in a restricted container (Apptainer):
# mpi4py.MPI.Exception: MPI_ERR_SPAWN: could not spawn processes
```

<br>

## 5. Why Spawn Fails in Containers (为什么容器中派生会失败)

Spawn needs a process manager (进程管理器) running in the background, which lightweight containers like Apptainer usually do not provide. (派生需要后台有一个进程管理器,而像 Apptainer 这样的轻量容器通常不提供。)

The framework's decision logic:

$$ \text{action} = \begin{cases} \text{reuse existing session} & \text{if already in an MPI group} \ \text{try to spawn} & \text{if not in any group} \end{cases} $$

```python
# Pseudocode of what TensorRT-LLM does at startup
def start_workers():
    if already_in_mpi_group():        # Did mpirun put us in a group? (mpirun 把我们放进组了吗?)
        reuse_existing_session()       # -> "Refreshed the MPI local session" (安全)
    else:
        MPI.COMM_SELF.Spawn(...)       # -> MPI_ERR_SPAWN in containers (在容器中报错)

# Launched as `python3 script.py`     -> not in group -> tries spawn -> FAILS
# Launched as `mpirun -n 1 python3 ...` -> in group   -> reuses        -> WORKS
```

<br>

## 6. Applying It: TensorRT-LLM Inference (实际应用:TensorRT-LLM 推理)

TensorRT-LLM uses MPI to coordinate one process per GPU, so you start it with `mpirun -n <number_of_gpus>`. (TensorRT-LLM 用 MPI 协调每块 GPU 一个进程,所以用 mpirun -n GPU数量 来启动。)

```bash
# Single GPU: still use mpirun -n 1 to avoid the broken spawn path
mpirun -n 1 python3 examples/llm-api/llm_inference.py
# Output: [RANK 0] Refreshed the MPI local session ... runs successfully

# Multi-GPU (e.g. 2-way tensor parallel): one process per GPU
mpirun -n 2 python3 examples/llm-api/llm_inference.py
# Each GPU runs one rank; ranks use all-reduce to sync results every layer
```

Even with one GPU, the point is being inside an MPI group, not how many processes there are. (即使只有一块 GPU,关键是"处在 MPI 组内",而不是进程有几个。)

The key idea in one sentence: `mpirun` does not fix spawn — it makes spawn unnecessary. (一句话核心:mpirun 不是修好了派生,而是让派生变得没有必要。)

<br> <br>
