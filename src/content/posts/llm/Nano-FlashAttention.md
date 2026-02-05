---
title: "Nano-FlashAttention"
published: 2026-02-04
description: "Nano-FlashAttention"
image: ""
tags: ["llm","Nano-FlashAttention"]
category: llm
draft: false
lang: ""
---

```mermaid
sequenceDiagram
    autonumber
    participant PY as Python
    participant EXT as C++/CUDA 扩展
    participant LAUNCH as CUDA 启动器
    participant GPU as GPU Kernel (forward_kernel)
    participant HBM as 全局显存 (HBM)
    participant SMEM as 共享内存 (SRAM)
    participant REG as 寄存器

    Note over PY: q,k,v 已经是 CUDA Tensor（数据在 GPU 上）

    PY->>EXT: minimal_attn.forward(q,k,v)
    EXT->>EXT: 读取维度 B, nh, N, d
    EXT->>EXT: 设置分块大小 Bc=32, Br=32
    EXT->>EXT: 计算 Tc=ceil(N/Bc), Tr=ceil(N/Br)
    EXT->>EXT: 计算 softmax_scale = 1/sqrt(d)

    EXT->>HBM: 分配输出 O = zeros_like(Q)
    EXT->>HBM: 分配 l = zeros([B,nh,N])
    EXT->>HBM: 分配 m = full([B,nh,N], -inf)

    EXT->>LAUNCH: 启动 forward_kernel<<<grid=(B,nh), block=(Bc), smem=sram_size>>>()
    LAUNCH->>GPU: 开始执行 kernel

    Note over GPU: 每个 block 负责一个 (batch=bx, head=by)\n每个线程 tx 负责 Br 中的一行 Query

    GPU->>HBM: 计算 qkv_offset 与 lm_offset（全局内存偏移）
    GPU->>SMEM: 申请共享内存区域：Qi, Kj, Vj, S

    loop j = 0..Tc-1（遍历 Key/Value 的列分块）
        GPU->>HBM: 读取 Kj 分块 (Bc x d)
        GPU->>HBM: 读取 Vj 分块 (Bc x d)
        GPU->>SMEM: 写入共享内存 Kj, Vj
        GPU-->>GPU: __syncthreads() 同步（确保所有线程都加载完）

        loop i = 0..Tr-1（遍历 Query 的行分块）
            GPU->>HBM: 读取 Qi 分块 (Br x d)
            GPU->>SMEM: 写入共享内存 Qi
            GPU->>HBM: 读取 row_m_prev / row_l_prev
            GPU->>REG: 保存到寄存器 row_m_prev,row_l_prev

            GPU->>SMEM: 计算 S = Qi * Kj^T (Br x Bc)
            GPU->>REG: row_m = 每行最大值 max(S)
            GPU->>SMEM: 计算 P = exp(S - row_m)
            GPU->>REG: row_l = 每行求和 sum(P)

            GPU->>REG: row_m_new = max(row_m_prev, row_m)
            GPU->>REG: row_l_new = exp(m_prev-m_new)*l_prev + exp(m-m_new)*l

            GPU->>SMEM: 计算 pv = P * Vj (Br x d)
            GPU->>HBM: 读取旧的 O_prev (Br x d)
            GPU->>HBM: 写回新的 O_new (Br x d)
            GPU->>HBM: 写回 m_new, l_new (Br)
        end

        GPU-->>GPU: __syncthreads() 同步（避免下一轮 j 使用错误的 Kj/Vj）
    end

    GPU-->>LAUNCH: kernel 执行结束
    LAUNCH-->>EXT: 返回 Host 侧
    EXT-->>PY: 返回输出 O tensor

    PY->>PY: 与 manual_attn 结果对比 + profiler 统计

```









这段代码实现了一个简化版的 Flash Attention 算法。为了清晰地展示其工作流程，我为你准备了两个层次的时序图：

1.  **系统级交互流程**：展示 Python 端、C++ Host 端与 GPU Device 端的数据流向与调用关系。
2.  **Kernel 内部算法流程（核心）**：展示 GPU 线程（Thread Block）如何利用 SRAM（共享内存）和 HBM（显存）进行分块计算（Tiling）。

### 1. 系统级交互流程 (System Level)

这个图展示了从 Python 代码发起请求到 GPU 执行并返回结果的全过程。

Code snippet

```
sequenceDiagram
    participant Py as Python Script (Main)
    participant Cpp as C++ Extension (Host)
    participant HBM as GPU HBM (Global Mem)
    participant GPU as GPU Cores (Kernel)

    Note over Py: 准备数据 Q, K, V
    Py->>HBM: 将 Tensor Q, K, V 移动到 CUDA
    Py->>Cpp: 调用 minimal_attn.forward(Q, K, V)
    
    rect rgb(240, 248, 255)
        Note over Cpp: C++ Host端逻辑
        Cpp->>Cpp: 计算分块大小 (Tc, Tr)<br/>设置 Grid/Block 维度
        Cpp->>HBM: 分配结果张量 O, l, m
        Cpp->>Cpp: 计算所需的 Shared Memory (SRAM) 大小
        Cpp->>GPU: 启动 CUDA Kernel (forward_kernel)
    end

    activate GPU
    Note over GPU: 执行 Flash Attention 计算<br/>(详见下方 Kernel 流程图)
    GPU-->>HBM: 写入最终结果 O
    deactivate GPU

    Cpp-->>Py: 返回 Tensor O
    
    Note over Py: PyTorch Profiler 记录时间
    Py->>Py: 与手动 Attention 结果比对
```

------

### 2. Kernel 内部算法与内存交互流程 (Kernel Logic)

这是 Flash Attention 的核心。代码通过**双重循环**（外层循环 K/V 块，内层循环 Q 块）来减少内存访问。

**注意代码中的特殊逻辑**：

在此实现中，外层循环遍历 `Tc` (Key/Value 分块)，内层循环遍历 `Tr` (Query 分块)。这与标准的 FlashAttention (v2) 稍有不同（通常外层是 row/Query，内层是 col/Key），这意味着该代码在内层循环中需要不断从 HBM 读取并更新 `O`、`l`、`m`。

Code snippet

```mermaid
sequenceDiagram
    participant PY as Python
    participant EXT as C++/CUDA Extension
    participant LAUNCH as CUDA Launch
    participant TB as Thread Block (Registers)
    participant SRAM as Shared Memory (SRAM)
    participant HBM as Global Memory (HBM)

    %% =======================
    %% 0) Python 侧数据预处理
    %% =======================
    Note over PY: Python 侧数据预处理（准备 Q/K/V）
    PY->>PY: q,k,v = torch.randn(B,nh,N,d)
    PY->>HBM: q.cuda(), k.cuda(), v.cuda() (CPU->GPU 拷贝)

    %% =======================
    %% 1) C++ 扩展侧预处理
    %% =======================
    Note over EXT: C++/CUDA 扩展侧预处理（分块参数 + 初始化显存）
    PY->>EXT: minimal_attn.forward(q,k,v)

    EXT->>EXT: 读取维度 B, nh, N, d
    EXT->>EXT: 设置分块 Bc=32, Br=32
    EXT->>EXT: 计算 Tc=ceil(N/Bc), Tr=ceil(N/Br)
    EXT->>EXT: 计算 softmax_scale = 1/sqrt(d)

    EXT->>HBM: 分配并初始化输出 O = zeros_like(Q)
    EXT->>HBM: 分配并初始化 l = zeros([B,nh,N])
    EXT->>HBM: 分配并初始化 m = full([B,nh,N], -INF)

    %% =======================
    %% 2) 启动 Kernel
    %% =======================
    EXT->>LAUNCH: forward_kernel<<<grid=(B,nh), block=(Bc), smem=sram_size>>>()
    LAUNCH->>TB: Kernel 启动

    Note over TB, HBM: 每个 block 负责一个 (bx=batch, by=head)\n每个线程 tx 处理一行 Query（Br中的一行）

    %% =======================
    %% 3) Kernel 内部循环
    %% =======================
    loop 外层循环 j = 0..Tc-1（遍历 Key/Value 分块）
        Note left of TB: [预取] 加载 K/V 分块到 SRAM
        TB->>HBM: 读取当前块 K[j], V[j]
        TB->>SRAM: 写入 Kj, Vj

        Note over TB, SRAM: __syncthreads()（等待 Kj/Vj 加载完成）

        loop 内层循环 i = 0..Tr-1（遍历 Query 分块）
            Note left of TB: [预取] 加载 Q 分块到 SRAM
            TB->>HBM: 读取当前块 Q[i]
            TB->>SRAM: 写入 Qi

            Note left of TB: 读取历史统计量 m/l（用于在线 softmax）
            TB->>HBM: 读取 m_prev, l_prev
            TB->>TB: 保存在寄存器 row_m_prev,row_l_prev

            rect rgb(255, 250, 240)
                Note over TB: 计算注意力分数 S & 当前块 softmax 统计量
                TB->>SRAM: 读取 Qi, Kj
                TB->>TB: S = Qi * Kj^T (点积 * softmax_scale)
                TB->>TB: row_m = max(S) (当前块最大值)
                TB->>TB: P = exp(S - row_m)
                TB->>TB: row_l = sum(P) (当前块求和)
            end

            rect rgb(230, 240, 255)
                Note over TB: 在线 Softmax 更新（融合 prev + curr）
                TB->>TB: row_m_new = max(row_m_prev, row_m)
                TB->>TB: row_l_new = exp(m_prev-m_new)*l_prev + exp(m-m_new)*l

                TB->>SRAM: 读取 Vj
                TB->>HBM: 读取旧的 O[i]
                TB->>TB: O_new = (1/row_l_new) * ( row_l_prev*exp(m_prev-m_new)*O_old + exp(m-m_new)*(P*Vj) )
            end

            Note left of TB: 写回结果到 HBM
            TB->>HBM: 写入更新后的 O[i]
            TB->>HBM: 写入更新后的 m_new, l_new
        end

        Note over TB, SRAM: __syncthreads()（同步后进入下一轮 K/V 分块）
    end

    TB-->>LAUNCH: Kernel 完成
    LAUNCH-->>EXT: 返回 C++ 侧
    EXT-->>PY: 返回输出 O
    PY->>PY: 与 manual_attn 对比/Profiler 统计
```

### 关键步骤解析

1.  **SRAM 缓存 (Tiling)**:
    -   代码显式分配了 `extern __shared__ float sram[]`。
    -   **外层循环 ($j$)** 负责将 $K$ 和 $V$ 的切片（Tile）加载到高速的 SRAM 中。这是 Flash Attention 能够加速的关键，因为它复用了加载到 SRAM 中的 $K/V$ 数据来计算多个 $Q$。
2.  **HBM 读写瓶颈**:
    -   你可以从第二个图中看到，在内层循环 ($i$) 中，线程块频繁地从 HBM 读取 $O_{old}, l, m$ 并写回 $O_{new}, l, m$。
    -   **优化提示**: 更加优化的 Flash Attention 实现（如 v2）通常会交换循环顺序（外层 $Q$，内层 $K/V$），这样 $O, l, m$ 可以一直保存在寄存器或 SRAM 中，直到计算完所有的 $K/V$ 后再一次性写入 HBM，从而进一步减少 HBM 的带宽压力。
3.  **计算逻辑**:
    -   $S = QK^T$ 以及 Softmax 的指数计算是在寄存器层面完成的，非常快。主要的延迟通常来自于图中标注 `HBM` 的读写操作。







