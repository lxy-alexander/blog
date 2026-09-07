---
title: "vllm kv_connector"
published: 2026-09-02
description: "vllm kv_connector"
image: ""
tags: ["llm_inference","vllm kv_connector"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-02T19:55:12.061.488291493Z"
---

```mermaid
flowchart LR
    U["用户请求<br/>req-42"]

    subgraph EC["vLLM EngineCore 进程"]
        S["Core Scheduler<br/>决定本轮调度哪些请求"]
        KVM["KVCacheManager<br/>分配/释放 GPU blocks"]
        MCS["MooncakeStoreScheduler<br/>维护 LoadSpec / RequestTracker<br/>生成 ReqMeta"]
        SNAP["KVConnectorBlockState<br/>Core 当前 block-table 快照"]
        ASSERT{"req-42 是否存在于<br/>当前 block table？"}
    end

    subgraph WP["vLLM GPU Worker 进程"]
        LKC["LookupKeyServer<br/>Worker Rank 0"]
        MW["MooncakeStoreWorker"]
        RT["KV Load Threads<br/>Mooncake → GPU"]
        ST["KV Store Thread<br/>GPU → Mooncake"]
        GPU["GPU KV Cache<br/>blocks 101, 102, 103"]
    end

    subgraph MC["Mooncake 系统"]
        MASTER["mooncake_master<br/>元数据与存储位置管理"]
        MEM["CPU Memory Segments"]
        SSD["SSD Offload"]
    end

    U --> S

    S -->|"查询本地 prefix"| KVM
    S -->|"查询外部 prefix"| MCS

    MCS -->|"ZMQ LOOKUP_MSG<br/>hash H1,H2,H3"| LKC
    LKC -->|"查询 key 是否存在"| MW
    MW --> MASTER
    MASTER --> MEM
    MASTER --> SSD
    LKC -->|"返回 hit_length=48"| MCS

    S -->|"allocate_slots()"| KVM
    KVM -->|"分配 blocks<br/>[101,102,103]"| GPU
    KVM -->|"update_state_after_alloc()"| MCS

    MCS -->|"LoadSpec<br/>can_load=True"| S
    S -->|"构造本轮 snapshot"| SNAP
    MCS -->|"构造 ReqMeta"| ASSERT
    SNAP --> ASSERT

    ASSERT -->|"存在：使用当前 block IDs"| MW
    MW -->|"load_spec.can_load=True"| RT
    RT -->|"RDMA/TCP GET"| MEM
    RT -->|"SSD GET"| SSD
    RT -->|"写入"| GPU

    MW -->|"can_save=True"| ST
    ST -->|"读取"| GPU
    ST -->|"RDMA/TCP PUT"| MEM
    ST -->|"SSD PUT"| SSD

    RT -.->|"invalid_block_ids<br/>加载失败反馈"| S

    ASSERT -.->|"不存在：旧代码 assert<br/>EngineCore 崩溃"| CRASH["AssertionError<br/>所有 in-flight 请求失败"]

    style EC fill:#e8f1ff,stroke:#3973ac
    style WP fill:#e9f8ec,stroke:#39804a
    style MC fill:#fff3df,stroke:#b57a22
    style ASSERT fill:#fff0f0,stroke:#d33,stroke-width:2px
    style CRASH fill:#ffd9d9,stroke:#c00,stroke-width:2px
```



```mermaid
sequenceDiagram
    autonumber

    actor U as 用户
    participant C as vLLM Core Scheduler
    participant K as KVCacheManager
    participant MS as MooncakeStoreScheduler
    participant LS as LookupKeyServer<br/>Worker Rank 0
    participant W as MooncakeStoreWorker
    participant M as MooncakeDistributedStore
    participant G as GPU KV Cache

    Note over C,M: 一、启动并建立连接

    C->>MS: 创建 MooncakeStoreConnector(SCHEDULER)
    MS->>LS: 创建 ZMQ LookupKeyClient 连接

    W->>M: MooncakeDistributedStore.setup()
    M-->>W: 连接 Master/内存/SSD 成功
    W->>LS: 启动 LookupKeyServer

    Note over U,G: 二、req-42 第一次进入

    U->>C: 请求 req-42（64 tokens）
    C->>K: 查询本地 prefix cache
    K-->>C: 本地命中 0 tokens

    C->>MS: get_num_new_matched_tokens(req-42, 0)
    MS->>LS: ZMQ LOOKUP_MSG<br/>hash=[H1,H2,H3,H4]
    LS->>W: lookup(64, block_hashes)
    W->>M: 查询 H1～H4 是否存在
    M-->>W: H1,H2,H3 存在，H4 不存在
    W-->>LS: hit_length=48
    LS-->>MS: MooncakeLookupResult(48)

    MS->>MS: 创建 LoadSpec<br/>vllm_cached=0<br/>kvpool_cached=48<br/>can_load=False
    MS-->>C: 需要分配 48 tokens<br/>load_async=True

    Note over C,G: 三、Core 分配 GPU blocks

    C->>K: allocate_slots(req-42, 48)
    K->>G: 分配 blocks [101,102,103]
    G-->>K: 分配成功
    K-->>C: KVCacheBlocks([101,102,103])

    C->>MS: update_state_after_alloc()<br/>blocks=[101,102,103]
    MS->>MS: _unfinished_requests[req-42]<br/>= [101,102,103]
    MS->>MS: load_spec.can_load=True

    C->>C: req-42 状态改为<br/>WAITING_FOR_REMOTE_KVS

    Note over C,W: 四、生成并下发 load-only ReqMeta

    C->>MS: build_connector_meta(scheduler_output)
    MS->>MS: 创建 RequestTracker<br/>token_len=48<br/>num_saved_tokens=0
    MS->>MS: from_request_tracker()

    Note right of MS: load_spec.can_load=True<br/>所以 skip_save=True

    MS-->>C: ReqMeta<br/>req_id=req-42<br/>can_save=False<br/>load_spec.can_load=True<br/>block_ids=[101,102,103]

    C->>W: 下发 SchedulerOutput/ReqMeta
    W->>W: start_load_kv()

    Note right of W: can_load=True<br/>加入异步接收队列

    W->>M: GET H1,H2,H3
    M-->>W: H1成功，H2/H3读取失败
    W->>G: H1写入block 101<br/>102/103无效
    W-->>C: invalid_block_ids={102,103}

    Note over C,G: 五、vLLM 执行失败恢复

    C->>C: _handle_invalid_blocks()
    C->>C: 截断有效KV范围
    C->>C: failed_recving_kv_req_ids<br/>加入 req-42

    C->>K: 释放/回收无效 blocks
    K->>G: free([101,102,103])
    C->>C: req-42 状态<br/>WAITING_FOR_REMOTE_KVS → WAITING

    Note right of G: 101/102/103 后续<br/>可能分配给其他请求

    Note over C,M: 六、req-42 再次尝试 Mooncake lookup

    C->>MS: get_num_new_matched_tokens(req-42, 0)
    MS->>LS: 再次查询 H1～H4
    LS->>W: lookup()
    W->>M: 查询 key
    M-->>W: 仍然报告前48 tokens存在
    W-->>LS: hit_length=48
    LS-->>MS: hit_length=48

    MS->>MS: 创建新的 LoadSpec<br/>can_load=False

    C->>K: 尝试重新分配 GPU blocks
    K-->>C: 本轮未成功分配/未调度

    Note over C: req-42 不在 scheduled_new_reqs<br/>也不在 scheduled_cached_reqs

    C->>C: 构造当前 block snapshot
    C->>C: snapshot_req_ids={}
    C->>C: block_state.block_ids={}

    Note over MS: Mooncake仍可能保留<br/>req-42的历史unfinished状态<br/>旧blocks=[101,102,103]

    Note over C,MS: 七、问题发生

    C->>MS: build_connector_meta(scheduler_output)

    MS->>MS: pending-load循环发现 req-42<br/>不在new/cached列表
    MS->>MS: pop LoadSpec(can_load=False)
    MS->>MS: 创建 RequestTracker<br/>token_len=48<br/>旧blocks=[101,102,103]

    MS->>MS: from_request_tracker()

    Note right of MS: 48 tokens跨过16/32/48边界<br/>can_load=False不会强制skip_save<br/>因此产生can_save=True

    MS->>MS: 生成 ReqMeta<br/>can_save=True<br/>load_spec=None<br/>block_ids=[101,102,103]

    MS->>MS: _apply_current_save_block_ids()
    MS->>MS: block_state.block_ids.get("req-42")
    MS->>MS: 返回 None

    rect rgb(255, 220, 220)
        MS->>MS: assert block_ids is not None
        MS--xC: AssertionError
        Note over C,MS: EngineCore主调度路径退出<br/>所有in-flight请求可能返回500
    end
```

整个架构可以分为三层：EngineCore 调度层、GPU Worker 执行层、Mooncake 存储层。

## 1. EngineCore 调度层

主要组件：

```
Core Scheduler
KVCacheManager
MooncakeStoreScheduler
```

### Core Scheduler

负责决定：

-   本轮调度哪些请求；
-   每个请求计算多少 token；
-   请求处于 `WAITING`、`RUNNING` 还是 `WAITING_FOR_REMOTE_KVS`；
-   KV 加载失败后，是请求失败还是重新计算；
-   生成 `SchedulerOutput` 下发给 worker。

它是整个流程的总调度者。

### KVCacheManager

负责 GPU block 的生命周期：

-   为请求分配 GPU blocks；
-   查询请求当前的完整 block table；
-   加载失败时释放无效 blocks；
-   prefix cache 的缓存和复用；
-   防止仍被异步任务使用的 block 提前释放。

例如：

```
req-42 → GPU blocks [101,102,103]
```

这个映射的最终权威来源是 `KVCacheManager`。

### MooncakeStoreScheduler

这是 vLLM 内部的 Mooncake 调度适配器，负责：

-   查询 Mooncake 中是否存在请求前缀；
-   保存每个请求的 `LoadSpec`；
-   使用 `RequestTracker` 跨轮记录请求状态；
-   为当前调度轮生成 `ReqMeta`；
-   决定本轮执行 load 还是 store；
-   在异步 store 期间 pin GPU blocks。

它不直接搬运 KV 数据，主要生成操作计划。

## 2. GPU Worker 执行层

主要组件：

```
MooncakeStoreWorker
LookupKeyServer
Load Threads
Store Thread
GPU KV Cache
```

### MooncakeStoreWorker

每个 GPU rank 都有一个 worker-side connector。

它负责：

-   创建 `MooncakeDistributedStore`；
-   连接 Mooncake Master；
-   注册 RDMA/TCP 传输资源；
-   接收 Scheduler 下发的 `ReqMeta`；
-   执行真正的 KV load/store；
-   把加载失败的 block ID 返回给 EngineCore；
-   报告 store job 完成。

### LookupKeyServer

只有 worker rank 0 启动 LookupKeyServer。

它负责接收 Scheduler 的查询：

```
这些 block hashes 在 Mooncake 中存在吗？
```

通信方式是 vLLM 内部的 ZMQ。

查询链路：

```
MooncakeStoreScheduler
→ LookupKeyClient
→ ZMQ
→ Worker Rank 0 的 LookupKeyServer
→ MooncakeStoreWorker.lookup()
→ MooncakeDistributedStore
```

查询只传递：

-   token 数；
-   block hashes；
-   查询结果。

它不传输真正的 KV Tensor。

### Load Threads

执行数据加载：

```
Mooncake → GPU
```

例如：

```
Mooncake H1 → GPU block 101
Mooncake H2 → GPU block 102
Mooncake H3 → GPU block 103
```

可能有多个接收线程并发加载。

### Store Thread

执行数据保存：

```
GPU → Mooncake
```

例如：

```
GPU block 101 → Mooncake H1
GPU block 102 → Mooncake H2
GPU block 103 → Mooncake H3
```

store 是异步的，所以 Scheduler 必须 pin 相关 GPU blocks，防止 worker 读取之前被其他请求复用。

## 3. Mooncake 存储层

主要组件：

```
mooncake_master
MooncakeDistributedStore
CPU Memory Segments
SSD Offload
```

### Mooncake Master

负责管理：

-   存储节点；
-   segment 位置；
-   key 的分布；
-   多个 vLLM 实例之间的共享；
-   CPU/SSD 存储层协调。

Master 主要管理元数据和拓扑，不直接参与每一字节的 GPU KV 搬运。

### MooncakeDistributedStore

这是 vLLM worker 使用的 Mooncake 客户端对象：

```
self.store = MooncakeDistributedStore()
```

它通过：

```
self.store.setup(...)
```

连接 Mooncake Master，并建立 RDMA/TCP 数据传输能力。

### CPU/SSD 存储

Mooncake 的 KV 数据可以位于：

```
CPU Memory
SSD
其他存储节点
```

从 vLLM 角度看，这些都属于 GPU KV Cache 之外的外部共享存储。

## 4. 两条通信通道

架构中最重要的是区分控制面和数据面。

### 控制面

负责决定“做什么”：

```
Core Scheduler
→ MooncakeStoreScheduler
→ LookupKeyClient
→ LookupKeyServer
→ Mooncake 查询
```

传递的数据很小，例如：

```
req_id
block hashes
hit_length
LoadSpec
ReqMeta
失败 block IDs
```

### 数据面

负责真正搬运 KV：

```
GPU KV Cache
↔ MooncakeStoreWorker
↔ MooncakeDistributedStore
↔ CPU/SSD
```

数据量很大，通常通过 RDMA/TCP。

## 5. `ReqMeta` 在架构中的位置

`ReqMeta` 是 Scheduler 和 Worker 之间的工作单：

```
ReqMeta(
    req_id="req-42",
    block_ids=([101,102,103],),
    can_save=False,
    load_spec=LoadSpec(can_load=True),
)
```

表示加载：

```
Mooncake → GPU blocks [101,102,103]
```

另一种：

```
ReqMeta(
    req_id="req-42",
    block_ids=([101,102,103],),
    can_save=True,
    load_spec=None,
)
```

表示保存：

```
GPU blocks [101,102,103] → Mooncake
```

## 6. #54870 发生在哪一层

问题发生在 EngineCore 调度层：

```
Core Scheduler
与
MooncakeStoreScheduler
```

两边对 `req-42` 的当前状态理解不一致：

```
Core Scheduler：
req-42 本轮没有调度
因此当前 snapshot 中没有 req-42

MooncakeStoreScheduler：
根据历史 tracker/load 状态
生成了 ReqMeta(can_save=True)
```

随后 MooncakeStoreScheduler 查询：

```
block_state.block_ids.get("req-42")
```

结果是：

```
None
```

并触发：

```
assert block_ids is not None
```

所以这个问题：

-   不是 Mooncake Master 崩溃；
-   不是 GPU kernel 崩溃；
-   不是 worker 后台 store 直接崩溃；
-   是 EngineCore 在构造 connector metadata 时发生 Python 断言。

一句话描述整个架构：

>   EngineCore Scheduler 制定 KV load/store 计划，GPU Worker 根据 `ReqMeta` 真正通过 MooncakeDistributedStore 搬运数据，Mooncake Master 管理外部存储拓扑；#54870 是 Scheduler 与 MooncakeStoreScheduler 的请求状态不同步，导致在任务下发前触发致命断言。

# 正常 pending-load 生命周期

```python
Mooncake lookup 命中48 tokens
    ↓
load_specs["req-42"] = LoadSpec(can_load=False)
    ↓
Core 分配GPU blocks [101,102,103]
    ↓
update_state_after_alloc()
    ↓
LoadSpec.can_load=True
    ↓
pending-load循环发现req-42
    ↓
生成ReqMeta(can_save=False, load_spec=...)
    ↓
worker执行 Mooncake → GPU
    ↓
load_spec从self.load_specs中移除
```

