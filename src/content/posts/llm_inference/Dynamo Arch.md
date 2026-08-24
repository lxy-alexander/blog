---
title: "Dynamo Arch"
published: 2026-08-24
description: "Dynamo Arch"
image: ""
tags: ["llm_inference","Dynamo Arch"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-24T23:01:30.717.875836006Z"
---

![image-20260824190134306](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260824190134306)



flowchart TB

```mermaid
flowchart TB

    %% ============================================================
    %% Client
    %% ============================================================
    subgraph CLIENT["Client / Application"]
        APP["Application"]
        API["Dynamo API<br/><br/>GET(key)<br/>PUT(key, context, value)"]

        APP --> API
    end


    %% ============================================================
    %% Dynamo Request Processing
    %% ============================================================
    subgraph REQUEST["Request Processing"]
        ENTRY["Dynamo Node<br/>接收客户端请求"]

        COORD["Coordinator<br/><br/>当前请求的协调者<br/>并非固定 Master"]

        HASH["Hash(key)<br/><br/>计算 key 在 token space 中的位置"]

        ENTRY --> COORD
        COORD --> HASH
    end

    API --> ENTRY


    %% ============================================================
    %% Cluster Metadata
    %% ============================================================
    subgraph CLUSTER["Cluster Membership & Metadata"]
        MEMBERS["Membership State<br/><br/>Node List<br/>Token Ownership<br/>Node State"]

        GOSSIP["Gossip Protocol<br/><br/>传播节点状态和 membership"]

        FAILURE["Failure Detection<br/><br/>本地判断节点<br/>Alive / Suspected"]

        JOIN["Node Join"]
        LEAVE["Node Leave / Failure"]

        JOIN --> MEMBERS
        LEAVE --> MEMBERS

        MEMBERS <--> GOSSIP

        GOSSIP --> FAILURE
    end


    %% ============================================================
    %% Partitioning
    %% ============================================================
    subgraph PARTITION["Partitioning — Consistent Hashing"]
        TOKENSPACE["Token Space<br/><br/>0 ... 2^m - 1"]

        V1["VNode / Token A"]
        V2["VNode / Token B"]
        V3["VNode / Token C"]
        V4["VNode / Token D"]
        V5["VNode / Token E"]
        V6["VNode / Token F"]

        PN1["Physical Node A"]
        PN2["Physical Node B"]
        PN3["Physical Node C"]

        TOKENSPACE --> V1
        TOKENSPACE --> V2
        TOKENSPACE --> V3
        TOKENSPACE --> V4
        TOKENSPACE --> V5
        TOKENSPACE --> V6

        V1 --> PN1
        V4 --> PN1

        V2 --> PN2
        V5 --> PN2

        V3 --> PN3
        V6 --> PN3
    end

    HASH --> TOKENSPACE

    MEMBERS -. "token ownership" .-> TOKENSPACE


    %% ============================================================
    %% Preference List
    %% ============================================================
    subgraph PL["Preference List"]
        PREF["Preference List<br/><br/>从 key 所在位置开始<br/>沿 Ring 顺时针选择节点"]

        DISTINCT["Skip duplicate physical nodes<br/><br/>保证副本位于不同物理节点"]

        TARGETS["Preferred Replicas<br/><br/>例如：A → B → C"]

        PREF --> DISTINCT
        DISTINCT --> TARGETS
    end

    TOKENSPACE --> PREF
    MEMBERS --> PREF
    FAILURE --> PREF


    %% ============================================================
    %% Replication
    %% ============================================================
    subgraph REPLICATION["Replication"]
        N["Replication Factor N<br/><br/>例如 N = 3"]

        R1["Replica A"]
        R2["Replica B"]
        R3["Replica C"]

        N --> R1
        N --> R2
        N --> R3
    end

    TARGETS --> N


    %% ============================================================
    %% Coordinator Logic
    %% ============================================================
    subgraph QUORUM["Coordinator Quorum Logic"]
        WRITE["PUT Request"]

        READ["GET Request"]

        WCOND["Write Condition<br/><br/>等待至少 W 个 Replica ACK"]

        RCOND["Read Condition<br/><br/>等待至少 R 个 Replica Response"]

        WSUCCESS["PUT Success<br/><br/>ACK ≥ W"]

        RSUCCESS["Continue Read Processing<br/><br/>Responses ≥ R"]

        WRITE --> WCOND
        WCOND --> WSUCCESS

        READ --> RCOND
        RCOND --> RSUCCESS
    end

    COORD --> WRITE
    COORD --> READ

    WRITE --> R1
    WRITE --> R2
    WRITE --> R3

    R1 -. ACK .-> WCOND
    R2 -. ACK .-> WCOND
    R3 -. ACK .-> WCOND

    READ --> R1
    READ --> R2
    READ --> R3

    R1 -. Version .-> RCOND
    R2 -. Version .-> RCOND
    R3 -. Version .-> RCOND


    %% ============================================================
    %% Object Versioning
    %% ============================================================
    subgraph VERSION["Object Versioning"]
        OBJECT["Object Version<br/><br/>Key<br/>Value<br/>Context"]

        VC["Vector Clock<br/><br/>例如<br/>{A:3, B:2}"]

        COMPARE["Version Comparison"]

        DOMINATE["Causal Ordering<br/><br/>一个版本 dominates 另一个"]

        CONCURRENT["Concurrent Versions<br/><br/>Sibling Versions"]

        WINNER["Latest Causal Version"]

        RECONCILE["Application Reconciliation<br/><br/>客户端 / 业务逻辑合并"]

        MERGED["Merged Version<br/><br/>重新 PUT"]

        OBJECT --> VC

        VC --> COMPARE

        COMPARE -->|"存在 causal ordering"| DOMINATE
        COMPARE -->|"无法互相 dominate"| CONCURRENT

        DOMINATE --> WINNER

        CONCURRENT --> RECONCILE
        RECONCILE --> MERGED
    end

    RSUCCESS --> OBJECT

    MERGED --> COORD


    %% ============================================================
    %% Sloppy Quorum
    %% ============================================================
    subgraph SLOPPY["Sloppy Quorum"]
        CHECK["Preferred Replica<br/>Available?"]

        NORMAL["Use Preferred Replica"]

        FALLBACK["Select Next Healthy Node<br/><br/>Preference List 之外的<br/>临时替代节点"]

        TEMP["Temporary Replica"]

        CHECK -->|"Yes"| NORMAL
        CHECK -->|"No"| FALLBACK

        FALLBACK --> TEMP
    end

    FAILURE --> CHECK
    TARGETS --> CHECK

    NORMAL --> R1
    NORMAL --> R2
    NORMAL --> R3


    %% ============================================================
    %% Hinted Handoff
    %% ============================================================
    subgraph HINTED["Hinted Handoff"]
        HINT["Hint Metadata<br/><br/>Intended Replica = Node C"]

        WAIT["Temporary Storage<br/><br/>数据暂存在 Substitute Node"]

        RECOVERY["Original Replica Recovers"]

        HANDOFF["Transfer Data<br/>to Original Replica"]

        CLEAN["Delete Hint<br/>and Temporary Copy"]

        HINT --> WAIT
        WAIT --> RECOVERY
        RECOVERY --> HANDOFF
        HANDOFF --> CLEAN
    end

    TEMP --> HINT

    HANDOFF --> R1
    HANDOFF --> R2
    HANDOFF --> R3


    %% ============================================================
    %% Read Repair
    %% ============================================================
    subgraph READREPAIR["Read Repair"]
        MULTI["Versions from Replicas"]

        RCOMPARE["Compare Vector Clocks"]

        CURRENT["Determine Current Version"]

        STALE["Detect Stale Replica"]

        FIX["Write Current Version<br/>to Stale Replica"]

        MULTI --> RCOMPARE
        RCOMPARE --> CURRENT
        CURRENT --> STALE
        STALE --> FIX
    end

    RSUCCESS --> MULTI

    R1 -. Version .-> MULTI
    R2 -. Version .-> MULTI
    R3 -. Version .-> MULTI

    FIX -. Repair .-> R1
    FIX -. Repair .-> R2
    FIX -. Repair .-> R3


    %% ============================================================
    %% Anti Entropy
    %% ============================================================
    subgraph ENTROPY["Anti-Entropy — Merkle Tree"]
        MA["Merkle Tree<br/>Replica A"]

        MB["Merkle Tree<br/>Replica B"]

        ROOT["Compare Root Hash"]

        BRANCH["Descend Different Branch"]

        RANGE["Locate Different<br/>Hash Range"]

        DIFF["Exchange Different Keys"]

        SYNC["Replica Synchronization"]

        MA --> ROOT
        MB --> ROOT

        ROOT -->|"same"| OK["No Sync Needed"]

        ROOT -->|"different"| BRANCH

        BRANCH --> RANGE
        RANGE --> DIFF
        DIFF --> SYNC
    end


    %% ============================================================
    %% Persistent Storage
    %% ============================================================
    subgraph STORAGE["Local Persistent Storage"]
        SA["Node A Local Store"]

        SB["Node B Local Store"]

        SC["Node C Local Store"]

        DATAA["Key<br/>Value<br/>Vector Clock<br/>Replica Metadata"]

        DATAB["Key<br/>Value<br/>Vector Clock<br/>Replica Metadata"]

        DATAC["Key<br/>Value<br/>Vector Clock<br/>Replica Metadata"]

        SA --> DATAA
        SB --> DATAB
        SC --> DATAC
    end

    R1 --> SA
    R2 --> SB
    R3 --> SC

    SA --> MA
    SB --> MB

    SYNC -. update .-> SA
    SYNC -. update .-> SB
    SYNC -. update .-> SC


    %% ============================================================
    %% Client Responses
    %% ============================================================
    WSUCCESS -->|"PUT OK"| API

    WINNER -->|"GET value"| API

    CONCURRENT -->|"GET sibling versions"| API
```



```python
Dynamo
├── lib
│   ├── bindings
│   │   └── python
│   │       ├── Rust → Python Binding          # Dynamo 的 Rust 与 Python 桥接层
│   │       ├── maturin                        # 用于编译 Rust Python 扩展
│   │       └── maturin develop --uv           # 编译并安装到当前 .venv
│   │
│   ├── gpu_memory_service
│   │   ├── GPU Memory Service                 # GPU 显存管理相关组件
│   │   └── uv pip install -e ...              # 以 editable 模式安装到 .venv
│   │
│   └── Runtime
│       ├── Rust                               # Dynamo 底层 Runtime 主要实现语言
│       ├── Worker 管理                         # 管理推理 Worker 生命周期
│       ├── 通信                                # Worker / Frontend 之间通信
│       └── 内存管理                             # GPU / KV Cache 等资源管理能力
│
├── Python 主包
│   ├── Frontend                               # 对外 HTTP / OpenAI API 入口
│   │   └── python3 -m dynamo.frontend         # 启动 Frontend
│   │
│   ├── Router                                 # 请求路由
│   ├── Discovery                              # Worker 服务发现
│   └── API                                    # API 相关实现
│
├── Backend
│   ├── vLLM                                   # 外部推理引擎，不是 Dynamo 自己实现
│   │   ├── Python + C++ + CUDA                # vLLM 自己内部的技术栈
│   │   └── python3 -m dynamo.vllm             # Dynamo 启动 vLLM Worker
│   │
│   ├── SGLang                                 # 可选外部推理引擎
│   └── TensorRT-LLM                           # 可选 NVIDIA 推理引擎
│
├── 外部 Native 组件
│   ├── NIXL                                   # 高性能数据 / GPU 内存传输组件
│   │   └── C++                                # NIXL 主要使用 C++
│   ├── PyTorch                                # 深度学习运行时
│   │   └── C++ / CUDA                         # PyTorch 底层大量使用 C++ / CUDA
│   ├── NCCL                                   # NVIDIA GPU 集合通信库
│   └── CUDA                                   # GPU 编程与运行时
│
└── GPU
    └── RTX A4000                              # 最终执行模型推理的硬件
```

