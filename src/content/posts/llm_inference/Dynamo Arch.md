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



```mermaid
flowchart TB

    %% ============================================================
    %% 1. Clients and external entry
    %% ============================================================
    subgraph CLIENT["1. Client and External Entry"]
        APP["Client Application<br/>OpenAI SDK / curl / Service"]
        OAI["OpenAI-compatible API<br/>POST /v1/chat/completions<br/>POST /v1/completions<br/>POST /v1/embeddings"]
        KSERVE["KServe gRPC API"]
        GATEWAY["Optional Kubernetes Gateway API<br/>Auth / Rate Limit / Traffic Policy"]
        EPP["Optional Endpoint Picker Plugin<br/>Select Frontend sidecar or worker"]

        APP --> OAI
        APP --> KSERVE
        APP --> GATEWAY
        GATEWAY --> EPP
    end

    %% ============================================================
    %% 2. Frontend
    %% ============================================================
    subgraph FRONTEND["2. Dynamo Frontend"]
        HTTP["HTTP / SSE Server<br/>OpenAI-compatible endpoints<br/><br/>components/src/dynamo/frontend/main.py"]
        GRPC["KServe gRPC Server<br/><br/>lib/llm runtime entrypoint"]
        VALIDATE["Request Validation<br/>model / sampling / limits / tools"]
        TEMPLATE["Chat Template<br/>messages to prompt"]
        TOKENIZER["Tokenizer<br/>prompt to token_ids"]
        PREPROCESS["Preprocessor<br/>PreprocessedRequest<br/><br/>model<br/>token_ids<br/>sampling_options<br/>stop_conditions<br/>routing metadata"]
        ENGINE["Dynamic LLM Engine<br/>make_engine<br/><br/>lib/llm<br/>lib/bindings/python"]
        POSTPROCESS["Postprocessor<br/>detokenization<br/>tool calls<br/>reasoning fields<br/>usage aggregation"]
        STREAM["HTTP SSE / JSON Response"]

        OAI --> HTTP
        KSERVE --> GRPC
        EPP -->|"direct mode"| HTTP

        HTTP --> VALIDATE
        GRPC --> VALIDATE
        VALIDATE --> TEMPLATE
        TEMPLATE --> TOKENIZER
        TOKENIZER --> PREPROCESS
        PREPROCESS --> ENGINE

        ENGINE --> POSTPROCESS
        POSTPROCESS --> STREAM
        STREAM --> APP
    end

    %% ============================================================
    %% 3. Distributed runtime
    %% ============================================================
    subgraph RUNTIME["3. Dynamo Distributed Runtime"]
        DR["DistributedRuntime<br/><br/>lib/runtime<br/>components communicate through endpoints"]
        NS["Namespace<br/>isolates one model deployment"]
        COMP["Component<br/>frontend / router / prefill / decode / backend"]
        ENDPOINT["Endpoint<br/>namespace.component.endpoint<br/><br/>examples:<br/>model.prefill.generate<br/>model.decode.generate"]
        CLIENTWATCH["Runtime Client<br/>watches endpoint membership"]
        REQUESTPLANE["Request Plane<br/>TCP default / HTTP / NATS legacy"]
        INHIBIT["Local Worker Inhibition<br/>temporarily avoids failed worker"]

        DR --> NS
        NS --> COMP
        COMP --> ENDPOINT
        CLIENTWATCH --> ENDPOINT
        ENDPOINT --> REQUESTPLANE
        REQUESTPLANE --> INHIBIT
    end

    ENGINE --> DR

    %% ============================================================
    %% 4. Service discovery
    %% ============================================================
    subgraph DISCOVERY["4. Service Discovery and Membership"]
        DISCOVERYMODE{"Discovery backend"}
        K8SDISC["Kubernetes-native Discovery<br/>DynamoWorkerMetadata CRD<br/>EndpointSlice / API watch"]
        ETCD["etcd Discovery<br/>lease and keepalive<br/>/services/namespace/component/endpoint"]
        REGISTRY["Live Endpoint Registry<br/>instance ID<br/>address<br/>worker role<br/>runtime configuration"]
        HEALTH["Health and Membership Update<br/>register / drain / deregister / lease expiry"]
        WORKERCONFIG["Worker RuntimeConfig<br/>worker ID<br/>DP rank / TP size<br/>KV capacity<br/>backend metadata<br/>taints"]

        DISCOVERYMODE --> K8SDISC
        DISCOVERYMODE --> ETCD
        K8SDISC --> REGISTRY
        ETCD --> REGISTRY
        HEALTH --> REGISTRY
        WORKERCONFIG --> REGISTRY
    end

    REGISTRY -. "watch updates" .-> CLIENTWATCH
    INHIBIT -. "request failure" .-> REGISTRY

    %% ============================================================
    %% 5. Routing layer
    %% ============================================================
    subgraph ROUTING["5. Request Routing Layer"]
        ROUTERMODE{"Router mode"}
        RR["Round Robin"]
        RANDOM["Random"]
        LEAST["Least Loaded"]
        DIRECT["Direct<br/>upstream-selected worker"]
        DEVICE["Device-aware Weighted"]
        KVROUTER["KV-aware Router<br/><br/>lib/kv-router"]
        ROUTERREQ["SchedulingRequest<br/>request_id<br/>token sequence<br/>routing constraints<br/>config override"]
        ELIGIBLE["Eligibility Filter<br/>healthy worker<br/>model / LoRA match<br/>role match<br/>taints<br/>optional pinned worker"]
        HASH["KV Block Hashing<br/>split token_ids by block size<br/>compute chained block hashes<br/><br/>lib/kv-router/src/protocols.rs"]
        MATCH["KV Prefix Lookup<br/>Radix Tree / Cuckoo Index<br/>find matching blocks per worker"]
        OVERLAP["Overlap Analysis<br/>GPU blocks<br/>host-pinned blocks<br/>disk blocks<br/>shared-cache hits"]
        LOAD["Worker Load State<br/>active prefill tokens<br/>decode blocks<br/>active requests<br/>KV utilization"]
        SCORE["Worker Cost Function<br/><br/>adjusted prefill<br/>= raw prefill - cache credit<br/><br/>cost<br/>= prefill cost + decode cost<br/>+ active request cost"]
        QUEUE["Scheduling Queue<br/>FCFS / WSPT policy<br/>admission control<br/>optional overlap refresh"]
        PICK["Worker Selection<br/>minimum cost when temperature=0<br/>softmax selection otherwise"]
        RESULT["WorkerSelectionResult<br/>worker_id<br/>dp_rank<br/>overlap_blocks"]

        ROUTERMODE --> RR
        ROUTERMODE --> RANDOM
        ROUTERMODE --> LEAST
        ROUTERMODE --> DIRECT
        ROUTERMODE --> DEVICE
        ROUTERMODE --> KVROUTER

        KVROUTER --> ROUTERREQ
        ROUTERREQ --> ELIGIBLE
        ROUTERREQ --> HASH
        HASH --> MATCH
        MATCH --> OVERLAP
        OVERLAP --> SCORE
        LOAD --> SCORE
        ELIGIBLE --> SCORE
        SCORE --> QUEUE
        QUEUE --> PICK
        PICK --> RESULT

        RR --> RESULT
        RANDOM --> RESULT
        LEAST --> RESULT
        DIRECT --> RESULT
        DEVICE --> RESULT
    end

    ENGINE --> ROUTERMODE
    REGISTRY -. "available workers" .-> ELIGIBLE
    WORKERCONFIG -. "capacity and topology" .-> ELIGIBLE

    %% ============================================================
    %% 6. Aggregated execution
    %% ============================================================
    subgraph AGG["6A. Aggregated Serving - Optional Path"]
        AGGWORKER["Aggregated Worker<br/>same engine performs prefill and decode"]
        AGGPREFILL["Prefill Phase<br/>process prompt tokens<br/>build or reuse KV cache"]
        AGGDECODE["Decode Phase<br/>generate one or more tokens per step"]
        AGGOUTPUT["Streaming Engine Output<br/>token_ids / text<br/>finish_reason / usage"]

        AGGWORKER --> AGGPREFILL
        AGGPREFILL --> AGGDECODE
        AGGDECODE --> AGGOUTPUT
    end

    RESULT -->|"aggregated topology"| AGGWORKER
    AGGOUTPUT --> ENGINE

    %% ============================================================
    %% 7. Disaggregated execution
    %% ============================================================
    subgraph DISAGG["6B. Disaggregated Prefill and Decode - Optional Path"]
        ORCHESTRATOR["Prefill Router / Disaggregated Orchestrator"]
        PREFSELECT["Select Prefill Worker<br/>KV overlap plus prefill load"]
        PREFPOOL["Prefill Worker Pool<br/>independently scalable"]
        PREF1["Prefill Worker P1"]
        PREF2["Prefill Worker P2"]
        PREFEXEC["Compute Missing Prompt KV<br/>reuse cached prefix when available"]
        META["Return Transfer Metadata<br/><br/>vLLM: kv_transfer_params<br/>SGLang: bootstrap_info<br/>TensorRT-LLM: opaque_state"]
        DECSELECT["Select Decode Worker<br/>decode load / capacity<br/>conditional cache locality"]
        DECPOOL["Decode Worker Pool<br/>independently scalable"]
        DEC1["Decode Worker D1"]
        DEC2["Decode Worker D2"]
        KVTRANSFER["Asynchronous KV Transfer<br/>Prefill GPU to Decode GPU"]
        DECEXEC["Autoregressive Decode<br/>continuous batching<br/>token generation"]
        DECOUTPUT["Streaming Engine Output"]

        ORCHESTRATOR --> PREFSELECT
        PREFSELECT --> PREFPOOL
        PREFPOOL --> PREF1
        PREFPOOL --> PREF2
        PREF1 --> PREFEXEC
        PREF2 --> PREFEXEC
        PREFEXEC --> META

        META --> DECSELECT
        DECSELECT --> DECPOOL
        DECPOOL --> DEC1
        DECPOOL --> DEC2

        META --> KVTRANSFER
        PREFEXEC --> KVTRANSFER
        KVTRANSFER --> DECEXEC
        DEC1 --> DECEXEC
        DEC2 --> DECEXEC
        DECEXEC --> DECOUTPUT
    end

    RESULT -->|"disaggregated topology"| ORCHESTRATOR
    DECOUTPUT --> ENGINE

    %% ============================================================
    %% 8. Backend engines
    %% ============================================================
    subgraph BACKENDS["7. Inference Backend Integration"]
        BACKENDAPI["Dynamo Backend Adapter<br/>common generate and streaming contract"]
        VLLM["vLLM Engine<br/>scheduler / paged attention<br/>KV event publisher"]
        SGLANG["SGLang Engine<br/>radix cache / HiCache<br/>KV event publisher"]
        TRTLLM["TensorRT-LLM Engine<br/>optimized kernels<br/>KV event publisher"]
        MOCKER["Mocker Engine<br/>simulation and testing"]
        GPU["GPU Execution<br/>CUDA kernels<br/>attention / GEMM / collectives"]
        PARALLEL["Parallelism<br/>TP / PP / DP / EP"]
        NCCL["NCCL Collectives<br/>multi-GPU execution"]

        BACKENDAPI --> VLLM
        BACKENDAPI --> SGLANG
        BACKENDAPI --> TRTLLM
        BACKENDAPI --> MOCKER

        VLLM --> GPU
        SGLANG --> GPU
        TRTLLM --> GPU

        GPU --> PARALLEL
        PARALLEL --> NCCL
    end

    AGGWORKER --> BACKENDAPI
    PREF1 --> BACKENDAPI
    PREF2 --> BACKENDAPI
    DEC1 --> BACKENDAPI
    DEC2 --> BACKENDAPI

    %% ============================================================
    %% 9. KV event plane and router index
    %% ============================================================
    subgraph KVEVENTS["8. KV Cache Event and Index Plane"]
        KVEVENT["RouterEvent<br/>Stored / Removed / Cleared<br/>worker_id / event_id / block hashes"]
        EVENTTRANSPORT{"KV Event Transport"}
        NATS["NATS / JetStream<br/>event pub-sub"]
        ZMQ["ZMQ Event Channel"]
        PREDICT["Prediction-based Tracking<br/>when KV events are disabled"]
        LISTENER["KV Event Listener"]
        INDEXER["Global KV Location Index<br/>block prefix to workers"]
        RADIX["Concurrent Radix Tree<br/>prefix overlap lookup"]
        CUCKOO["Optional Cuckoo Index<br/>high-throughput lookup"]
        REPLICASYNC["Router Replica Synchronization<br/>synchronize routing view<br/>not model-data replication"]
        RECOVERY["Indexer Recovery<br/>rebuild routing state after restart"]

        KVEVENT --> EVENTTRANSPORT
        EVENTTRANSPORT --> NATS
        EVENTTRANSPORT --> ZMQ
        NATS --> LISTENER
        ZMQ --> LISTENER
        PREDICT --> INDEXER
        LISTENER --> INDEXER
        INDEXER --> RADIX
        INDEXER --> CUCKOO
        INDEXER <--> REPLICASYNC
        RECOVERY --> INDEXER
    end

    VLLM -. "KV lifecycle events" .-> KVEVENT
    SGLANG -. "KV lifecycle events" .-> KVEVENT
    TRTLLM -. "KV lifecycle events" .-> KVEVENT

    RADIX -. "overlap scores" .-> MATCH
    CUCKOO -. "overlap scores" .-> MATCH

    %% ============================================================
    %% 10. KV Block Manager and storage tiers
    %% ============================================================
    subgraph KVBM["9. KV Block Manager and Cache Hierarchy"]
        CONNECTOR["Backend KV Connector<br/>vLLM / TensorRT-LLM integration"]
        LOGICAL["Logical Block Manager<br/>block identity<br/>ownership and lifecycle"]
        PHYSICAL["Physical Block Manager<br/>allocation / handles / layout"]
        DEVICECACHE["G1 Device Pool<br/>GPU HBM<br/>fastest tier"]
        HOSTCACHE["G2 Host Pool<br/>CPU pinned memory"]
        DISKCACHE["G3 Disk Pool<br/>local NVMe / filesystem"]
        REMOTECACHE["G4 Remote Storage<br/>remote memory / filesystem<br/>object or cloud storage"]
        OFFLOAD["Offload and Onboard Scheduler<br/>Device to Host to Disk<br/>Disk or Host to Device"]
        EVICT["Eviction and Capacity Management"]
        KVCONSOLIDATE["KV Consolidator<br/>track and publish block state"]

        CONNECTOR --> LOGICAL
        LOGICAL --> PHYSICAL
        PHYSICAL --> DEVICECACHE
        PHYSICAL --> HOSTCACHE
        PHYSICAL --> DISKCACHE
        PHYSICAL --> REMOTECACHE

        DEVICECACHE <--> OFFLOAD
        HOSTCACHE <--> OFFLOAD
        DISKCACHE <--> OFFLOAD
        REMOTECACHE <--> OFFLOAD

        EVICT --> LOGICAL
        LOGICAL --> KVCONSOLIDATE
    end

    VLLM --> CONNECTOR
    TRTLLM --> CONNECTOR
    SGLANG -. "HiCache or backend-native integration" .-> HOSTCACHE
    KVCONSOLIDATE -. "store / remove events" .-> KVEVENT
    OVERLAP -. "tier-aware cache hits" .-> KVBM

    %% ============================================================
    %% 11. NIXL data movement
    %% ============================================================
    subgraph NIXL["10. NIXL Data Movement Layer"]
        NIXLAPI["NIXL Agent and Transfer API<br/>memory registration<br/>descriptor exchange<br/>async get / put"]
        NVLINK["NVLink / GPU P2P"]
        UCX["UCX / InfiniBand / RoCE"]
        MEMCPY["CUDA or Host memcpy"]
        GDS["GPUDirect Storage"]
        FILEIO["POSIX / Remote Filesystem I/O"]

        NIXLAPI --> NVLINK
        NIXLAPI --> UCX
        NIXLAPI --> MEMCPY
        NIXLAPI --> GDS
        NIXLAPI --> FILEIO
    end

    KVTRANSFER --> NIXLAPI
    OFFLOAD --> NIXLAPI
    NIXLAPI --> DEVICECACHE
    NIXLAPI --> HOSTCACHE
    NIXLAPI --> DISKCACHE
    NIXLAPI --> REMOTECACHE

    %% ============================================================
    %% 12. Fault handling
    %% ============================================================
    subgraph FAULT["11. Request Fault Handling"]
        CANCEL["Request Cancellation<br/>client disconnect propagation"]
        REJECT["Admission and Request Rejection<br/>load and capacity thresholds"]
        MIGRATE["Optional In-flight Migration<br/>move eligible request after failure"]
        RETRY["Reroute New Request<br/>avoid failed or inhibited worker"]
        RECOMPUTE["KV Cache Miss or Loss<br/>recompute prompt KV"]

        CANCEL --> AGGWORKER
        CANCEL --> PREFPOOL
        CANCEL --> DECPOOL
        REJECT --> ROUTERMODE
        MIGRATE --> DECPOOL
        RETRY --> ROUTERMODE
        RECOMPUTE --> PREFEXEC
    end

    HEALTH -. "worker failure" .-> RETRY
    INHIBIT -. "temporary suppression" .-> RETRY
    DECEXEC -. "eligible interrupted request" .-> MIGRATE

    %% ============================================================
    %% 13. Observability and autoscaling
    %% ============================================================
    subgraph OPS["12. Observability, Planning and Autoscaling"]
        METRICS["Metrics<br/>request rate<br/>TTFT / ITL / latency<br/>tokens per second<br/>KV usage<br/>queue depth"]
        TRACES["Distributed Tracing<br/>request_id / trace_id"]
        LOGS["Structured Logs"]
        PROM["Prometheus"]
        GRAFANA["Grafana"]
        PLANNER["Dynamo Planner<br/>SLA-driven scaling decisions"]
        PROFILER["Profiler and AIConfigurator<br/>offline deployment configuration"]
        SCALE["Scale Prefill and Decode Pools<br/>replica counts / GPU allocation"]

        METRICS --> PROM
        PROM --> GRAFANA
        METRICS --> PLANNER
        PROFILER --> PLANNER
        PLANNER --> SCALE
    end

    HTTP -. "HTTP metrics" .-> METRICS
    KVROUTER -. "routing metrics" .-> METRICS
    PREFPOOL -. "load metrics" .-> METRICS
    DECPOOL -. "load metrics" .-> METRICS
    NIXLAPI -. "transfer metrics" .-> METRICS
    ENGINE -. "request spans" .-> TRACES
    BACKENDAPI -. "engine logs" .-> LOGS

    SCALE -. "desired replicas" .-> PREFPOOL
    SCALE -. "desired replicas" .-> DECPOOL

    %% ============================================================
    %% 14. Kubernetes control plane
    %% ============================================================
    subgraph K8S["13. Kubernetes Deployment Control Plane"]
        USER["Operator / GitOps / kubectl"]
        DGDR["DynamoGraphDeploymentRequest<br/>model / hardware / SLA intent"]
        PROFILERCTRL["Profiler Controller<br/>generate candidate deployment"]
        DGD["DynamoGraphDeployment<br/>explicit component graph"]
        OPERATOR["Dynamo Kubernetes Operator<br/>reconcile desired state"]
        PODS["Pods and Services<br/>Frontend / Prefill / Decode / Backend"]
        AUTOSCALER["Scaling Adapter / Autoscaler"]
        GROVE["Optional Grove<br/>gang and topology-aware scheduling"]
        MODELSTORE["Model Storage / PVC / Object Store"]
        MODELEXPRESS["Optional ModelExpress<br/>fast GPU-to-GPU weight loading"]

        USER --> DGDR
        USER --> DGD
        DGDR --> PROFILERCTRL
        PROFILERCTRL --> DGD
        DGD --> OPERATOR
        OPERATOR --> PODS
        AUTOSCALER --> OPERATOR
        OPERATOR --> GROVE
        MODELSTORE --> PODS
        MODELEXPRESS --> PODS
    end

    PODS -. "runs components" .-> FRONTEND
    PODS -. "runs components" .-> PREFPOOL
    PODS -. "runs components" .-> DECPOOL
    SCALE -. "scaling recommendation" .-> AUTOSCALER
    PODS -. "worker registration" .-> HEALTH

    %% ============================================================
    %% Styling
    %% ============================================================
    classDef external fill:#eef4ff,stroke:#3167b1,color:#111;
    classDef frontend fill:#e8f7ff,stroke:#0783b5,color:#111;
    classDef runtime fill:#f0ebff,stroke:#6f42c1,color:#111;
    classDef routing fill:#fff3cd,stroke:#b8860b,color:#111;
    classDef compute fill:#e9f8e9,stroke:#2e8b57,color:#111;
    classDef cache fill:#fff0f5,stroke:#c94f7c,color:#111;
    classDef infra fill:#f2f2f2,stroke:#666,color:#111;
    classDef control fill:#ffe8df,stroke:#c65d2e,color:#111;

    class APP,OAI,KSERVE,GATEWAY,EPP external;
    class HTTP,GRPC,VALIDATE,TEMPLATE,TOKENIZER,PREPROCESS,ENGINE,POSTPROCESS,STREAM frontend;
    class DR,NS,COMP,ENDPOINT,CLIENTWATCH,REQUESTPLANE,INHIBIT runtime;
    class ROUTERMODE,RR,RANDOM,LEAST,DIRECT,DEVICE,KVROUTER,ROUTERREQ,ELIGIBLE,HASH,MATCH,OVERLAP,LOAD,SCORE,QUEUE,PICK,RESULT routing;
    class AGGWORKER,AGGPREFILL,AGGDECODE,AGGOUTPUT,ORCHESTRATOR,PREFSELECT,PREFPOOL,PREF1,PREF2,PREFEXEC,META,DECSELECT,DECPOOL,DEC1,DEC2,KVTRANSFER,DECEXEC,DECOUTPUT,BACKENDAPI,VLLM,SGLANG,TRTLLM,MOCKER,GPU,PARALLEL,NCCL compute;
    class KVEVENT,EVENTTRANSPORT,NATS,ZMQ,PREDICT,LISTENER,INDEXER,RADIX,CUCKOO,REPLICASYNC,RECOVERY,CONNECTOR,LOGICAL,PHYSICAL,DEVICECACHE,HOSTCACHE,DISKCACHE,REMOTECACHE,OFFLOAD,EVICT,KVCONSOLIDATE cache;
    class DISCOVERYMODE,K8SDISC,ETCD,REGISTRY,HEALTH,WORKERCONFIG,NIXLAPI,NVLINK,UCX,MEMCPY,GDS,FILEIO,METRICS,TRACES,LOGS,PROM,GRAFANA infra;
    class USER,DGDR,PROFILERCTRL,DGD,OPERATOR,PODS,AUTOSCALER,GROVE,MODELSTORE,MODELEXPRESS,PLANNER,PROFILER,SCALE,CANCEL,REJECT,MIGRATE,RETRY,RECOMPUTE control;
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

