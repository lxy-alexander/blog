---
title: "vLLM RL"
published: 2026-09-18
description: "vLLM RL"
image: ""
tags: ["llm_inference","vLLM RL"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-19T03:00:49.708.672816897Z"
---

```mermaid
flowchart TB

    subgraph L1["第一层：外部 RL 训练与编排层"]
        direction LR

        RL["RL 算法<br/>PPO / GRPO / DAPO"]
        TRAIN["训练模型<br/>Forward / Backward / Optimizer"]
        SOURCE["WeightSource<br/>ModuleSource / 自定义 Source"]
        TFACTORY["WeightTransferTrainerFactory"]
        TENGINE["TrainerWeightTransferEngine<br/>NCCL / IPC / Sparse NCCL / Sharded RDT"]
        CLIENT["VLLMWeightSyncClient<br/>HTTP / Ray / 本地适配器"]

        RL --> TRAIN
        TRAIN --> SOURCE
        SOURCE --> TFACTORY
        TFACTORY --> TENGINE
        TENGINE --> CLIENT
    end

    subgraph L2["第二层：vLLM 控制面入口层"]
        direction LR

        HTTP["HTTP RLHF Router<br/>/pause<br/>/resume<br/>/init_weight_transfer_engine<br/>/start_weight_update<br/>/update_weights<br/>/finish_weight_update"]

        RAY["Ray Actor 接口"]
        LOCAL["LLM / AsyncLLM 本地接口"]
        ENGINE_API["EngineClient / AsyncLLM / LLM"]

        HTTP --> ENGINE_API
        RAY --> ENGINE_API
        LOCAL --> ENGINE_API
    end

    subgraph L3["第三层：vLLM Engine 与调度层"]
        direction LR

        SCHEDULER["EngineCore Scheduler<br/>abort / wait / keep"]
        DP_PAUSE["DPEP 两阶段暂停<br/>本地暂停 → 全局一致"]
        RPC["collective_rpc<br/>向所有推理 Worker 广播命令"]

        SCHEDULER --> DP_PAUSE
        ENGINE_API --> SCHEDULER
        ENGINE_API --> RPC
    end

    subgraph L4["第四层：推理 Worker 与权重传输层"]
        direction LR

        WORKER["GPUWorker"]
        WFACTORY["WeightTransferEngineFactory"]
        WENGINE["推理侧 WeightTransferEngine"]

        NCCL["NCCL<br/>独立训练/推理 GPU"]
        IPC["CUDA IPC<br/>训练推理共置"]
        SPARSE["Sparse NCCL<br/>稀疏参数 Patch"]
        RDT["Sharded RDT<br/>按 Worker 拉取权重切片"]

        RPC --> WORKER
        WORKER --> WFACTORY
        WFACTORY --> WENGINE

        WENGINE --> NCCL
        WENGINE --> IPC
        WENGINE --> SPARSE
        WENGINE --> RDT
    end

    subgraph L5["第五层：模型执行与 MoE 推理层"]
        direction LR

        PARAM["模型参数<br/>Model Weights"]
        RUNNER["GPUModelRunner"]
        MOE["MoE Backend<br/>Triton / CUTLASS / DeepGEMM / Humming 等"]
        OUTPUT["Rollout 输出<br/>Tokens / Logprobs / Routed Experts"]

        RUNNER --> PARAM
        RUNNER --> MOE
        MOE --> OUTPUT
    end

    CLIENT -->|"控制面<br/>HTTP / Ray RPC"| HTTP
    CLIENT -->|"Ray 调用"| RAY
    CLIENT -->|"进程内调用"| LOCAL

    TENGINE ==>|"权重数据面<br/>NCCL / IPC / Sparse / RDT"| WENGINE
    SCHEDULER -->|"恢复后执行推理"| RUNNER
    WENGINE -->|"更新或覆盖权重"| PARAM
    OUTPUT -->|"Rollout 返回训练框架"| RL

    classDef external fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#111;
    classDef control fill:#e0f2fe,stroke:#0369a1,stroke-width:2px,color:#111;
    classDef engine fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#111;
    classDef transfer fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#111;
    classDef inference fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#111;

    class RL,TRAIN,SOURCE,TFACTORY,TENGINE,CLIENT external;
    class HTTP,RAY,LOCAL,ENGINE_API control;
    class SCHEDULER,DP_PAUSE,RPC engine;
    class WORKER,WFACTORY,WENGINE,NCCL,IPC,SPARSE,RDT transfer;
    class PARAM,RUNNER,MOE,OUTPUT inference;

    style L1 fill:#faf5ff,stroke:#7e22ce,stroke-width:3px
    style L2 fill:#f0f9ff,stroke:#0369a1,stroke-width:3px
    style L3 fill:#f0fdf4,stroke:#15803d,stroke-width:3px
    style L4 fill:#fffbeb,stroke:#b45309,stroke-width:3px
    style L5 fill:#fff1f2,stroke:#b91c1c,stroke-width:3px
```

```mermaid
flowchart TB

    %% ==================== L0 ====================
    subgraph L0["L0：外部 RL 训练系统"]
        direction LR
        ORCH["Rollout Orchestrator<br/>verl / Prime-RL / NeMo-RL / 自研框架"]
        REWARD["Reward / Verifier"]
        ADV["Advantage 计算<br/>GAE / GRPO"]
        TRAINER["Policy Trainer<br/>Loss / Backward / Optimizer"]
        TRAINMODEL["Trainer Policy Model<br/>新版本模型参数"]

        ORCH --> REWARD
        REWARD --> ADV
        ADV --> TRAINER
        TRAINER --> TRAINMODEL
    end

    %% ==================== L1 ====================
    subgraph L1["L1：vLLM 对外入口层"]
        direction LR
        CHATAPI["Rollout API<br/>/v1/chat/completions"]
        COMPLETIONAPI["Rollout API<br/>/v1/completions"]
        RESPAPI["Rollout API<br/>/v1/responses"]
        PYAPI["Python API<br/>LLM.generate<br/>AsyncLLM.generate"]
        RLAPI["RL 控制 API<br/>/pause /resume<br/>/start_weight_update<br/>/update_weights<br/>/finish_weight_update"]
        SLEEPAPI["显存控制 API<br/>/sleep /wake_up"]
        RAYAPI["Ray Actor API<br/>RayVLLMWeightSyncClient"]
    end

    %% ==================== L2 ====================
    subgraph L2["L2：Frontend 与 Engine Client 层"]
        direction LR
        OPENAISERVE["OpenAI Serving<br/>协议解析 / Streaming"]
        RENDERER["Renderer / Tokenizer<br/>Chat Template"]
        INPUTPROC["InputProcessor<br/>输入校验 / Tokenize<br/>构造 EngineCoreRequest"]
        ASYNCLLM["AsyncLLM / LLMEngine<br/>EngineClient 实现"]
        OUTPUTPROC["OutputProcessor<br/>Detokenize / Logprobs"]
        COLLECTOR["RequestOutputCollector<br/>每请求输出队列"]
    end

    %% ==================== L3 ====================
    subgraph L3["L3：EngineCore 控制与调度层"]
        direction LR
        CORECLIENT["EngineCoreClient<br/>进程内或 ZeroMQ IPC"]
        ENGINECORE["EngineCore<br/>请求生命周期<br/>Utility RPC"]
        SCHEDULER["Scheduler<br/>Continuous Batching<br/>Token Budget / PauseState"]
        KVMANAGER["KV Cache Manager<br/>Block 分配 / Prefix Cache"]
        VERSION["Weight Version Manager<br/>当前提交版本"]
    end

    %% ==================== L4 ====================
    subgraph L4["L4：分布式 Executor 层"]
        direction LR
        EXECUTOR["Executor 抽象"]
        UNIPROC["UniProcExecutor"]
        MULTIPROC["MultiprocExecutor"]
        RAYEXEC["RayDistributedExecutor"]
        CRPC["collective_rpc<br/>向全部 Worker 广播"]
        WORKERGROUP["GPUWorker Group<br/>TP × PP × DP"]

        EXECUTOR --> UNIPROC
        EXECUTOR --> MULTIPROC
        EXECUTOR --> RAYEXEC
        CRPC --> WORKERGROUP
        UNIPROC --> WORKERGROUP
        MULTIPROC --> WORKERGROUP
        RAYEXEC --> WORKERGROUP
    end

    %% ==================== L5 ====================
    subgraph L5["L5：GPU Worker 与 ModelRunner 层"]
        direction LR
        GPUWORKER["GPUWorker<br/>每个分布式 Rank 一个"]
        MODELRUNNER["GPUModelRunner<br/>Batch 准备 / CUDA Graph"]
        FORWARD["Policy Model Forward"]
        ATTENTION["Attention Backend"]
        MOE["MoE / Expert Routing"]
        SAMPLER["Sampler<br/>Token / Logprobs"]
        RUNNEROUT["ModelRunnerOutput<br/>Token / Logprobs<br/>Routed Experts"]

        GPUWORKER --> MODELRUNNER
        MODELRUNNER --> FORWARD
        FORWARD --> ATTENTION
        FORWARD --> MOE
        ATTENTION --> SAMPLER
        MOE --> SAMPLER
        SAMPLER --> RUNNEROUT
    end

    %% ==================== L6 ====================
    subgraph L6["L6：GPU 状态与模型存储层"]
        direction LR
        POLICYMODEL["Inference Policy Model<br/>当前 Rollout 参数"]
        KVCACHE["GPU KV Cache"]
        PREFIXCACHE["Prefix Cache"]
        BLOCKTABLE["Block Tables"]
        ROUTECAPTURE["Routed Experts Capturer"]
        SLEEPBACKEND["SleepModeBackend<br/>默认 CuMemBackend"]
        MEMORY["GPU Memory Pool"]

        POLICYMODEL <--> KVCACHE
        KVCACHE <--> BLOCKTABLE
        KVCACHE <--> PREFIXCACHE
        POLICYMODEL --> ROUTECAPTURE
        SLEEPBACKEND <--> MEMORY
    end

    %% ==================== L7 ====================
    subgraph L7["L7：Trainer 侧权重传输层"]
        direction LR
        WEIGHTSOURCE["WeightSource<br/>ModuleSource / Sparse Patch<br/>Sharded Weight Groups"]
        TRAINFACTORY["WeightTransferTrainerFactory"]
        TRAINENGINE["TrainerWeightTransferEngine"]
        SYNCCLIENT["VLLMWeightSyncClient<br/>HTTP / Ray"]
        DATAPLANE["模型张量数据面<br/>NCCL / CUDA IPC / NIXL RDT"]

        WEIGHTSOURCE --> TRAINFACTORY
        TRAINFACTORY --> TRAINENGINE
        TRAINENGINE --> SYNCCLIENT
        TRAINENGINE --> DATAPLANE
    end

    %% ==================== L8 ====================
    subgraph L8["L8：vLLM Worker 权重接收层"]
        direction LR
        WTCONFIG["WeightTransferConfig<br/>选择 Backend"]
        WORKFACTORY["WeightTransferEngineFactory"]
        WTSESSION["Weight Update Session<br/>Inactive / Active"]
        NCCLENGINE["NCCLWeightTransferEngine"]
        IPCENGINE["IPCWeightTransferEngine"]
        SPARSEENGINE["SparseNCCLWeightTransferEngine"]
        RDTENGINE["ShardedRDTWeightTransferEngine"]
        LAYERRELOAD["Layerwise Reload<br/>接收 / 转换 / 恢复 Storage"]
        COMMIT["finish_weight_update<br/>提交 Weight Version"]

        WTCONFIG --> WORKFACTORY
        WORKFACTORY --> NCCLENGINE
        WORKFACTORY --> IPCENGINE
        WORKFACTORY --> SPARSEENGINE
        WORKFACTORY --> RDTENGINE
        WTSESSION --> WORKFACTORY
        NCCLENGINE --> LAYERRELOAD
        IPCENGINE --> LAYERRELOAD
        SPARSEENGINE --> LAYERRELOAD
        RDTENGINE --> LAYERRELOAD
        LAYERRELOAD --> COMMIT
    end

    %% ==================== Rollout 主路径 ====================
    ORCH -->|"Prompt + SamplingParams"| CHATAPI
    ORCH --> COMPLETIONAPI
    ORCH --> RESPAPI
    ORCH --> PYAPI

    CHATAPI --> OPENAISERVE
    COMPLETIONAPI --> OPENAISERVE
    RESPAPI --> OPENAISERVE
    PYAPI --> ASYNCLLM

    OPENAISERVE --> RENDERER
    RENDERER --> INPUTPROC
    INPUTPROC --> ASYNCLLM
    ASYNCLLM --> CORECLIENT

    CORECLIENT --> ENGINECORE
    ENGINECORE --> SCHEDULER
    SCHEDULER <--> KVMANAGER
    SCHEDULER --> EXECUTOR

    WORKERGROUP --> GPUWORKER
    MODELRUNNER --> POLICYMODEL
    ATTENTION <--> KVCACHE
    MOE --> ROUTECAPTURE

    RUNNEROUT --> SCHEDULER
    SCHEDULER --> ENGINECORE
    ENGINECORE --> CORECLIENT
    CORECLIENT --> OUTPUTPROC
    OUTPUTPROC --> COLLECTOR
    COLLECTOR --> OPENAISERVE
    OPENAISERVE -->|"Rollout Result"| ORCH

    %% ==================== 控制路径 ====================
    ORCH -->|"Pause / Resume"| RLAPI
    ORCH -->|"Sleep / Wake"| SLEEPAPI
    RLAPI --> ASYNCLLM
    SLEEPAPI --> ASYNCLLM
    ASYNCLLM --> CRPC
    GPUWORKER --> SLEEPBACKEND

    %% ==================== 权重更新路径 ====================
    TRAINMODEL --> WEIGHTSOURCE
    SYNCCLIENT --> RLAPI
    SYNCCLIENT --> RAYAPI
    RAYAPI --> ASYNCLLM

    DATAPLANE --> NCCLENGINE
    DATAPLANE --> IPCENGINE
    DATAPLANE --> SPARSEENGINE
    DATAPLANE --> RDTENGINE

    CRPC --> WTSESSION
    COMMIT --> VERSION
    COMMIT --> POLICYMODEL

    %% ==================== 每层外框样式 ====================
    style L0 fill:#FFF8E1,stroke:#F57F17,stroke-width:3px
    style L1 fill:#E3F2FD,stroke:#1565C0,stroke-width:3px
    style L2 fill:#E8EAF6,stroke:#3949AB,stroke-width:3px
    style L3 fill:#E0F2F1,stroke:#00796B,stroke-width:3px
    style L4 fill:#F3E5F5,stroke:#7B1FA2,stroke-width:3px
    style L5 fill:#FCE4EC,stroke:#C2185B,stroke-width:3px
    style L6 fill:#FFEBEE,stroke:#C62828,stroke-width:3px
    style L7 fill:#F1F8E9,stroke:#558B2F,stroke-width:3px
    style L8 fill:#FFF3E0,stroke:#EF6C00,stroke-width:3px
```

图中有三条主要路径：

-   Rollout 路径：`L0 → L1 → L2 → L3 → L4 → L5 → L6 → 原路返回`
-   控制路径：`RL API → AsyncLLM → EngineCore/collective_rpc → GPUWorker`
-   权重同步路径：`L0 Trainer → L7 → L1/L2 控制面 + 数据面 → L8 → L6 Policy Model`



这里的“第一层”指总图中的 `L0：外部 RL 训练系统`。它逻辑上不属于 vLLM 核心，但负责调用 vLLM、消费 rollout、训练模型，再把新参数同步回 vLLM。

当前仓库示例并没有真正执行 PPO/GRPO，而是用一个已经训练好的模型模拟“Optimizer 更新后的新权重”。

## 1. 第一层带数值展开

实线是仓库示例真实执行的路径，虚线是真实 GRPO/PPO 系统需要补充的路径。

````
```mermaid
flowchart LR

    subgraph L0["L0：外部 RL 训练系统"]
        direction LR

        ORCH["Rollout Orchestrator<br/>一次提交 13 个 Prompt<br/>每个 Prompt 生成 1 条 Response"]

        REQUEST["Rollout 配置<br/>temperature = 0<br/>max_tokens = 10 + 100 = 110<br/>最多生成 13 × 110 = 1430 Token"]

        RESULT["Rollout Results<br/>13 条生成结果<br/>记录 pause_idx<br/>区分新旧权重生成的 Token"]

        VERIFY["示例中的 Verifier<br/>对比新模型重新生成结果<br/>NVIDIA 要求 13/13 完全一致<br/>ROCm 至少 12/13 一致"]

        REWARD["真实 RL Reward<br/>示例没有实现<br/>例如：[1.0, 0.5, 0.0, -0.5]"]

        ADV["真实 RL Advantage<br/>示例没有实现<br/>GRPO 组内归一化"]

        TRAINER["真实 Policy Trainer<br/>Loss / Backward / Optimizer<br/>示例没有执行这些步骤"]

        DEMOTRAINER["示例 TrainModel Ray Actor<br/>占用 1 张 GPU<br/>直接加载 Qwen3-1.7B<br/>BF16"]

        POLICY["Trainer Policy Model<br/>约 1.7B 参数<br/>BF16 原始参数约 3.4 GB<br/>作为新权重 V1"]

        WEIGHTS["权重同步输出<br/>ModuleSource<br/>NCCL world_size = 2<br/>1 Trainer + 1 vLLM Worker"]

        ORCH --> REQUEST
        RESULT --> VERIFY
        DEMOTRAINER --> POLICY
        POLICY --> WEIGHTS

        RESULT -.-> REWARD
        REWARD -.-> ADV
        ADV -.-> TRAINER
        TRAINER -.-> POLICY
    end

    subgraph VLLM["vLLM 边界"]
        direction LR
        OLD["初始 Rollout Model<br/>Qwen3-1.7B-Base<br/>权重版本 V0"]
        PAUSE["任意请求达到 10 Token<br/>pause mode = keep"]
        SWAP["接收 Trainer 权重<br/>V0 → V1"]
        RESUME["恢复生成<br/>继续生成剩余 Token"]

        OLD --> PAUSE
        PAUSE --> SWAP
        SWAP --> RESUME
    end

    REQUEST --> OLD
    RESUME --> RESULT
    WEIGHTS --> SWAP

    style L0 fill:#FFF8E1,stroke:#F57F17,stroke-width:3px
    style VLLM fill:#E3F2FD,stroke:#1565C0,stroke-width:3px
```
````

## 2. Rollout Orchestrator：13 个并发请求

示例定义了 13 个 Prompt：

```
PROMPTS = [
    "The president of the United States is",
    "The capital of France is",
    ...
    "DNA stands for deoxyribonucleic acid and it",
]
```

对应代码：[rlhf_async_new_apis.py (line 217)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:217)

然后为每个 Prompt 创建一个异步 Ray 调用：

```
gen_futures = [
    llm.do_generate.remote(ptids, sampling_params)
    for ptids in batch_prompt_token_ids
]
```

对应代码：[rlhf_async_new_apis.py (line 258)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:258)

所以这里的数值是：

-   Prompt 数量：13
-   每个 Prompt 生成数量：默认 `n=1`
-   总 rollout 数量：`13 × 1 = 13`
-   13 条请求并发提交，不是一条执行完再执行下一条

实际工业 GRPO 通常会对每个 Prompt 生成多条结果。例如：

```
Prompt batch size = 128
每个 Prompt 生成 G = 8 条 Response
每轮 Rollout 数 = 128 × 8 = 1024
```

当前示例只是最小化演示，`G=1`，因此不能直接计算 GRPO 的组内相对 advantage。

## 3. Rollout 参数：每条最多生成 110 Token

代码中：

```
PAUSE_TOKEN_THRESHOLD = 10
N_NEW_TOKENS = 100

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=PAUSE_TOKEN_THRESHOLD + N_NEW_TOKENS,
)
```

对应位置：

-   [rlhf_async_new_apis.py (line 63)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:63)
-   [rlhf_async_new_apis.py (line 245)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:245)
-   [rlhf_async_new_apis.py (line 254)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:254)

代入数值：

```
max_tokens = 10 + 100 = 110
```

因此理论最大生成量：

```
13 个 Prompt × 110 Token = 1430 个生成 Token
```

`temperature=0` 表示贪心生成，主要用于保证更新前后结果能够确定性比较。真实 RL rollout 通常不会设成零，例如可能使用：

```
SamplingParams(
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
    logprobs=1,
)
```

因为 RL 需要从当前策略分布中采样，而不是永远选择概率最大的 token。

## 4. 为什么在 10 Token 时暂停

每个请求在流式生成期间统计已经生成的 token：

```
cur_token_count = len(output.outputs[0].token_ids)

if cur_token_count >= PAUSE_TOKEN_THRESHOLD:
    self._request_pause_flag = True
```

对应代码：[rlhf_async_new_apis.py (line 99)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:99)

任意一条请求达到 10 个 token 后：

```
await super().pause_generation(mode="keep")
```

对应代码：[rlhf_async_new_apis.py (line 111)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:111)

假设某条请求最后记录到：

```
pause_idx = 10
总输出长度 = 110
```

那么它的输出被划分为：

```
Token 0～9：旧权重 V0 生成，共 10 个
Token 10～109：新权重 V1 生成，共 100 个
```

但实际 `pause_idx` 不一定严格等于 10。因为请求是并发和分批执行的，pause 命令生效前可能又完成了一次 decode。例如：

```
请求 A：pause_idx = 10
请求 B：pause_idx = 12
请求 C：pause_idx = 9
```

代码因此为每个请求单独记录 `pause_idx`，而不是假设所有请求都是 10。

## 5. Reward / Verifier：示例只有验证，没有奖励

当前示例没有 Reward Model，也没有类似下面的逻辑：

```
reward = reward_model(prompt, response)
```

它使用“新鲜启动的 V2 模型是否能生成相同后缀”作为正确性验证：

```
expected = output.outputs[0].token_ids[pause_idx:]
actual = val_output.outputs[0].token_ids
match = actual == expected
```

对应代码：[rlhf_async_new_apis.py (line 316)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:316)

验证门槛为：

```
MIN_PASS_RATE = 1.0 if not current_platform.is_rocm() else 0.9
```

代入 13 个 Prompt：

-   NVIDIA：必须 `13/13 = 100%`
-   ROCm：至少需要 `12/13 ≈ 92.31%`
-   `11/13 ≈ 84.62%`，达不到 90%，会失败

这个 verifier 只是测试权重热更新是否正确，不是 RL reward。

## 6. Advantage：真实 GRPO 如何代入数值

假设真实 GRPO 对同一个 Prompt 生成 4 条结果，Reward 分别为：

```
r = [1.0, 0.5, 0.0, -0.5]
```

组内平均值：

```
mean = (1.0 + 0.5 + 0.0 - 0.5) / 4
     = 0.25
```

组内标准差：

```
std ≈ 0.559
```

GRPO 风格的归一化 advantage：

```
A_i = (r_i - mean) / std
```

代入后：

| Response | Reward | Advantage |
| -------- | ------ | --------- |
| 1        | 1.0    | 1.342     |
| 2        | 0.5    | 0.447     |
| 3        | 0.0    | -0.447    |
| 4        | -0.5   | -1.342    |

含义是：

-   `A > 0`：增加这些 token 的生成概率
-   `A < 0`：降低这些 token 的生成概率
-   这一步发生在外部 Trainer，不在 vLLM 内

## 7. Trainer：当前示例没有真正训练

示例中的 Trainer 是：

```
@ray.remote(num_gpus=1)
class TrainModel:
    ...
```

它占用 1 张 GPU，并直接加载：

```
self.model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype=torch.bfloat16,
).to("cuda:0")
```

对应代码：[rlhf_async_new_apis.py (line 120)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:120)

因此示例的真实情况是：

```
vLLM 初始模型 V0：Qwen/Qwen3-1.7B-Base
Trainer 模型 V1：Qwen/Qwen3-1.7B
```

它没有执行：

```
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

而是直接把 `Qwen3-1.7B` 当成“已经完成训练的新版本参数”。

对于约 17 亿参数、BF16 每参数 2 字节，单纯参数大小粗略为：

```
1.7 × 10⁹ × 2 bytes
= 3.4 × 10⁹ bytes
≈ 3.4 GB
≈ 3.17 GiB
```

实际显存还会包含：

-   临时通信 Buffer
-   CUDA/NCCL 状态
-   模型 Buffer
-   推理侧 KV Cache
-   真实训练时的梯度和 Optimizer State

所以真实训练显存会远大于 3.17 GiB。

## 8. 第一层最后输出什么

Trainer 模型通过：

```
source=ModuleSource(self.model)
```

变成可枚举的模型权重源，然后：

```
self.engine.send_weights()
```

触发权重同步。

示例数值：

```
Trainer Rank = 0
Trainer GPU = 1 张
vLLM Worker = 1 个
NCCL world_size = 2
packed = True
backend = nccl
```

代码位置：[rlhf_async_new_apis.py (line 138)](/data/home/xli49/lxy/vllm/examples/rl/rlhf_async_new_apis.py:138)

完整第一层的数据关系可以概括为：

```
13 个 Prompt
    ↓
最多 13 条 × 110 Token 的 Rollout
    ↓
示例：只验证结果，不计算 Reward
真实 RL：Reward → Advantage → Loss
    ↓
Optimizer 更新 Trainer Policy
    ↓
得到新模型参数 V1
    ↓
ModuleSource + NCCL
    ↓
同步给 vLLM
```

所以第一层真正更新的是“Trainer 里的模型参数”，不是 vLLM 内部的数据集。vLLM 生成 rollout 数据并接收训练完成后的新权重。



