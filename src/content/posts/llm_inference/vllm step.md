---
title: "vllm step"
published: 2026-09-26
description: "vllm step"
image: ""
tags: ["llm_inference","vllm step"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-26T14:42:54.656.756822401Z"
---

# 完整 step

```python
def step(self):                                                   # 一次 Engine 迭代
    scheduler_output = self.scheduler.schedule(...)               # 1. 决定本轮计算什么
    future = self.model_executor.execute_model(                   # 2. 提交模型 forward
        scheduler_output, non_block=True
    )
    grammar_output = self.scheduler.get_grammar_bitmask(...)      # 3. 准备采样约束

    model_output = future.result()                                # 4. 等待 forward 完成
    if model_output is None:
        model_output = self.model_executor.sample_tokens(...)     # 5. 必要时执行采样

    self._process_aborts_queue()                                  # 6. 处理本轮发生的取消
    outputs = self.scheduler.update_from_output(                  # 7. 用结果更新请求状态
        scheduler_output, model_output
    )
    return outputs, ...                                           # 8. 本轮状态闭环完成
```

```
一个完整的 Engine step
├── 输入：上一步结束后的请求状态
│
├── schedule
│   └── 决定本轮计算哪些 token
│
├── execute_model
│   └── 真正执行这些 token 的 forward
│
├── sample_tokens
│   └── 根据 logits 得到新 token（如果本轮需要）
│
├── update_from_output
│   └── 把计算结果写回 Scheduler 状态
│
└── 输出：下一步可以继续使用的一致状态
```



# `SchedulerOutput` 

`SchedulerOutput` 是 Scheduler 发给 Worker 的“本轮执行清单”。它描述：

```
这轮有哪些请求
每个请求计算几个 token
使用哪些 KV blocks
哪些请求结束/抢占
是否有 Spec Decode、图片、Grammar、KV 传输
```

源码：[output.py (line 209)](/data/home/xli49/lxy/vllm/vllm/v1/core/sched/output.py:209)

## 数值场景

本轮有四个请求：

```
A：第一次调度，Prompt 共 6 tokens，本轮 Prefill 4 tokens
B：之前已经运行过，本轮验证 3 个 draft tokens
C：上一轮结束，需要通知 Worker 清理
D：本轮因为 KV Cache 不足被 preempt

本轮总 token 预算：8
```

所以：

```
A：4 tokens
B：4 tokens = 1 个基础位置 + 3 个 draft tokens
总计：8 tokens
```

下面 Connector metadata 的内容是简化示意，因为不同 Connector 的具体类型不同。

```
scheduler_output = SchedulerOutput(#Scheduler 给 Worker 的本轮执行计划
    scheduled_new_reqs=[#第一次发送给 Worker 的新请求
        NewRequestData(#请求 A 的完整初始化数据
            req_id="A",#请求唯一 ID
            prompt_token_ids=[11, 12, 13, 14, 15, 16],#A 的完整 Prompt，共 6 个 token
            mm_features=[#A 的多模态输入
                "image_feature_0",#第 0 张图片的特征描述
                "image_feature_1",#第 1 张图片的特征描述
            ],
            sampling_params=SamplingParams(max_tokens=20),#A 最多生成 20 个输出 token
            pooling_params=None,#A 是生成请求，不是 pooling 请求
            block_ids=([40],),#KV group 0 当前使用 physical block 40
            num_computed_tokens=0,#调度前 A 尚未计算任何 token
            lora_request=None,#A 不使用 LoRA
            prompt_embeds=None,#输入是 token IDs，不是直接提供 embedding
            prompt_is_token_ids=[True] * 6,#6 个 Prompt 位置全部来自 token IDs
            prefill_token_ids=[11, 12, 13, 14, 15, 16],#MRV2 Worker 保存的完整输入 token 序列
        ),
    ],

    scheduled_cached_reqs=CachedRequestData(#Worker 已缓存过的请求，只发送变化量
        req_ids=["B"],#本轮再次调度已有请求 B
        resumed_req_ids=set(),#B 没有从 preemption 中恢复
        new_token_ids=[],#PP=1 时不需要由 Scheduler 补发 token
        all_token_ids={},#MRV1 专用；MRV2 中保持为空
        new_block_ids=[([41],)],#B 本轮新增 KV group 0 的 physical block 41
        num_computed_tokens=[50],#调度前 B 已经计算了 50 个位置
        num_output_tokens=[8],#B 当前已经生成了 8 个输出 token
    ),

    num_scheduled_tokens={#本轮每个请求送入模型的位置数量
        "A": 4,#A 本轮 prefill 前 4 个 Prompt token
        "B": 4,#B 本轮计算 1 个基础位置和 3 个 draft token
    },

    total_num_scheduled_tokens=8,#本轮总计算位置数：A 的 4 + B 的 4

    scheduled_spec_decode_tokens={#本轮 Spec Decode 的 draft token
        "B": [501, 502, 503],#Draft Model 为 B 提议了 3 个 token
    },

    scheduled_encoder_inputs={#本轮需要处理的多模态 Encoder 输入
        "A": [0, 1],#A 需要处理第 0、1 张图片
    },

    num_common_prefix_blocks=[2],#KV group 0 中所有运行请求共有 2 个前缀 block

    finished_req_ids={"C"},#通知 Worker 删除已结束请求 C 的缓存状态

    free_encoder_mm_hashes=[#本轮需要释放的 Encoder cache
        "image_hash_old_001",#释放该图片哈希对应的 Encoder 输出
    ],

    scheduled_encoder_input_stats=ScheduledEncoderInputStats(#本轮 Encoder 计算统计
        num_inputs=2,#本轮处理 2 个 Encoder 输入
        output_tokens=576,#Encoder 共输出 576 个 embedding token
    ),

    preempted_req_ids={"D"},#D 因资源不足在本轮被抢占

    has_structured_output_requests=True,#本轮包含 JSON Schema/Grammar 请求

    pending_structured_output_tokens=False,#Grammar 所需的历史输出已经准备完成

    num_invalid_spec_tokens={#不应计入接受率的无效 draft token 数
        "B": 1,#B 有 1 个 draft token 无效
    },

    kv_connector_metadata=KVConnectorMetadata(#KV Connector 的本轮传输计划
        reqs_to_recv=["B"],#B 需要接收远程 KV Cache
    ),

    ec_connector_metadata=ECConnectorMetadata(#Encoder Cache Connector 的本轮传输计划
        reqs_to_recv=["A"],#A 需要接收远程图片 Encoder 输出
    ),

    ec_manager_metadata=EncoderCacheManagerMetadata(),#Encoder Cache Manager 的内部管理信息

    new_block_ids_to_zero=[40, 41, 42],#forward 前清零新分配的 physical blocks

    kv_cache_block_copies=[#清零完成后执行的 KV Cache CoW 复制
        KVCacheBlockCopy(#一条 block 复制操作
            src_block_id=12,#从已有 block 12 读取 KV
            dst_block_id=42,#将 KV 复制到新 block 42
        ),
    ],

    partial_tail_offloads={#交给外部 KV Connector 的 partial-tail 信息
        "A": [
            (0, 40, 6),#KV group 0、block 40、有效边界为 6 token
        ],
    },

    num_spec_tokens_to_schedule=4,#Dynamic Spec Decode 为下一 step 选择 K=4
)
```

## Worker需要的

```
SchedulerOutput
├── 请求数据
│   ├── scheduled_new_reqs
│   └── scheduled_cached_reqs
│
├── 本轮模型工作量
│   ├── num_scheduled_tokens = {"A":4, "B":4}
│   └── total_num_scheduled_tokens = 8
│
├── 特殊计算
│   ├── scheduled_spec_decode_tokens
│   ├── scheduled_encoder_inputs
│   └── Grammar 标志
│
├── 生命周期变化
│   ├── finished_req_ids = {"C"}
│   └── preempted_req_ids = {"D"}
│
└── Cache 操作
    ├── new_block_ids_to_zero
    ├── kv_cache_block_copies
    ├── kv_connector_metadata
    └── ec_connector_metadata
```

## `scheduled_new_reqs` 与 `scheduled_cached_reqs`

区别是传输数据量：

```
A 第一次出现：
Scheduler → Worker 发送完整信息
├── Prompt token IDs
├── SamplingParams
├── KV block IDs
├── 多模态输入
└── LoRA 信息

B 之前出现过：
Worker 已经缓存 B
Scheduler 只发送变化
├── 新 block IDs
├── num_computed_tokens
├── 新 token
└── 输出 token 数
```

这样每个 step 不需要反复传输完整 Prompt。完整 Prompt 保存在每个 Worker 的 `self.req_states.all_token_ids` 中。

## GrammarOutput 数值例子

假设请求 B 要求结构化输出，当前 Grammar 只允许 token ID：

```
2 和 5
```

为了简化，假设词表不超过 32 个 token，一个 `int32` 就能表示整个允许集合：

```
grammar_output = GrammarOutput(
    # bitmask 的第 0 行属于请求 B。
    structured_output_request_ids=[
        "B",
    ],

    # bit 2 和 bit 5 为 1。
    # (1 << 2) | (1 << 5) = 4 + 32 = 36。
    grammar_bitmask=np.array(
        [
            [36],
        ],
        dtype=np.int32,
    ),
)
```

二进制：

```
token ID： 7 6 5 4 3 2 1 0
允许位：   0 0 1 0 0 1 0 0

bitmask = 0b00100100 = 36
```

对应：

```
token 2：允许
token 5：允许
其他 token：禁止
```

Sampler 应用后近似：

```
logits[2] = 原值
logits[5] = 原值

logits[其他token] = -inf
```

真实模型词表可能有 128,000 个 token，所以 bitmask 会有很多个 `int32`：

```
grammar_bitmask.shape = [
    structured_request_count,
    ceil(vocab_size / 32),
]
```

例如：

```
structured request 数 = 3
vocab_size = 128000

shape = [3, 4000]
```

`structured_output_request_ids` 的顺序必须和 bitmask 行顺序一致：

```
structured_output_request_ids[0]
↔ grammar_bitmask[0]

structured_output_request_ids[1]
↔ grammar_bitmask[1]
```

## `make_empty()`

没有请求需要执行时：

```
empty = SchedulerOutput.make_empty()
```

它等价于：

```
SchedulerOutput(
    # 没有新请求。
    scheduled_new_reqs=[],

    # 没有已缓存请求的增量更新。
    scheduled_cached_reqs=CachedRequestData.make_empty(),

    # 没有请求被调度。
    num_scheduled_tokens={},

    # 本轮计算 0 个 token。
    total_num_scheduled_tokens=0,

    # 没有 Spec Decode tokens。
    scheduled_spec_decode_tokens={},

    # 没有 Encoder 输入。
    scheduled_encoder_inputs={},

    # 没有公共 prefix block 信息。
    num_common_prefix_blocks=[],

    # 没有新结束的请求。
    finished_req_ids=set(),

    # 没有 Encoder cache 需要释放。
    free_encoder_mm_hashes=[],

    # 使用 dataclass 默认值 None。
    scheduled_encoder_input_stats=None,

    # 使用默认值 None。
    preempted_req_ids=None,

    # 使用默认值 False。
    has_structured_output_requests=False,

    # 使用默认值 False。
    pending_structured_output_tokens=False,

    # 使用默认值 None。
    num_invalid_spec_tokens=None,

    # 使用默认值 None。
    kv_connector_metadata=None,

    # 使用默认值 None。
    ec_connector_metadata=None,

    # 使用默认值 None。
    ec_manager_metadata=None,

    # 使用默认值 None。
    new_block_ids_to_zero=None,

    # 使用默认值 None。
    kv_cache_block_copies=None,

    # 使用默认值 None。
    partial_tail_offloads=None,

    # 使用默认值 0。
    num_spec_tokens_to_schedule=0,
)
```

一句话：

>   `SchedulerOutput` 不是模型输出，而是 Scheduler 为 Worker 生成的本轮执行命令；`GrammarOutput` 则是采样阶段用于限制合法 token 的位图。
