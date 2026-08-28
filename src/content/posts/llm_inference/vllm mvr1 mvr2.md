---
title: "vllm mvr1 mvr2"
published: 2026-08-28
description: "vllm mvr1 mvr2"
image: ""
tags: ["llm_inference","vllm mvr1 mvr2"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-28T21:59:11.772.051269411Z"
---

下面按照架构图中的箭头，用同一组数值追踪一次真实 MRV1 step。

假设：

```
max_model_len = 8
KV block_size = 4
hidden_size = H
vocab_size = V

请求 A：chunked prefill
token_ids             = [11, 12, 13, 14, 15, 16]
num_computed_tokens   = 3
num_scheduled_tokens  = 2
block_ids             = [7, 9]

请求 B：decode
token_ids             = [21, 22, 23, 31, 32]
num_computed_tokens   = 4
num_scheduled_tokens  = 1
block_ids             = [12, 15]
```

## 1. Engine → Scheduler

Engine Core 的主循环位于 [core.py (line 608)](/data/home/xli49/lxy/vllm/vllm/v1/engine/core.py:608)：

```
scheduler_output = self.scheduler.schedule(
    self._should_throttle_prefills()
)

future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)
```

Scheduler 根据本轮 token budget，得到：

```
num_scheduled_tokens = {
    "A": 2,
    "B": 1,
}

total_num_scheduled_tokens = 3
```

概念上对应：

```
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=[],
    scheduled_cached_reqs=...,
    num_scheduled_tokens={
        "A": 2,
        "B": 1,
    },
    total_num_scheduled_tokens=3,
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    finished_req_ids=set(),
    ...
)
```

这里 Scheduler 只决定：

```
A 本轮计算 2 个 token
B 本轮计算 1 个 token
```

它不负责构造 GPU tensor。

------

## 2. GPU Worker → MRV1

GPU Worker 根据配置选择 MRV1，见
[gpu_worker.py (line 470)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_worker.py:470)：

```
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner as GPUModelRunnerV1,
)

self.model_runner = GPUModelRunnerV1(
    self.vllm_config,
    self.device,
)
```

Worker 最终调用：

```
output = self.model_runner.execute_model(
    scheduler_output,
    intermediate_tensors,
)
```

MRV1 主入口位于
[gpu_model_runner.py (line 4235)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:4235)。

------

## 3. `_update_states()`：更新持久 InputBatch

MRV1 首先执行：

```
deferred_state_corrections_fn = self._update_states(
    scheduler_output
)
```

对应 [gpu_model_runner.py (line 4273)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:4273)。

`InputBatch` 是 MRV1 持久保存的请求表，主要字段定义在
[gpu_input_batch.py (line 127)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_input_batch.py:127)：

```
self.token_ids_cpu_tensor = torch.zeros(
    (max_num_reqs, max_model_len),
    dtype=torch.int32,
)

self.num_computed_tokens_cpu_tensor = torch.zeros(
    (max_num_reqs,),
    dtype=torch.int32,
)
```

代入两个请求：

```
input_batch.req_ids = ["A", "B"]

input_batch.req_id_to_index = {
    "A": 0,
    "B": 1,
}

input_batch.token_ids_cpu = np.array([
    [11, 12, 13, 14, 15, 16,  0, 0],
    [21, 22, 23, 31, 32,  0,  0, 0],
], dtype=np.int32)

input_batch.num_computed_tokens_cpu = np.array([
    3,  # A
    4,  # B
])
```

请求加入 InputBatch 时，实际代码会复制 prompt 和 output token，见
[gpu_input_batch.py (line 368)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_input_batch.py:368)：

```
self.token_ids_cpu[
    req_index, :num_prompt_tokens
] = request.prompt_token_ids

self.token_ids_cpu[
    req_index, start_idx:end_idx
] = request.output_token_ids

self.num_computed_tokens_cpu[
    req_index
] = request.num_computed_tokens
```

`_update_states()` 还会处理：

-   完成请求：从 `requests` 和 `InputBatch` 删除。
-   暂停请求：从当前 batch 删除，但保留 cached state。
-   恢复请求：重新放入 InputBatch。
-   新分配 KV block。
-   speculative token。
-   sampling metadata。

------

## 4. `_prepare_inputs()`：构造 packed batch

MRV1 根据 InputBatch 和 SchedulerOutput 准备输入：

```
tokens = [
    scheduler_output.num_scheduled_tokens[i]
    for i in self.input_batch.req_ids
]

num_scheduled_tokens_np = np.array(
    tokens,
    dtype=np.int32,
)

logits_indices, spec_metadata, max_sampled = (
    self._prepare_inputs(
        scheduler_output,
        num_scheduled_tokens_np,
    )
)
```

本例：

```
num_scheduled_tokens_np = [2, 1]
```

核心实现在
[gpu_model_runner.py (line 1961)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:1961)。

### 4.1 计算 req_indices

实际代码：

```
req_indices = np.repeat(
    self.arange_np[:num_reqs],
    num_scheduled_tokens,
)
```

代入：

```
np.repeat(
    [0, 1],
    [2, 1],
)
```

得到：

```
req_indices = [0, 0, 1]
```

解释：

| packed index | req index | 请求 |
| ------------ | --------- | ---- |
| 0            | 0         | A    |
| 1            | 0         | A    |
| 2            | 1         | B    |

------

### 4.2 计算 query_pos 和 query_start_loc

MRV1 的 `_get_cumsum_and_arange()` 对 `[2,1]` 计算：

```
cu_num_tokens = np.cumsum([2, 1])
query_pos = concatenate([
    arange(2),
    arange(1),
])
```

结果：

```
cu_num_tokens = [2, 3]
query_pos = [0, 1, 0]
```

接着构造：

```
query_start_loc = [0, 2, 3]
```

对应源码：

```
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
```

见 [gpu_model_runner.py (line 2074)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:2074)。

请求在 packed tensor 中的范围为：

```
A: input_ids[0:2]
B: input_ids[2:3]
```

------

### 4.3 计算绝对 position

实际代码：

```
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[:cu_num_tokens[-1]]
)
```

代入：

```
num_computed_tokens = [3, 4]
req_indices = [0, 0, 1]

num_computed_tokens[req_indices]
= [3, 3, 4]

query_pos
= [0, 1, 0]
```

所以：

```
positions = [3, 4, 4]
```

对应关系：

| packed index | 请求 | input position |
| ------------ | ---- | -------------- |
| 0            | A    | 3              |
| 1            | A    | 4              |
| 2            | B    | 4              |

------

### 4.4 从 InputBatch 提取 input_ids

实际代码：

```
token_indices = (
    positions_np
    + req_indices * self.input_batch.token_ids_cpu.shape[1]
)

torch.index_select(
    self.input_batch.token_ids_cpu_tensor.flatten(),
    0,
    torch.from_numpy(token_indices),
    out=self.input_ids.cpu[:total_num_scheduled_tokens],
)
```

其中：

```
max_model_len = 8

token_indices = [
    3 + 0 * 8,
    4 + 0 * 8,
    4 + 1 * 8,
]
```

因此：

```
token_indices = [3, 4, 12]
```

扁平 token 表：

```
token_ids_cpu.flatten() == [
    11, 12, 13, 14, 15, 16, 0, 0,
    21, 22, 23, 31, 32,  0, 0, 0,
]
```

执行索引后：

```
input_ids = [14, 15, 32]
```

此时核心 GPU 输入是：

```
input_ids       = [14, 15, 32]
positions       = [ 3,  4,  4]
req_indices     = [ 0,  0,  1]
query_start_loc = [ 0,  2,  3]
```

------

## 5. 预分配 GPU Buffers

MRV1 不会每轮重新创建这些 tensor，而是使用预分配 buffer，再取有效 slice：

```
input_ids = self.input_ids.gpu[:num_tokens_padded]
positions = self.positions[:num_tokens_padded]
```

对本例，无 CUDA Graph padding 时：

```
num_tokens_unpadded = 3
num_tokens_padded = 3

input_ids.shape = [3]
positions.shape = [3]
```

如果 CUDA Graph 选择的固定 batch size 是 4，则可能变成：

```
num_tokens_unpadded = 3
num_tokens_padded = 4

input_ids = [14, 15, 32, PAD]
positions = [3, 4, 4, 0]
```

第四个 token 只是 CUDA Graph padding，不属于真实请求。

------

## 6. Attention Metadata

### 6.1 seq_lens

实际代码：

```
self.seq_lens[:num_reqs] = (
    self.num_computed_tokens[:num_reqs]
    + num_scheduled_tokens_gpu
)
```

见 [gpu_model_runner.py (line 2190)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:2190)。

代入：

```
seq_lens = [3, 4] + [2, 1]
         = [5, 5]
```

含义是本轮完成后：

```
A 的已计算 KV 长度 = 5
B 的已计算 KV 长度 = 5
```

### 6.2 KV slot_mapping

普通 PagedAttention 下：

```
logical_block = position // block_size
block_offset = position % block_size

physical_block = block_table[req_index][logical_block]

slot = physical_block * block_size + block_offset
```

本例 block table：

```
block_table = [
    [7, 9],    # A
    [12, 15],  # B
]
```

逐个 token 计算：

#### A，position 3

```
logical_block = 3 // 4 = 0
block_offset = 3 % 4 = 3
physical_block = block_table[0][0] = 7

slot = 7 * 4 + 3 = 31
```

#### A，position 4

```
logical_block = 4 // 4 = 1
block_offset = 4 % 4 = 0
physical_block = block_table[0][1] = 9

slot = 9 * 4 + 0 = 36
```

#### B，position 4

```
logical_block = 4 // 4 = 1
block_offset = 4 % 4 = 0
physical_block = block_table[1][1] = 15

slot = 15 * 4 + 0 = 60
```

最终：

```
slot_mapping = [31, 36, 60]
```

图中的：

```
Buffers --> Forward
AttnMeta --> Forward
```

现在具体对应：

```
input_ids       = [14, 15, 32]
positions       = [3, 4, 4]
query_start_loc = [0, 2, 3]
seq_lens        = [5, 5]
slot_mapping    = [31, 36, 60]
```

------

## 7. `_model_forward()` 与 KV Cache

MRV1 设置 forward context：

```
with set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    slot_mapping=slot_mappings,
):
    model_output = self._model_forward(
        input_ids=input_ids,
        positions=positions,
        ...
    )
```

对应 [gpu_model_runner.py (line 4493)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:4493)。

模型看到的逻辑输入：

```
packed token 0:
    request=A
    token_id=14
    position=3
    KV write slot=31

packed token 1:
    request=A
    token_id=15
    position=4
    KV write slot=36

packed token 2:
    request=B
    token_id=32
    position=4
    KV write slot=60
```

如果 `hidden_size=4096`：

```
hidden_states.shape = [3, 4096]
```

注意模型本轮只计算 3 个 token，但 attention 会通过 KV block table 读取历史 token 的 K/V。

------

## 8. hidden_states → logits

普通非 speculative decode 路径：

```
logits_indices = query_start_loc[1:] - 1
```

代入：

```
query_start_loc = [0, 2, 3]

logits_indices = [2, 3] - 1
               = [1, 2]
```

即：

```
sample_hidden_states = hidden_states[[1, 2]]
```

对应：

```
hidden_states[1] = A 最后一个本轮 token，即 token 15
hidden_states[2] = B 最后一个本轮 token，即 token 32
```

假设：

```
hidden_states.shape = [3, 4096]
sample_hidden_states.shape = [2, 4096]
vocab_size = 128000
```

那么：

```
logits = self.model.compute_logits(sample_hidden_states)
logits.shape == [2, 128000]
```

代码路径位于 `execute_model()` 的 postprocess 部分：

```
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

------

## 9. 为什么 `execute_model()` 返回 None

MRV1 完成 forward 和 logits 后，把中间结果暂存在：

```
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    ...
)

return None
```

然后 Engine Core 看到 `None`：

```
model_output = future.result()

if model_output is None:
    model_output = self.model_executor.sample_tokens(
        grammar_output
    )
```

这样 structured output 可以在 forward 和 sampling 之间生成 grammar mask。

------

## 10. logits → sample_tokens()

`sample_tokens()` 入口位于
[gpu_model_runner.py (line 4614)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu_model_runner.py:4614)。

普通采样路径：

```
sampler_output = self._sample(
    logits,
    spec_decode_metadata,
)
```

无 spec decode 时，`_sample()` 最终调用：

```
return self.sampler(
    logits=logits,
    sampling_metadata=sampling_metadata,
)
```

假设 greedy sampling：

```
argmax(logits[0]) = 99
argmax(logits[1]) = 33
```

得到：

```
sampled_token_ids = [
    [99],  # A
    [33],  # B
]
```

------

## 11. `_bookkeeping_sync()` 过滤结果

A 是 chunked prefill。

```
A 原始 token 数 = 6
A 本轮结束后的 seq_len = 3 + 2 = 5
```

所以：

```
discard_A = 5 < 6
          = True
```

B：

```
B 当前 token 数 = 5
B 本轮结束后的 seq_len = 4 + 1 = 5
```

所以：

```
discard_B = 5 < 5
          = False
```

最终：

```
A 采样出的 99：丢弃
B 采样出的 33：保留
```

`ModelRunnerOutput` 近似为：

```
ModelRunnerOutput(
    req_ids=["A", "B"],
    req_id_to_index={
        "A": 0,
        "B": 1,
    },
    sampled_token_ids=[
        [],     # A 的采样被过滤
        [33],   # B 的有效输出
    ],
    logprobs=...,
)
```

------

## 12. Scheduler 更新状态

Engine Core 调用：

```
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

更新后：

### 请求 A

```
token_ids:
[11, 12, 13, 14, 15, 16]

num_computed_tokens:
3 → 5

输出：
无
```

A 还剩 prompt token `16` 未计算。

### 请求 B

```
token_ids:
[21, 22, 23, 31, 32]
→
[21, 22, 23, 31, 32, 33]

num_computed_tokens:
4 → 5
```

下一轮 B 的模型输入是：

```
input_id = 33
position = 5
```

用 token 33 的 hidden state 预测再下一个 token。

------

## 完整数值总表

| 阶段            | A        | B       | Packed 结果    |
| --------------- | -------- | ------- | -------------- |
| Scheduler 调度  | 2 tokens | 1 token | `[2,1]`        |
| 已计算长度      | 3        | 4       | `[3,4]`        |
| req_indices     | 0,0      | 1       | `[0,0,1]`      |
| query_pos       | 0,1      | 0       | `[0,1,0]`      |
| positions       | 3,4      | 4       | `[3,4,4]`      |
| input_ids       | 14,15    | 32      | `[14,15,32]`   |
| query_start_loc | `[0,2]`  | `[2,3]` | `[0,2,3]`      |
| 新 seq_len      | 5        | 5       | `[5,5]`        |
| KV slot         | 31,36    | 60      | `[31,36,60]`   |
| logits index    | 1        | 2       | `[1,2]`        |
| sampled token   | 99       | 33      | `[[99],[33]]`  |
| 是否丢弃        | 是       | 否      | `[True,False]` |
| 有效输出        | 无       | 33      | `[[],[33]]`    |

最终，这张架构图最核心的数据流可以具体化为：

```
SchedulerOutput
    num_scheduled_tokens = {"A":2, "B":1}
                         │
                         ▼
InputBatch
    token table + computed lengths
                         │
                         ▼
_prepare_inputs()
    input_ids       = [14,15,32]
    positions       = [3,4,4]
    query_start_loc = [0,2,3]
    slot_mapping    = [31,36,60]
                         │
                         ▼
Transformer
    hidden_states.shape = [3,H]
                         │
                         ▼
compute_logits()
    logits_indices = [1,2]
    logits.shape = [2,V]
                         │
                         ▼
Sampler
    sampled = [[99],[33]]
                         │
                         ▼
Bookkeeping
    discard = [True,False]
                         │
                         ▼
ModelRunnerOutput
    A=[]
    B=[33]
```









