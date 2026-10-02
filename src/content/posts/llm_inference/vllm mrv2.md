---
title: "vllm mrv2"
published: 2026-09-25
description: "vllm mrv2"
image: ""
tags: ["llm_inference","vllm mrv2"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-25T21:46:25.171.953711816Z"
---

# mrv2

这是 MRV2 的核心 forward 函数。它负责把 Scheduler 的“请求计划”转换成 GPU 输入，执行模型，并把 R3 路由结果保存下来；采样在后面的 `sample_tokens()` 完成。

## 整体调用树

```
execute_model(scheduler_output)
├── 1. 更新 Worker 中的请求状态
├── 2. 整理当前 batch
│   ├── 请求排序
│   ├── 统计 token
│   └── DP rank 间同步 token 数
├── 3. 准备模型输入
│   ├── input_ids
│   ├── positions
│   ├── query_start_loc
│   └── seq_lens
├── 4. 准备 Attention
│   ├── block_tables
│   ├── slot_mappings
│   └── attn_metadata
├── 5. 建立 ForwardContext
├── 6. 执行 model.forward()
│   └── Transformer Layer
│       └── MoE Router
│           ├── 选择 Top-K Experts
│           └── capture(topk_ids)
├── 7. 提取 hidden_states
├── 8. 快照 R3 routing_data + slot_mapping
└── 9. 保存 ExecuteModelState
    └── 后续 sample_tokens() 采样并异步 D2H
```

源码入口：[model_runner.py (line 1494)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1494)

------

## 数值例子

假设当前配置：

```
DP size = 2
当前是 DP rank 1
TP size = 2
PP size = 1

MoE 层数 = 2
Expert 数 = 8
每个 token Top-K = 2
KV block_size = 16
```

当前 DP rank 有两个请求：

```
请求 A：
  已经计算 10 个 token
  本轮调度 2 个 token：[101, 102]

请求 B：
  已经计算 21 个 token
  本轮调度 3 个 token：[201, 202, 203]
```

Scheduler 传进来的核心数据：

```
scheduler_output.num_scheduled_tokens = {
    "A": 2,
    "B": 3,
}

scheduler_output.total_num_scheduled_tokens = 5
scheduler_output.scheduled_spec_decode_tokens = {}
scheduler_output.finished_req_ids = set()
```

另一个 DP rank 有 3 个 token，因此：

```
num_tokens_across_dp = [3, 5]
```

下面假设使用 Eager 模式、没有 CUDA Graph padding。

------

## 0. `@torch.inference_mode()`

```
@torch.inference_mode()
def execute_model(...):
```

它表示整个函数不构建 autograd 计算图：

```
不记录梯度
不保存 backward 中间状态
降低显存占用
减少 Python/PyTorch 开销
```

它和 `set_forward_context()` 没有直接关系，只是推理优化。

------

## 1. 更新请求状态

```
if not dummy_run:
    self.update_pp_decode_requests()
    self.finish_requests(scheduler_output)
    self.free_states(scheduler_output)
    self.add_requests(scheduler_output)
    self.update_requests(scheduler_output)
    self.block_tables.apply_staged_writes()
```

对应含义：

### update_pp_decode_requests

└── PP 场景下接收之前 stage 的采样结果

按 `batch size = 2`，即当前有两个请求 A、B 来看。

```
PP0 → PP1 → PP2 → PP3
                    └── 最后一个 rank，负责采样
```

假设 PP3 得到：

```
sampled_tokens = [
    [777],  # 请求 A
    [888],  # 请求 B
]

num_sampled = [1, 1]
num_rejected = [0, 0]
```

PP3 会把整个 batch 的结果广播给 PP0、PP1、PP2：

```
PP3
├── broadcast [777, 888] → PP0
├── broadcast [777, 888] → PP1
└── broadcast [777, 888] → PP2
```

每个非最后 rank 保存一个 `PendingRecv`：

```
PendingRecv(
    sampled_tokens=[[777], [888]],
    idx_mapping=[4, 9],      # A→slot 4，B→slot 9
    num_sampled=[1, 1],
    num_rejected=[0, 0],
)
```

注意：不是为两个请求分别创建两个记录，而是整个 batch 一个记录，内部有两行。

因为 `PP size=4`，队列初始有 4 个占位：

```
[None, None, None, None]
```

时间线：

```
Step 0：收到结果 R0 → [None, None, None, R0]
Step 1：收到结果 R1 → [None, None, R0, R1]
Step 2：收到结果 R2 → [None, R0, R1, R2]
Step 3：收到结果 R3 → [R0, R1, R2, R3]
Step 4：取出 R0，更新 A/B 状态
```

更新后：

```
last_sampled_tokens[4] = 777  # A
last_sampled_tokens[9] = 888  # B
```

所以：

```
PP=4
→ 采样结果由 PP3 产生
→ 广播给 PP0/PP1/PP2
→ 每个 rank 将整个 batch 记录进队列
→ 延迟 4 个 execute_model step 后消费
```

这里的延迟长度由 `PP size=4` 决定，`batch size=2` 只决定广播 Tensor 有两行。









```
update_pp_decode_requests
└── PP 场景下接收之前 stage 的采样结果

finish_requests
└── 移除已经完成或 preempt 的请求状态

free_states
└── 清理不再使用的 encoder cache

add_requests
└── 注册第一次出现的新请求

update_requests
├── 更新已有请求的 num_computed_tokens
└── 添加新分配的 KV block

apply_staged_writes
└── 把 CPU 上暂存的 block table 更新写到 GPU
```

如果本轮没有 token：

```
if scheduler_output.total_num_scheduled_tokens == 0:
    return ...
```

就不会运行模型。

相关代码：[model_runner.py (line 1503)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1503)

------

## 2. 统计并整理 batch

```
num_reqs = len(scheduler_output.num_scheduled_tokens)
num_toks = scheduler_output.total_num_scheduled_tokens
max_query_len = max(scheduler_output.num_scheduled_tokens.values())
```

代入数值：

```
num_reqs = 2
num_toks = 5
max_query_len = 3
```

然后：

```
batch_req_state, uniform_tok_count = self.gather_batch_req_state(
    scheduler_output, dummy_run
)
```

它会排序请求并关联 Worker 内部的 request state。

假设排序后仍然是：

```
batch_req_state.req_ids = ["A", "B"]
batch_req_state.num_scheduled_tokens = [2, 3]
batch_req_state.num_tokens = 5
batch_req_state.idx_mapping_np = [4, 9]
```

`idx_mapping=[4,9]` 表示：

```
batch 中的 A → persistent request-state slot 4
batch 中的 B → persistent request-state slot 9
```

因为 A、B 调度的 token 数不同：

```
uniform_tok_count = None
```

相关代码：[`gather_batch_req_state()` (line 1087)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1087)

------

## 3. DP 同步与 CUDA Graph 选择

```
batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
    ...,
    num_reqs=2,
    num_tokens=5,
    dp_size=2,
    dp_rank=1,
)
```

DP0 有 3 个 token，DP1 有 5 个：

```
num_tokens_across_dp = torch.tensor([3, 5])
```

假设使用 Eager：

```
batch_desc.cg_mode = CUDAGraphMode.NONE
batch_desc.num_tokens = 5
batch_desc.num_reqs = 2
```

这个阶段的作用是：

```
DP0 ── 3 tokens ─┐
                 ├── 同步得到 [3, 5]
DP1 ── 5 tokens ─┘

同时决定：
- Eager
- PIECEWISE CUDA Graph
- FULL CUDA Graph
- 是否需要 padding
```

相关代码：[dp_utils.py (line 98)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/dp_utils.py:98)

------

## 4. 准备 InputBatch

```
input_batch = self.prepare_inputs(
    scheduler_output,
    batch_req_state,
    batch_desc,
)
```

产生的关键数据如下。

### `query_start_loc`

```
query_start_loc = [0, 2, 5]
```

含义：

```
A 使用扁平 batch 的 [0:2]
B 使用扁平 batch 的 [2:5]
flat token index:
             0    1    2    3    4
request:     A    A    B    B    B
```

### `input_ids`

```
input_batch.input_ids = [
    101, 102,       # A
    201, 202, 203,  # B
]
```

### `positions`

A 已计算 10 个，B 已计算 21 个：

```
input_batch.positions = [
    10, 11,         # A
    21, 22, 23,     # B
]
```

### `seq_lens`

执行本轮之后：

```
input_batch.seq_lens = [
    12,  # A: 10 + 2
    24,  # B: 21 + 3
]
```

### token 数

```
input_batch.num_tokens = 5
input_batch.num_tokens_after_padding = 5
```

如果 CUDA Graph 要求补到 8：

```
input_batch.num_tokens = 5
input_batch.num_tokens_after_padding = 8
input_batch.is_padding = [
    False, False, False, False, False,
    True,  True,  True,
]
```

相关代码：[`prepare_inputs()` (line 1140)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1140)

------

## 5. 准备 Attention 和 physical slots

```
block_tables, slot_mappings = self.prepare_attn(input_batch)
```

假设 `block_size=16`。

### 请求 A

A 的 positions 是 10、11，所在逻辑 block 映射到 physical block 2：

```
slot = physical_block_id × block_size + offset

2 × 16 + 10 = 42
2 × 16 + 11 = 43
```

### 请求 B

B 的 positions 是 21、22、23。

它们位于请求的第二个逻辑 block，假设映射到 physical block 5：

```
offset = position % 16

5 × 16 + 5 = 85
5 × 16 + 6 = 86
5 × 16 + 7 = 87
```

最终：

```
slot_mappings = [[42, 43, 85, 86, 87]]
```

第一维是 KV cache group。

对应关系：

```
batch row 0 → A position 10 → physical slot 42
batch row 1 → A position 11 → physical slot 43
batch row 2 → B position 21 → physical slot 85
batch row 3 → B position 22 → physical slot 86
batch row 4 → B position 23 → physical slot 87
```

随后：

```
slot_mappings_by_layer = build_slot_mappings_by_layer(
    slot_mappings,
    self.kv_cache_config,
)
```

把 KV group 的 slot mapping 分发给对应 attention layer：

```
{
    "model.layers.0.self_attn": [42, 43, 85, 86, 87],
    "model.layers.1.self_attn": [42, 43, 85, 86, 87],
}
```

相关代码：

-   [`prepare_attn()` (line 1345)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1345)
-   [`build_slot_mappings_by_layer()` (line 232)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/attn_utils.py:232)

------

## 6. 构造模型参数

```
model_inputs = {
    "input_ids": input_ids,
    "positions": input_batch.positions,
    "inputs_embeds": inputs_embeds,
    "intermediate_tensors": None,
    **self.model_state.prepare_inputs(...),
}
```

当前例子相当于：

```
model_inputs = {
    "input_ids": tensor([101, 102, 201, 202, 203]),
    "positions": tensor([10, 11, 21, 22, 23]),
    "inputs_embeds": None,
    "intermediate_tensors": None,
}
```

如果是非首个 PP rank：

```
model_inputs["input_ids"] = None
model_inputs["intermediate_tensors"] = 上一个PP rank传来的hidden_states
```

因为只有第一个 PP rank 直接消费 token IDs。

------

## 7. 建立 ForwardContext

```
with set_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=5,
    num_tokens_across_dp=tensor([3, 5]),
    slot_mapping=slot_mappings_by_layer,
    is_padding=tensor([False, False, False, False, False]),
):
    model_output = self.model(**model_inputs)
```

进入 `with` 后，当前 context 大致是：

```
ForwardContext(
    attn_metadata=attn_metadata,
    slot_mapping={
        "model.layers.0.self_attn": [42, 43, 85, 86, 87],
        ...
    },
    dp_metadata=DPMetadata(
        num_tokens_across_dp_cpu=tensor([3, 5]),
    ),
    is_padding=tensor([False, False, False, False, False]),
)
```

这个 context 在整个模型调用期间有效：

```
model.forward()
└── layer.forward()
    └── MoE.forward()
        └── router.forward()
            └── capture()
                └── get_forward_context()
```

------

## 8. MoE Router 选择 Experts

假设某一层输入：

```
hidden_states.shape = [5, hidden_size]
router_logits.shape = [5, 8]
```

8 表示 8 个 experts。

Layer 0 选择 Top-2：

```
topk_ids_layer0 = [
    [0, 1],  # A token 101
    [2, 3],  # A token 102
    [1, 5],  # B token 201
    [4, 6],  # B token 202
    [0, 7],  # B token 203
]
```

Router 中执行：

```
if self.capture_fn is not None:
    self.capture_fn(topk_ids)
```

进入：

```
capturer.capture(layer_id=0, topk_ids=topk_ids_layer0)
```

相关代码：[base_router.py (line 296)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/router/base_router.py:296)

------

## 9. R3 capture 如何解释这 5 行

```
ctx = get_forward_context()

num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
token_num_per_dp = int(num_tokens_dp[self.dp_rank])
total = int(num_tokens_dp.sum())
n = topk_ids.shape[0]
```

代入：

```
num_tokens_dp = [3, 5]
self.dp_rank = 1

token_num_per_dp = 5
total = 8
n = 5
```

命中：

```
elif n == token_num_per_dp:
    start_loc = 0
    end_loc = 5
```

说明 Router 输入是当前 DP rank 的本地 5 个 token。

然后写入预分配 GPU buffer：

```
self.device_buffer[:5, layer_id, :] = topk_ids[0:5, :]
```

Layer 0 写入：

```
device_buffer[:5, 0, :] = topk_ids_layer0
```

假设 Layer 1 路由结果：

```
topk_ids_layer1 = [
    [3, 6],
    [1, 2],
    [0, 4],
    [2, 7],
    [3, 5],
]
```

则：

```
device_buffer[:5, 1, :] = topk_ids_layer1
```

最终 buffer 的前 5 行为：

```
# shape = [token, layer, top_k]

[
    [[0, 1], [3, 6]],  # A token 101
    [[2, 3], [1, 2]],  # A token 102
    [[1, 5], [0, 4]],  # B token 201
    [[4, 6], [2, 7]],  # B token 202
    [[0, 7], [3, 5]],  # B token 203
]
```

相关代码：[routed_experts_capturer.py (line 108)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:108)

如果 Router 收到的是两个 DP rank 合并后的 8 行：

```
topk_ids.shape = [8, 2]
```

则：

```
n == total == 8
start_loc = 3
end_loc = 8
```

当前 DP1 只保存：

```
topk_ids[3:8]
```

------

## 10. model forward 返回

单 PP 情况下：

```
model_output.shape = [5, hidden_size]
hidden_states = model_output
```

例如 hidden size 为 4096：

```
hidden_states.shape = [5, 4096]
```

如果当前不是最后一个 PP rank：

```
model_output = IntermediateTensors(...)
```

这些 tensor 会发给下一个 PP stage。

------

## 11. 快照 R3 数据

forward 完成后：

```
routed_experts = capturer.get_routed_experts(
    slot_mappings,
    num_toks,
)
```

代入：

```
num_toks = 5
```

内部执行：

```
RoutedExpertsTensors(
    routing_data=self.device_buffer[:5].clone(),
    slot_mapping=slot_mappings[self.attn_gid, :5].clone(),
)
```

得到：

```
routing_data.shape = [5, 2, 2]

slot_mapping = [42, 43, 85, 86, 87]
```

组合起来：

```
physical slot 42 → [[0,1], [3,6]]
physical slot 43 → [[2,3], [1,2]]
physical slot 85 → [[1,5], [0,4]]
physical slot 86 → [[4,6], [2,7]]
physical slot 87 → [[0,7], [3,5]]
```

这里必须 `clone()`，因为：

```
device_buffer 和 slot_mappings 都是复用 buffer
下一个 step 会覆盖它们
异步 D2H 可能还没完成
```

源码：[`get_routed_experts()` (line 232)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:232)

------

## 12. 为什么最后返回 `None`

最后保存：

```
self.execute_model_state = ExecuteModelState(
    input_batch=input_batch,
    hidden_states=hidden_states,
    routed_experts=routed_experts,
    ...
)
```

然后：

```
if not self.is_last_pp_rank:
    return output_intermediate_tensors

return None
```

最后一个 PP rank 返回 `None` 不代表没有结果，而是 MRV2 把工作拆成了两步：

```
execute_model()
├── 执行 Transformer
├── 得到 hidden_states
└── 捕获 R3 routing

sample_tokens()
├── 从 execute_model_state 取 hidden_states
├── 计算 logits
├── 采样 token
├── 启动异步 GPU→CPU copy
└── 返回 AsyncOutput / ModelRunnerOutput
```

`sample_tokens()` 会取出：

```
routed_experts = self.execute_model_state.routed_experts
```

然后传给：

```
AsyncOutput(
    ...,
    routed_experts=routed_experts,
)
```

在这里启动异步输出拷贝。

源码：[`sample_tokens()` (line 1790)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1790)

------

补充一个 FULL CUDA Graph 细节：

```
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
```

运行时没有显式 `with`，因为它不是重新执行 Python `model.forward()`，而是在 replay 之前捕获好的 GPU graph。`capture()` 的 buffer 写入操作已经在捕获阶段记录进 CUDA Graph；捕获阶段仍然位于 [`set_forward_context()` (line 559)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/cudagraph_utils.py:559) 中。





# Concepts

## NCCL

NCCL stands for **NVIDIA Collective Communications Library**.

It is mainly used for high-speed communication between multiple GPUs.

1）`all_reduce` — combine results from multiple GPUs
2）`all_gather` — gather data from all GPUs
3）`reduce_scatter` — reduce data, then split it across GPUs
4）`broadcast` — send data from one GPU to others
