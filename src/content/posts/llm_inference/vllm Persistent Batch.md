---
title: "vllm Persistent Batch"
published: 2026-09-01
description: "vllm Persistent Batch"
image: ""
tags: ["llm_inference","vllm Persistent Batch"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-01T19:42:26.849.173505990Z"
---

# Persistent Batch：从“批次行号”改成“请求固定槽位”

## 总结

这次改动的核心是：**把“请求的身份位置”和“本轮计算的位置”解耦。**

### MRV1：批次行号就是请求身份

在 MRV1 中，一个请求的状态直接绑定到 batch row：

```
row 0 -> A
row 1 -> B
row 2 -> C
```

当 B 结束后，为保持 batch 连续，需要把 C 从 row 2 搬到 row 1：

```
row 0 -> A
row 1 -> C
```

问题在于，请求状态分散在许多结构中，因此 C 搬行时，所有相关数据都必须同步搬迁：

-   token IDs
-   block table
-   sampling parameters
-   computed token count
-   LoRA 信息
-   speculative decoding 状态

这导致 MRV1 需要 `condense()`、`req_id_to_index`、状态缓存与恢复，以及大量请求重排逻辑。其根本原因是：

>   batch row 同时承担了“请求是谁”和“这轮在哪里执行”两个职责。

### MRV2：固定状态槽位 + 临时执行映射

MRV2 为每个存活请求分配一个稳定的 `req_state_idx`：

```
slot 0 -> A
slot 1 -> free
slot 2 -> C
```

本轮只运行 A 和 C 时，再构造临时执行批次：

```
batch row 0 -> slot 0
batch row 1 -> slot 2

idx_mapping = [0, 2]
```

执行阶段通过 `idx_mapping` 从持久状态中 gather 数据。B 结束后只需释放 slot 1，C 仍留在 slot 2，不需要搬动任何请求状态。

生命周期规则是：

-   新请求：分配空闲 slot。
-   请求存活：slot 保持不变。
-   请求结束：释放 slot。
-   preemption：视为删除并释放 slot。
-   resume：视为新请求，重新分配 slot。

一句话概括：

>   MRV1 通过移动请求状态来适配连续 batch；MRV2 保持请求状态不动，通过映射构造每轮连续的执行 batch。

## 类比：固定储物柜与摆渡车座位

可以把每个请求想象成一名旅客。

### MRV1：旅客必须按照座位连续存放全部行李

最初：

```
座位 0：A 和 A 的全部行李
座位 1：B 和 B 的全部行李
座位 2：C 和 C 的全部行李
```

B 下车后，座位 1 空了。为了让乘客连续就座，工作人员必须让 C 从座位 2 搬到座位 1，而且 C 的所有物品都必须一起搬：

-   行李箱
-   车票
-   护照
-   餐食偏好
-   行程记录

如果漏搬任何一样，系统就可能把 C 的数据和别人的状态混在一起。

这就是 MRV1 的 `condense()`：**不仅调整执行顺序，还要迁移完整的请求身份和状态。**

### MRV2：行李放在固定储物柜，座位只是本次乘车位置

MRV2 给旅客分配固定储物柜：

```
柜子 0：A 的全部状态
柜子 1：空闲
柜子 2：C 的全部状态
```

本次摆渡车只接 A 和 C：

```
车座 0 -> 去柜子 0 取 A 的数据
车座 1 -> 去柜子 2 取 C 的数据
```

这里的乘车清单就是：

```
idx_mapping = [0, 2]
```

B 离开后，只需把柜子 1 标记为空闲。C 的行李仍在柜子 2，完全不用搬。下一班车乘客顺序变化时，也只需要重新制作一张乘车清单。

对应关系如下：

| 概念             | 类比                     |
| ---------------- | ------------------------ |
| `req_state_idx`  | 固定储物柜编号           |
| Persistent state | 储物柜中的全部行李       |
| batch row        | 本次摆渡车座位           |
| `idx_mapping`    | 座位到储物柜的取件清单   |
| 请求结束         | 清空并释放储物柜         |
| preemption       | 旅客离场，柜子释放       |
| resume           | 旅客重新入场，领取新柜子 |

因此，MRV2 并没有消除“整理 batch”的需要，而是把昂贵且危险的**状态搬迁**，变成了便宜且明确的**索引映射重建**。







## 流程

对，而且从代码执行路径看，MRV2 实际上形成了两层映射：

```
执行 batch row
    │ idx_mapping
    ▼
请求固定状态槽位 req_state_idx
    │ block table
    ▼
物理 KV cache block
```

下面从一次真实的 `execute_model()` 调用开始，用具体数字走完整个流程。

假设：

-   `max_num_reqs = 4`
-   `decode_query_len = 1`
-   KV block size = 16
-   暂时不考虑 speculative decoding、PP、DP、CUDA Graph padding
-   当前请求为 A、B、C

### 0. Persistent state 的存储布局

Persistent state 定义在 [states.py (line 9)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/states.py:9)。

初始化时：

```
self.free_indices = list(range(max_num_reqs))
```

所以：

```
free_indices = [0, 1, 2, 3]
```

分配时使用：

```
req_idx = self.free_indices.pop()
```

见 [RequestState.add_request() (line 91)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/states.py:91)。

因此当前实现实际会从高编号开始分配：

```
A -> slot 3
B -> slot 2
C -> slot 1
```

这和用 `slot 0/1/2` 举例没有本质区别，只是当前 allocator 使用栈式 `pop()`。

Persistent state 可以想象成下面几组按 slot 索引的数组：

```
req_id_to_index = {
    "A": 3,
    "B": 2,
    "C": 1,
}

all_token_ids[slot]
num_computed_tokens[slot]
prefill_len[slot]
last_sampled_tokens[slot]
draft_tokens[slot]
block_tables[slot]
sampling_state[slot]
LoRA_state[slot]
```

我们设定状态如下：

| slot | 请求 | token 状态            | `prefill_len` | `num_computed` | KV blocks |
| ---- | ---- | --------------------- | ------------- | -------------- | --------- |
| 0    | free | —                     | —             | —              | —         |
| 1    | C    | `[31,32,33,34,35,36]` | 5             | 6              | `[300]`   |
| 2    | B    | `[21,22,23]`          | 3             | 3              | `[200]`   |
| 3    | A    | `[11,12,13,14]`       | 4             | 2              | `[100]`   |

这里：

-   A 正在 chunked prefill：prompt 有 4 个 token，只计算了前 2 个。
-   C 已完成 prompt，并生成过 token `36`：下一轮是 decode。
-   B 即将结束。

------

## 1. 入口：`execute_model()`

入口是 [model_runner.py (line 1499)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1499)：

```
def execute_model(self, scheduler_output, ...):
```

非 dummy run 时，首先更新 persistent state：

```
self.update_pp_decode_requests()
self.finish_requests(scheduler_output)
self.free_states(scheduler_output)
self.add_requests(scheduler_output)
self.update_requests(scheduler_output)
self.block_tables.apply_staged_writes()
```

调用顺序很重要：

```
处理上轮 PP 结果
    ↓
删除 finished/preempted 请求
    ↓
释放额外状态
    ↓
添加新请求
    ↓
更新已有请求
    ↓
提交 block table 写入
```

------

## 2. B 结束：只释放 slot，不移动 C

假设：

```
scheduler_output.finished_req_ids = {"B"}  # 这是怎么更新的，
```

`finish_requests()` 位于 [model_runner.py (line 971)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:971)。

它最终调用：

```
self._remove_request("B")
```

`_remove_request()` 先通知其他 persistent state：

```
self.model_state.remove_request(req_id)
```

然后释放 `RequestState` 的槽位：

```
req_idx = self.req_states.remove_request(req_id)
```

`RequestState.remove_request()` 位于 [states.py (line 126)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/states.py:126)：

```
req_idx = self.req_id_to_index.pop(req_id)
self.index_to_req_id.pop(req_idx)
self.free_indices.append(req_idx)
```

于是：

```
删除前：
slot 1 -> C
slot 2 -> B
slot 3 -> A

删除后：
slot 1 -> C
slot 2 -> free
slot 3 -> A
```

映射变成：

```
req_id_to_index = {
    "A": 3,
    "C": 1,
}

free_indices = [0, 2]
```

关键是：

```
C 仍然在 slot 1
A 仍然在 slot 3
```

没有发生：

```
C: slot 1 -> slot 0
A: slot 3 -> slot 1
```

因此 C 的 token、block table、sampling state、LoRA、spec state 全都不需要移动。

如果 B 是 preempted，行为也一样。`finish_requests()` 会把：

```
preempted_req_ids
```

合并进待删除集合。以后 C 或 B resume 时，会作为新请求重新执行 `add_request()`，获得当时的空闲 slot。

------

# 3. 新请求分配固定槽位

假设同一轮又有新请求 D。

`add_requests()` 位于 [model_runner.py (line 998)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:998)。

它调用：

```
self.req_states.add_request(...)
```

当前：

```
free_indices = [0, 2]
```

执行 `pop()` 后：

```
D -> slot 2
free_indices = [0]
```

所以 B 原来的 slot 被 D 复用：

```
slot 1 -> C
slot 2 -> D
slot 3 -> A
```

注意：

>   slot 稳定指的是“请求存活期间稳定”，不是某个 `req_id` 永远绑定同一个 slot。

添加请求时，相同的 `req_index` 会传给：

```
model_state.add_request(req_index, ...)
block_tables.append_block_ids(req_index, ...)
lora_state.add_request(req_id, req_index, ...)
sampler.add_request(req_index, ...)
```

因此各种 persistent state 都使用同一个槽位坐标系。

------

# 4. Scheduler 决定本轮执行 A 和 C

假设这一轮：

```
scheduler_output.num_scheduled_tokens = {
    "A": 2,
    "C": 1,
}
```

含义是：

-   A 本轮运行 2 个 prefill token。
-   C 本轮运行 1 个 decode token。

注意，scheduler 给的是：

```
req_id -> 本轮 token 数
```

它不需要知道请求位于哪个 persistent slot。

------

# 5. 构造当前 batch 顺序

`execute_model()` 接下来调用：

```
batch_req_state, uniform_tok_count =
    self.gather_batch_req_state(...)
```

对应 [gather_batch_req_state() (line 1092)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1092)。

首先调用 `sort_batch_req_ids()` 排序：

```
req_ids = sort_batch_req_ids(
    num_tokens_per_req,
    draft_tokens,
    self.decode_query_len,
)
```

排序规则位于 [model_runner.py (line 2100)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:2100)，大体顺序是：

```
spec verification
    ↓
decode
    ↓
short extend
    ↓
prefill
```

因为：

```
C: scheduled_tokens = 1 == decode_query_len
A: scheduled_tokens = 2 != decode_query_len
```

所以本轮 batch 顺序是：

```
req_ids = ["C", "A"]
```

不是 persistent slot 顺序，也不一定是 scheduler 字典的插入顺序。

------

# 6. 建立 `idx_mapping`

核心代码是：

```
idx_mapping_iter = map(
    self.req_states.req_id_to_index.__getitem__,
    req_ids,
)
idx_mapping_np = np.fromiter(...)
```

见 [model_runner.py (line 1118)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1118)。

现在：

```
req_ids = ["C", "A"]

req_id_to_index = {
    "C": 1,
    "A": 3,
}
```

因此：

```
idx_mapping_np = [1, 3]
```

含义是：

```
batch row 0 -> persistent slot 1 -> C
batch row 1 -> persistent slot 3 -> A
```

这是整个设计的核心。

然后按映射 gather CPU 状态：

```
prefill_len_np =
    self.req_states.prefill_len.np[idx_mapping_np]

num_computed_prefill_tokens_np =
    self.req_states.num_computed_prefill_tokens[idx_mapping_np]
```

数值结果：

```
prefill_len_np = [5, 4]
num_computed_prefill_tokens_np = [5, 2]
is_prefilling_np = [False, True]
has_prefill = True
```

对应：

| batch row | 请求 | state slot | prefill_len | computed prefill | 是否 prefill |
| --------- | ---- | ---------- | ----------- | ---------------- | ------------ |
| 0         | C    | 1          | 5           | 5                | False        |
| 1         | A    | 3          | 4           | 2                | True         |

------

# 7. 构造临时 `InputBatch`

然后进入 [prepare_inputs() (line 1145)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1145)。

首先把映射复制到 GPU：

```
idx_mapping = async_copy_to_gpu(
    np.array([1, 3]),
    device=self.device,
)
```

临时 `InputBatch` 保存两种坐标：

```
req_ids       = ["C", "A"]   # batch row -> req_id
idx_mapping   = [1, 3]       # batch row -> req_state_idx
```

字段定义见 [input_batch.py (line 41)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/input_batch.py:41)。

本轮每个请求的 token 数是：

```
num_scheduled_tokens = [1, 2]
```

累计得到：

```
query_start_loc = [0, 1, 3]
```

解释：

```
C 使用 flattened input 的 [0, 1)
A 使用 flattened input 的 [1, 3)
```

也就是：

```
input token row 0       -> C
input token rows 1, 2  -> A
```

这又是一层临时布局，只在当前 forward 有意义。

------

# 8. 通过映射准备 input IDs

因为 batch 中包含 A 的 prefill，代码调用：

```
prepare_prefill_inputs(
    input_ids,
    next_prefill_tokens,
    idx_mapping,
    query_start_loc,
    all_token_ids,
    prefill_len,
    num_computed_tokens,
)
```

见 [input_batch.py (line 303)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/input_batch.py:303)。

Triton kernel 对每个 `batch_idx` 执行：

```
req_state_idx = idx_mapping[batch_idx]
```

### batch row 0：C

```
batch_idx = 0
req_state_idx = idx_mapping[0] = 1
num_computed = 6
prefill_len = 5
```

因为：

```
num_computed >= prefill_len
```

C 已经不是 prefill，kernel 直接返回。

### batch row 1：A

```
batch_idx = 1
req_state_idx = idx_mapping[1] = 3
query_start = 1
query_end = 3
query_len = 2
num_computed = 2
```

读取：

```
all_token_ids[3][2:4]
```

也就是：

```
[13, 14]
```

写入临时模型输入：

```
input_ids[1:3] = [13, 14]
```

然后 `combine_sampled_and_draft_tokens()` 处理 decode 请求，见 [input_batch.py (line 449)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/input_batch.py:449)。

C 的：

```
last_sampled_tokens[slot 1] = 36
```

被写入：

```
input_ids[0] = 36
```

最终模型输入为：

```
input_ids = [36, 13, 14]
```

注意这里已经完全是连续的模型输入，但 persistent state 没有移动：

```
input_ids[0] 来自 slot 1
input_ids[1:3] 来自 slot 3
```

------

# 9. 准备 position 和 sequence length

接下来调用：

```
prepare_pos_seq_lens(
    idx_mapping,
    query_start_loc,
    num_computed_tokens,
    positions,
    seq_lens,
)
```

见 [input_batch.py (line 367)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/input_batch.py:367)。

kernel 的核心逻辑是：

```
req_state_idx = idx_mapping[batch_idx]
num_computed = persistent_num_computed[req_state_idx]

seq_len = num_computed + query_len
positions = num_computed + local_offset
```

### C

```
state slot = 1
num_computed = 6
query_len = 1
```

得到：

```
position = [6]
seq_len = 7
```

### A

```
state slot = 3
num_computed = 2
query_len = 2
```

得到：

```
positions = [2, 3]
seq_len = 4
```

所以最终：

```
positions = [6, 2, 3]
seq_lens = [7, 4]
```

这些数组按 batch row 排列：

| batch row | 请求 | `input_ids` | `positions` | `seq_len` |
| --------- | ---- | ----------- | ----------- | --------- |
| 0         | C    | `[36]`      | `[6]`       | 7         |
| 1         | A    | `[13,14]`   | `[2,3]`     | 4         |

------

# 10. Gather block table：第一层映射连接到 PagedAttention

随后 `execute_model()` 调用：

```
block_tables, slot_mappings = self.prepare_attn(input_batch)
```

见 [prepare_attn() (line 1350)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1350)。

Persistent block table 当前是：

```
state slot 1 / C -> [300]
state slot 3 / A -> [100]
```

`gather_block_tables()` 使用：

```
idx_mapping = [1, 3]
```

见 [block_table.py (line 141)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/block_table.py:141)。

于是 forward 使用的连续 block table 变成：

```
batch row 0 -> persistent row 1 -> [300]
batch row 1 -> persistent row 3 -> [100]
```

也就是：

```
input_block_tables = [
    [300],  # C
    [100],  # A
]
```

注意这里复制的是本轮 attention kernel 所需的小型 block-table 视图，不是搬迁 C 和 A 的全部 persistent state。

------

# 11. 计算 KV slot mapping

接着 [compute_slot_mappings() (line 183)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/block_table.py:183) 再次使用：

```
idx_mapping = [1, 3]
positions = [6, 2, 3]
```

假设：

```
KV block size = kernel block size = 16
```

PagedAttention 中物理 KV slot 的简化计算为：

```
physical_slot =
    physical_block_number * block_size
    + offset_in_block
```

### C

```
logical position = 6
block table = [300]

logical block index = 6 // 16 = 0
offset = 6 % 16 = 6
physical block = block_table[0] = 300

physical slot = 300 * 16 + 6 = 4806
```

### A

位置 2：

```
physical slot = 100 * 16 + 2 = 1602
```

位置 3：

```
physical slot = 100 * 16 + 3 = 1603
```

最终：

```
slot_mapping = [4806, 1602, 1603]
```

所以两层映射完整展开是：

```
batch token 0
  -> batch row 0
  -> idx_mapping[0] = state slot 1
  -> C 的 block table [300]
  -> logical position 6
  -> physical KV slot 4806

batch token 1
  -> batch row 1
  -> idx_mapping[1] = state slot 3
  -> A 的 block table [100]
  -> logical position 2
  -> physical KV slot 1602
```

这就是 MRV2 和 PagedAttention 真正结合的位置。

------

# 12. 执行模型

准备完成后，模型看到的是连续输入：

```
input_ids = [36, 13, 14]
positions = [6, 2, 3]
seq_lens = [7, 4]
```

以及连续的 batch block table：

```
[
    [300],  # batch row 0 / C
    [100],  # batch row 1 / A
]
```

模型根本不需要知道：

```
C 的持久状态在 slot 1
A 的持久状态在 slot 3
slot 2 是空洞或已经被 D 复用
```

从模型 forward 的角度，batch 仍然是紧凑的。

------

# 13. Sampling 仍按 batch row 输出

无 speculative decoding 时，每个请求取一个 logits：

```
cu_num_logits = [0, 1, 2]
```

对应的 flattened logits 位置是：

```
logits_indices = [0, 2]
```

因为：

-   C 的最后一个 input token 在位置 0。
-   A 的最后一个 input token 在位置 2。

假设 sampler 输出：

```
batch row 0 / C -> token 37
batch row 1 / A -> token 15
```

即：

```
sampled_token_ids = [
    [37],
    [15],
]
```

这些输出依然按当前 batch row 排列，不是按 persistent slot 排列。

------

# 14. 通过同一个映射写回 persistent state

sampling 在独立的 [sample_tokens() (line 1800)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1800) 中完成。

随后调用：

```
self.postprocess_sampled(
    input_batch.idx_mapping,
    sampler_output.sampled_token_ids,
    num_sampled,
    num_rejected,
    input_batch.query_start_loc,
)
```

见 [model_runner.py (line 1903)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/model_runner.py:1903)。

最终进入 [post_update() (line 604)](/data/home/xli49/lxy/vllm/vllm/v1/worker/gpu/input_batch.py:604)。

写回 kernel 再次执行：

```
req_state_idx = idx_mapping[batch_row]
```

### batch row 0：写回 C 的 slot 1

```
idx_mapping[0] = 1
sampled token = 37
query_len = 1
```

更新：

```
last_sampled_tokens[1] = 37
all_token_ids[1][6] = 37
total_len[1] = 6 + 1 = 7
num_computed_tokens[1] = 6 + 1 = 7
```

### batch row 1：写回 A 的 slot 3

```
idx_mapping[1] = 3
sampled token = 15
query_len = 2
```

更新：

```
last_sampled_tokens[3] = 15
all_token_ids[3][4] = 15
total_len[3] = 4 + 1 = 5
num_computed_tokens[3] = 2 + 2 = 4
```

最终 persistent state：

| slot | 请求   | tokens                   | computed | last sampled |
| ---- | ------ | ------------------------ | -------- | ------------ |
| 1    | C      | `[31,32,33,34,35,36,37]` | 7        | 37           |
| 2    | D/free | 与本轮无关               | —        | —            |
| 3    | A      | `[11,12,13,14,15]`       | 4        | 15           |

仍然没有任何请求状态搬迁。

------

# 完整数据流

```
SchedulerOutput
  num_scheduled_tokens = {A: 2, C: 1}
                    │
                    ▼
execute_model()
                    │
       finish/add/update requests
                    │
                    ▼
Persistent RequestState
  slot 1 = C
  slot 2 = hole/D
  slot 3 = A
                    │
                    ▼
sort_batch_req_ids()
  req_ids = [C, A]
                    │
                    ▼
idx_mapping = [1, 3]
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
      token IDs   counters   block tables
          │         │          │
          └─────────┼──────────┘
                    ▼
Temporary InputBatch
  input_ids       = [36, 13, 14]
  positions       = [6, 2, 3]
  query_start_loc = [0, 1, 3]
  seq_lens        = [7, 4]
                    │
                    ▼
PagedAttention mapping
  block tables = [[300], [100]]
  KV slots     = [4806, 1602, 1603]
                    │
                    ▼
Model forward + sampling
  batch outputs = [37, 15]
                    │
                    ▼
idx_mapping = [1, 3]
                    │
                    ▼
写回 persistent state
  37 -> slot 1 / C
  15 -> slot 3 / A
```

最核心的一句话是：

>   `idx_mapping` 不仅负责从 persistent state 中读出本轮输入，也负责把本轮结果写回原来的固定槽位；它是一次执行期间双向使用的“请求页表”。

而它与 PagedAttention 的关系是：

>   `idx_mapping` 解决“这个 batch row 属于哪个请求状态槽位”，block table 继续解决“这个请求的逻辑 token 位于哪个物理 KV block”。两者是连续叠加的两层间接寻址。
