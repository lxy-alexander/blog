---
title: "vLLM Backend"
published: 2026-08-09
description: "vLLM Backend"
image: ""
tags: ["vllm","vLLM Backend"]
category: vllm
draft: false
lang: ""
createdAt: "2026-08-09T19:20:42.495.975400607Z"
---

可以。下面按 vLLM V1 GPU 路径画，重点是 **backend 在 load_model / forward / metadata / KV cache 中怎么生效**。

**符号**

```
T  = 本 step 实际调度 token 数 num_tokens
R  = 本 step request 数 num_reqs
L  = 层数
H  = num_heads
Hkv = num_kv_heads
D  = head_size
Dv = value head_size，通常等于 D
B  = KV cache block_size
Nblk = 本 batch 引用的 KV blocks 数
dtype = fp16/bf16/fp8...
```

**1. load_model 阶段**

```mermaid
sequenceDiagram
    participant Worker as GPUWorker
    participant Runner as GPUModelRunner
    participant Loader as ModelLoader
    participant Model as Model(nn.Module)
    participant Attn as Attention/MLAAttention Layer
    participant Selector as get_attn_backend()
    participant Platform as current_platform
    participant Backend as Backend Class
    participant Impl as Backend Impl

    Worker->>Runner: load_model()
    Runner->>Loader: model_loader.load_model(vllm_config, model_config)
    Loader->>Model: 构造模型模块

    loop 每一层 Transformer layer
        Model->>Attn: __init__(..., attn_backend=None)
        Attn->>Selector: get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla...)
        Selector->>Platform: get_attn_backend_cls(...)
        Platform->>Backend: validate_configuration(...)
        Backend-->>Platform: valid / invalid reasons
        Platform-->>Selector: backend class path
        Selector-->>Attn: Backend Class

        Attn->>Backend: get_impl_cls()
        Backend-->>Attn: Impl Class
        Attn->>Impl: self.impl = Impl(...)
    end

    Loader-->>Runner: self.model
```

这里 backend 第一次真正生效：

```
# attention.py
self.attn_backend = get_attn_backend(...)
impl_cls = self.attn_backend.get_impl_cls()
self.impl = impl_cls(...)
```

也就是说，`FLASH_ATTN` / `TRITON_ATTN` / `FLASHINFER` 在 **模型 layer 初始化时** 就已经变成了每层的 `self.impl`。

**2. 初始化 metadata builder / KV cache group**

```mermaid
sequenceDiagram
    participant Runner as GPUModelRunner
    participant Model as Loaded Model
    participant AttnLayer as Attention Layers
    participant Group as AttentionGroup
    participant Backend as Backend Class
    participant Builder as MetadataBuilder

    Runner->>Model: 扫描 attention layers / kv_cache_specs

    loop 每个 KV cache group
        Runner->>AttnLayer: layer.get_attn_backend()
        AttnLayer-->>Runner: FlashAttentionBackend / TritonAttentionBackend / ...
        Runner->>Group: 按 backend + KV spec 分组
    end

    loop 每个 AttentionGroup
        Group->>Backend: get_builder_cls()
        Backend-->>Group: FlashAttentionMetadataBuilder / TritonAttentionMetadataBuilder / ...
        Group->>Builder: create_metadata_builders(...)
    end
```

这一步决定了 **每次 forward 前 metadata 怎么构造**。不同 backend 不只 kernel 不同，metadata builder 也不同。

**3. 每次 execute_model / forward 阶段**

```mermaid
sequenceDiagram
    participant Engine as Scheduler/Engine
    participant Runner as GPUModelRunner.execute_model
    participant Prep as _preprocess / input buffers
    participant Meta as _build_attention_metadata
    participant Ctx as set_forward_context
    participant Model as self.model(...)
    participant Attn as Attention.forward
    participant Impl as self.impl
    participant Kernel as FA/Triton/FlashInfer Kernel
    participant KV as KV Cache

    Engine->>Runner: scheduler_output

    Runner->>Prep: 准备 input_ids / positions / slot_mapping
    Prep-->>Runner: input_ids[T], positions[T], slot_mapping[T]

    Runner->>Meta: _build_attention_metadata(...)
    Meta->>Meta: query_start_loc[R+1]
    Meta->>Meta: seq_lens[R]
    Meta->>Meta: block_table[R, max_blocks_per_req]
    Meta->>Meta: slot_mapping[T]
    Meta-->>Runner: attn_metadata[layer_name]

    Runner->>Ctx: set_forward_context(attn_metadata, slot_mapping, ...)
    Runner->>Model: self.model(input_ids, positions, ...)

    loop 每层 attention
        Model->>Attn: forward(query, key, value)
        Attn->>KV: 根据 slot_mapping 写 K/V cache
        Attn->>Impl: self.impl.forward(..., kv_cache, attn_metadata, output)
        Impl->>Kernel: 调用具体 backend kernel
        Kernel->>KV: 读历史 K/V blocks
        Kernel-->>Impl: attention output
        Impl-->>Attn: output[T, H*Dv]
        Attn-->>Model: hidden states
    end

    Model-->>Runner: hidden_states[T, hidden_size]
```

**数据量在哪**
每个 step 里，主要数据量大概是：

```
input_ids:      [T] int
positions:      [T] int
query:          [T, H, D]
key:            [T, Hkv, D]
value:          [T, Hkv, Dv]
output:         [T, H, Dv]
slot_mapping:   [T]
seq_lens:       [R]
query_start_loc:[R + 1]
block_table:    [R, max_blocks_per_req]
KV cache:       [num_blocks, B, Hkv, D/Dv]，按 layer/group 存
```

所以：

```
每层 Q 数据量 ≈ T * H * D * sizeof(dtype)
每层 K 数据量 ≈ T * Hkv * D * sizeof(dtype)
每层 V 数据量 ≈ T * Hkv * Dv * sizeof(dtype)
每层输出量 ≈ T * H * Dv * sizeof(dtype)
KV cache 写入量 ≈ T * Hkv * (D + Dv) * sizeof(kv_cache_dtype)
KV cache 读取量 ≈ attention 实际访问的历史 token 数 * Hkv * (D + Dv)
metadata 通常远小于 Q/K/V/KV cache
```

举个普通 MHA/GQA 的感觉：

```
hidden_size = H * D
query/output 是 [T, hidden_size]
KV cache 写入只写本 step 新 token
KV cache 读取会读历史上下文，所以 decode 时 T 小但读 KV 多
prefill 时 T 大，Q/K/V 和 attention 计算都大
```

**4. backend 到底改变了什么**

```mermaid
sequenceDiagram
    participant Attn as Attention Layer
    participant Backend as Backend Class
    participant Builder as MetadataBuilder
    participant Impl as Impl
    participant Kernel as Kernel

    Attn->>Backend: get_builder_cls()
    Backend-->>Attn: metadata builder 类型

    Attn->>Backend: get_impl_cls()
    Backend-->>Attn: impl 类型

    Builder->>Builder: build(attn_metadata)
    Impl->>Kernel: forward(query, key, value, kv_cache, attn_metadata)
```

所以 backend 主要影响两件事：

```
1. metadata 怎么准备
   FlashAttentionMetadataBuilder / TritonAttentionMetadataBuilder / FlashInferMetadataBuilder ...

2. forward 用哪个实现
   FlashAttentionImpl / TritonAttentionImpl / FlashInferImpl ...
```

**一句话总结**
`load_model()` 时，每个 Attention layer 会选定 backend 并创建 `self.impl`；初始化 KV cache 后，runner 会按 backend 创建 metadata builder；每次 `execute_model()` 时，runner 先根据本 step 的 `T/R/block_table/slot_mapping` 构造 metadata，再进入 `model.forward()`，最后每层 `Attention.forward()` 调用对应 backend 的 `self.impl.forward()`，在那里真正执行 FlashAttention/Triton/FlashInfer kernel。







