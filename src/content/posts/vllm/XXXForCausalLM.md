---
title: "XXXForCausalLM"
published: 2026-07-15
description: "XXXForCausalLM"
image: ""
tags: ["vllm","XXXForCausalLM"]
category: vllm
draft: false
lang: ""
createdAt: "2026-07-16T01:23:27.600.383709407Z"
---



# Qwen2ForCausalLM

![image-20260715212800251](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260715212800251)

```python
Qwen2ForCausalLM
├── Qwen2Model
│   ├── embed_tokens
│   │   └── weight: [152064, 3584]
│   │
│   ├── layers: 28 层
│   │
│   │   每一层 Qwen2DecoderLayer:
│   │   ├── input_layernorm
│   │   │   └── RMSNorm(3584)
│   │   │
│   │   ├── self_attn
│   │   │   ├── qkv_proj
│   │   │   │   └── [3584] -> [4608]
│   │   │   │       q: [3584] = 28 heads * 128
│   │   │   │       k: [512]  = 4 kv_heads * 128
│   │   │   │       v: [512]  = 4 kv_heads * 128
│   │   │   │
│   │   │   ├── rotary_emb
│   │   │   │   └── RoPE(head_dim=128, max_position=131072)
│   │   │   │
│   │   │   ├── attn
│   │   │   │   └── GQA attention
│   │   │   │       num_heads=28
│   │   │   │       num_kv_heads=4
│   │   │   │
│   │   │   └── o_proj
│   │   │       └── [3584] -> [3584]
│   │   │
│   │   ├── post_attention_layernorm
│   │   │   └── RMSNorm(3584)
│   │   │
│   │   └── mlp
│   │       ├── gate_up_proj
│   │       │   └── [3584] -> [37888]
│   │       │       gate: [18944]
│   │       │       up:   [18944]
│   │       │
│   │       ├── SiluAndMul
│   │       │   └── silu(gate) * up
│   │       │       [37888] -> [18944]
│   │       │
│   │       └── down_proj
│   │           └── [18944] -> [3584]
│   │
│   └── final norm
│       └── RMSNorm(3584)
│
├── lm_head
│   └── weight: [152064, 3584]
│       因为 tie_word_embeddings=false，所以不是 embed_tokens 复用
│
└── logits_processor
    └── hidden_states -> logits
```



为便于观察张量形状，先采用以下假设：

```python
# 模型：Qwen/Qwen2-7B
# Tensor Parallel，张量并行卡数：TP = 1
# Pipeline Parallel，流水线并行级数：PP = 1
# 本轮送入 vLLM 的 token 总数：T
#
# 注意：
# vLLM 通常不会保持传统的 [batch_size, seq_len, hidden_size] 三维布局，
# 而是把当前批次中的有效 token 压平，因此主要张量形状是：
#
# hidden_states.shape = [T, 3584]
```

Qwen2-7B 的官方配置是：

```python
vocab_size = 152064
hidden_size = 3584
intermediate_size = 18944

num_hidden_layers = 28
num_attention_heads = 28
num_key_value_heads = 4

head_dim = hidden_size // num_attention_heads
         = 3584 // 28
         = 128

hidden_act = "silu"
rms_norm_eps = 1e-6
rope_theta = 1_000_000.0
max_position_embeddings = 131072

tie_word_embeddings = False
torch_dtype = bfloat16
```

这些数值来自 Qwen2-7B 官方 `config.json`。Qwen2-7B 使用 decoder-only Transformer、SwiGLU、带 bias 的 QKV 投影和 GQA。([Hugging Face](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json))

------

# 1）模型整体执行路线

调用入口通常可以简化为：

```python
model = Qwen2ForCausalLM(vllm_config)

# 第一次前向传播
hidden_states = model.forward(
    input_ids=input_ids,
    positions=positions,
)

# 将最后一层隐状态转换成词表 logits
logits = model.compute_logits(hidden_states)
```

真实调用关系如下：

```text
Qwen2ForCausalLM.forward
    ↓
Qwen2Model.forward
    ↓
VocabParallelEmbedding
    ↓
28 × Qwen2DecoderLayer
    ├── RMSNorm
    ├── Qwen2Attention
    │   ├── QKVParallelLinear
    │   ├── RoPE
    │   ├── Attention + KV Cache
    │   └── RowParallelLinear
    ├── RMSNorm
    └── Qwen2MLP
        ├── MergedColumnParallelLinear
        ├── SiLUAndMul，也就是 SwiGLU
        └── RowParallelLinear
    ↓
最终 RMSNorm
    ↓
LM Head
    ↓
LogitsProcessor
    ↓
[T, 152064] logits
```

下面按照这个顺序展开。

------

# 2）第一步：构造 Qwen2ForCausalLM

对应代码：

```python
class Qwen2ForCausalLM(...):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config.get_text_config()

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
```

带入 Qwen2-7B 数值：

```python
config.hidden_size = 3584
config.vocab_size = 152064
config.num_hidden_layers = 28
config.num_attention_heads = 28
config.num_key_value_heads = 4
config.intermediate_size = 18944
```

这里首先构造的不是输出层，而是主体：

```python
self.model = Qwen2Model(...)
```

也就是：

```text
词嵌入
    +
28 层 DecoderLayer
    +
最终 RMSNorm
```

然后构造 LM Head：

```python
if get_pp_group().is_last_rank:
    if config.tie_word_embeddings:
        self.lm_head = self.model.embed_tokens
    else:
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
```

Qwen2-7B 中：

```python
config.tie_word_embeddings = False
```

所以执行的是：

```python
self.lm_head = ParallelLMHead(
    152064,   # vocab_size
    3584,     # hidden_size
)
```

LM Head 的逻辑权重形状为：

```python
lm_head.weight.shape = [152064, 3584]
```

它执行：

```text
[T, 3584]
    ×
[3584, 152064]
    =
[T, 152064]
```

因此每个 token 都会获得一个长度为 152064 的词表分数向量。

------

# 3）第二步：构造 Qwen2Model

核心代码：

```python
class Qwen2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config, prefix="", decoder_layer_type=Qwen2DecoderLayer):
        config = vllm_config.model_config.hf_config.get_text_config()

        self.config = config
        self.vocab_size = config.vocab_size
```

代入：

```python
self.vocab_size = 152064
```

## 3.1）构造词嵌入层

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
)

self.embed_tokens = VocabParallelEmbedding(
    152064,
    3584,
)
```

逻辑上的 embedding 权重：

```python
embed_tokens.weight.shape = [152064, 3584]
```

输入：

```python
input_ids.shape = [T]
```

例如：

```python
input_ids = tensor([
    151644,
    9707,
    198,
    105043,
    1773,
])
```

假设总共有 5 个 token：

```python
T = 5
input_ids.shape = [5]
```

经过词嵌入：

```python
hidden_states = self.embed_tokens(input_ids)
hidden_states.shape = [5, 3584]
```

含义是：

```text
每个 token ID
    ↓ 查 embedding 表
得到一个 3584 维向量
```

## 3.2）构造 28 个 DecoderLayer

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: decoder_layer_type(...),
)


def make_layers(
    num_hidden_layers: int,      # 模型总层数
    layer_fn: LayerFn,          # 用于创建单层实例的工厂函数
    prefix: str,                # 层名称的前缀（用于 PyTorch 参数命名）
) -> tuple[int, int, torch.nn.ModuleList]:
    """根据流水线并行（PP）配置，创建包含占位符和真实层的层列表。"""
    
    # 动态导入必要的分布式工具和离线处理器
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.offloader import get_offloader

    # 1. 计算当前 GPU 在流水线中所负责的层索引范围 [start_layer, end_layer)
    # 例如：80 层，2 个 GPU，rank 0 负责 [0, 40)，rank 1 负责 [40, 80)
    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )

    # 2. 构建 nn.ModuleList，确保总长度始终等于 num_hidden_layers
    modules = torch.nn.ModuleList(
        # A. 前部填充：在当前层之前的位置放入 PPMissingLayer 占位符
        [PPMissingLayer() for _ in range(start_layer)]
        
        # B. 中部加载：利用 offloader 处理该 GPU 负责的层（可能包含自动卸载到 CPU 的逻辑）
        # 通过 layer_fn 为每一层生成唯一前缀（如 'model.layers.0'）
        + get_offloader().wrap_modules(
            layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
        )
        
        # C. 后部填充：在当前层之后的位置放入 PPMissingLayer 占位符
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )

    # 返回当前的索引范围和组装好的模型层列表
    return start_layer, end_layer, modules
```

代入：

```python
config.num_hidden_layers = 28
```

在 `PP=1` 时：

```python
self.start_layer = 0
self.end_layer = 28

self.layers = [
    Qwen2DecoderLayer(layer=0),
    Qwen2DecoderLayer(layer=1),
    ...
    Qwen2DecoderLayer(layer=27),
]
```

因此后续会循环执行 28 次 DecoderLayer。

## 3.3）构造最终归一化

```python
self.norm = RMSNorm(
    config.hidden_size,
    eps=config.rms_norm_eps,
)
```

代入：

```python
self.norm = RMSNorm(
    3584,
    eps=1e-6,
)
```

------

# 4）第三步：构造一个 Qwen2DecoderLayer

每一层都有以下结构：

```python
class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, ...):
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(...)
        self.mlp = Qwen2MLP(...)

        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
```

代入 Qwen2-7B：

```python
self.hidden_size = 3584

self.self_attn = Qwen2Attention(
    hidden_size=3584,
    num_heads=28,
    num_kv_heads=4,
    max_position=131072,
    rms_norm_eps=1e-6,
)

self.mlp = Qwen2MLP(
    hidden_size=3584,
    intermediate_size=18944,
    hidden_act="silu",
)

self.input_layernorm = RMSNorm(
    3584,
    eps=1e-6,
)

self.post_attention_layernorm = RMSNorm(
    3584,
    eps=1e-6,
)
```

一层的计算流程可以写成：

```python
# 输入：
# hidden_states: [T, 3584]
# residual: 第一层开始时为 None

# Attention 子层
normalized_x, residual = RMSNorm(hidden_states, residual)
attention_output = Attention(normalized_x)

# MLP 子层
normalized_attention, residual = RMSNorm(attention_output, residual)
mlp_output = MLP(normalized_attention)

# 返回：
# hidden_states 实际上是尚未与 residual 合并的 MLP 输出
# residual 保存已经累积的主残差流
return mlp_output, residual
```

这里需要特别注意：vLLM 使用了融合残差加法的 RMSNorm，因此 `hidden_states` 和 `residual` 被分开保存。

------

# 5）第四步：构造 Qwen2Attention

原代码的关键数值计算：

```python
self.hidden_size = hidden_size

tp_size = get_tensor_model_parallel_world_size()

self.total_num_heads = num_heads
self.num_heads = self.total_num_heads // tp_size

self.total_num_kv_heads = num_kv_heads
self.num_kv_heads = max(
    1,
    self.total_num_kv_heads // tp_size,
)

self.head_dim = hidden_size // self.total_num_heads
self.q_size = self.num_heads * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim

self.scaling = self.head_dim ** -0.5
```

先代入 `TP=1`。

```python
hidden_size = 3584
tp_size = 1

total_num_heads = 28
num_heads = 28 // 1
          = 28

total_num_kv_heads = 4
num_kv_heads = max(1, 4 // 1)
             = 4

head_dim = 3584 // 28
         = 128

q_size = 28 * 128
       = 3584

kv_size = 4 * 128
        = 512

scaling = 128 ** -0.5
        = 1 / sqrt(128)
        ≈ 0.0883883476
```

最终：

```python
self.num_heads = 28
self.num_kv_heads = 4
self.head_dim = 128

self.q_size = 3584
self.kv_size = 512
self.scaling ≈ 0.088388
```

------

# 6）为什么 Q 是 3584 维，而 K、V 只有 512 维

Qwen2-7B 使用 GQA，即 Grouped Query Attention。

```python
query_heads = 28
key_heads = 4
value_heads = 4
```

每个头都是 128 维：

```python
Q:
28 heads × 128 = 3584

K:
4 heads × 128 = 512

V:
4 heads × 128 = 512
```

每个 KV 头服务的 Query 头数量为：

```python
28 // 4 = 7
```

也就是：

```text
KV 头 0 供 7 个 Query 头使用
KV 头 1 供 7 个 Query 头使用
KV 头 2 供 7 个 Query 头使用
KV 头 3 供 7 个 Query 头使用
```

可以理解成：

```text
28 个 Q 头
分成 4 组
每组 7 个 Q 头
每组共享一套 K、V 头
```

这样可以大幅减少 KV Cache。

普通 MHA 每个 token 的 K、V 元素数量是：

```python
28 * 128 * 2 = 7168
```

Qwen2-7B 的 GQA 每个 token 是：

```python
4 * 128 * 2 = 1024
```

比例：

```python
1024 / 7168 = 1 / 7
```

因此仅从 K、V 元素数量看，GQA 的 KV Cache 是相同 28 头 MHA 的约七分之一。

Qwen2 官方模型说明明确指出该模型使用 GQA、SwiGLU 和带 bias 的 Attention QKV 投影。([Hugging Face](https://huggingface.co/Qwen/Qwen2-7B))

------

# 7）QKVParallelLinear 的实际输出维度

构造代码：

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=True,
)
```

代入：

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=3584,
    head_size=128,
    total_num_heads=28,
    total_num_kv_heads=4,
    bias=True,
)
```

Q、K、V 的输出大小分别为：

```python
q_size = 28 * 128 = 3584
k_size = 4 * 128 = 512
v_size = 4 * 128 = 512
```

拼接后的总输出维度：

```python
qkv_output_size = 3584 + 512 + 512
                = 4608
```

所以逻辑权重形状是：

```python
qkv_proj.weight.shape = [4608, 3584]
qkv_proj.bias.shape = [4608]
```

PyTorch 线性层通常按下面的形式理解：

```python
output = input @ weight.T + bias

[T, 3584] @ [3584, 4608]
    =
[T, 4608]
```

------

# 8）Attention 前向传播的完整过程

原始代码：

```python
def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)

    q, k, v = qkv.split(
        [self.q_size, self.kv_size, self.kv_size],
        dim=-1,
    )

    if self.qk_norm:
        ...

    q, k = self.rotary_emb(positions, q, k)

    attn_output = self.attn(q, k, v)

    output, _ = self.o_proj(attn_output)

    return output
```

带入 Qwen2-7B 数值后，可以写成：

```python
def forward(self, positions, hidden_states):
    # hidden_states.shape = [T, 3584]

    # 1. 一次矩阵乘法同时生成 Q、K、V
    qkv, _ = self.qkv_proj(hidden_states)

    # qkv.shape = [T, 4608]
    #
    # 其中：
    # 前 3584 维是 Q
    # 接下来 512 维是 K
    # 最后 512 维是 V

    q, k, v = qkv.split(
        [3584, 512, 512],
        dim=-1,
    )

    # q.shape = [T, 3584]
    # k.shape = [T, 512]
    # v.shape = [T, 512]

    # 2. Qwen2-7B 原始配置没有 qk_norm 字段，
    # getattr(config, "qk_norm", False) 会得到 False。
    # 因此这一段不会执行。
    if False:
        pass

    # 3. 对 Q、K 应用旋转位置编码
    q, k = self.rotary_emb(positions, q, k)

    # 形状不变：
    # q.shape = [T, 3584]
    # k.shape = [T, 512]

    # 4. 执行因果 Attention，并读写 KV Cache
    attn_output = self.attn(q, k, v)

    # attn_output.shape = [T, 3584]
    #
    # 虽然 K、V 只有 4 个头，
    # 但最终仍然产生 28 个 Query 头的输出：
    # 28 × 128 = 3584

    # 5. 输出投影
    output, _ = self.o_proj(attn_output)

    # output.shape = [T, 3584]

    return output
```

------

# 9）RoPE 在这里做了什么

构造代码：

```python
self.rotary_emb = get_rope(
    self.head_dim,
    max_position=max_position,
    rope_parameters=rope_parameters,
)
```

代入：

```python
self.rotary_emb = get_rope(
    head_size=128,
    max_position=131072,
    rope_parameters={
        # 核心配置中包含 rope_theta = 1_000_000
    },
)
```

RoPE 只作用于：

```python
q
k
```

不作用于：

```python
v
```

从形状角度：

```python
# RoPE 前
q.shape = [T, 3584]
k.shape = [T, 512]

# 内部按照头拆开理解
q_heads.shape = [T, 28, 128]
k_heads.shape = [T, 4, 128]

# RoPE 后
q.shape = [T, 3584]
k.shape = [T, 512]
```

RoPE 不改变张量形状，只根据 `positions` 旋转每个头内部的部分维度。

例如：

```python
positions = tensor([0, 1, 2, 3, 4])
positions.shape = [5]
```

表示 5 个 token 的位置分别为 0 到 4。

在 decode 阶段，新 token 的位置可能是：

```python
positions = tensor([2048])
```

表示当前正在计算第 2048 号位置的 token。

Qwen2-7B 配置中的 `rope_theta` 为 1,000,000，最大位置配置为 131,072。([Hugging Face](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json))

------

# 10）Attention 内部的数学计算

虽然 `self.attn()` 的具体实现位于 vLLM Attention Backend 中，但逻辑上仍然是：

```python
Q.shape = [T, 28, 128]
K.shape = [T, 4, 128]
V.shape = [T, 4, 128]
```

GQA 会让每个 KV 头对应 7 个 Q 头。

逻辑计算：

```python
scores = Q @ K.transpose(-1, -2)
scores = scores * scaling
scores = scores + causal_mask
probs = softmax(scores)
context = probs @ V
```

其中缩放值是：

```python
scaling = 1 / sqrt(128)
        ≈ 0.088388
```

因果掩码保证：

```text
位置 0 只能看位置 0
位置 1 可以看位置 0、1
位置 2 可以看位置 0、1、2
……
```

在 vLLM 中，K、V 会写入 Paged KV Cache，因此 decode 阶段不需要重新计算之前所有 token 的 K、V。

## Prefill 阶段

假设输入 5 个 token：

```python
q.shape = [5, 3584]
k.shape = [5, 512]
v.shape = [5, 512]
```

这 5 个 token 的 K、V 都会被写入缓存。

## Decode 阶段

下一步只输入一个新 token：

```python
q.shape = [1, 3584]
k.shape = [1, 512]
v.shape = [1, 512]
```

新 K、V 被追加到缓存中，而新 Q 可以查询：

```text
之前缓存的所有 K、V
+
当前 token 的 K、V
```

------

# 11）Attention 输出投影 o_proj

构造代码：

```python
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    bias=False,
)
```

代入：

```python
self.total_num_heads * self.head_dim
= 28 * 128
= 3584
```

因此：

```python
self.o_proj = RowParallelLinear(
    3584,
    3584,
    bias=False,
)
```

逻辑权重：

```python
o_proj.weight.shape = [3584, 3584]
```

计算：

```python
attn_output.shape = [T, 3584]

output = attn_output @ o_proj.weight.T

output.shape = [T, 3584]
```

Attention 子层不会改变隐藏维度。

------

# 12）第五步：MLP 的初始化

原代码：

```python
class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        ...
    ):
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        self.act_fn = SiluAndMul()
```

代入：

```python
hidden_size = 3584
intermediate_size = 18944
```

因此：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    3584,
    [18944, 18944],
    bias=False,
)

self.down_proj = RowParallelLinear(
    18944,
    3584,
    bias=False,
)
```

`gate_up_proj` 合并了两个原本独立的线性层：

```python
gate_proj: 3584 -> 18944
up_proj:   3584 -> 18944
```

合并后的总输出维度：

```python
18944 + 18944 = 37888
```

因此逻辑权重形状是：

```python
gate_up_proj.weight.shape = [37888, 3584]
```

对应原始 Hugging Face 权重：

```python
gate_proj.weight.shape = [18944, 3584]
up_proj.weight.shape   = [18944, 3584]
```

vLLM 在加载时将二者堆叠到同一个参数中。

------

# 13）MLP 前向传播

原代码：

```python
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    x = self.act_fn(gate_up)
    x, _ = self.down_proj(x)
    return x
```

带入形状：

```python
def forward(self, x):
    # 输入
    # x.shape = [T, 3584]

    gate_up, _ = self.gate_up_proj(x)

    # gate_up.shape = [T, 37888]
    # 即：
    # [T, 18944] 的 gate 部分
    # +
    # [T, 18944] 的 up 部分

    x = self.act_fn(gate_up)

    # SiluAndMul 内部将 gate_up 一分为二：
    #
    # gate, up = gate_up.chunk(2, dim=-1)
    #
    # gate.shape = [T, 18944]
    # up.shape   = [T, 18944]
    #
    # x = silu(gate) * up
    #
    # x.shape = [T, 18944]

    x, _ = self.down_proj(x)

    # [T, 18944] -> [T, 3584]
    # x.shape = [T, 3584]

    return x
```

数学形式：

```python
gate = x @ W_gate.T
up   = x @ W_up.T

activated = SiLU(gate) * up

output = activated @ W_down.T
```

其中：

```python
W_gate.shape = [18944, 3584]
W_up.shape   = [18944, 3584]
W_down.shape = [3584, 18944]
```

这就是 SwiGLU。

------

# 14）SiluAndMul 的数值例子

假设只观察某个位置上的 3 个简化元素：

```python
gate = [-1.0, 0.0, 2.0]
up   = [ 3.0, 4.0, 5.0]
```

SiLU 定义：

```python
silu(x) = x * sigmoid(x)
```

计算：

```python
silu(-1.0) ≈ -0.2689
silu( 0.0) =  0.0
silu( 2.0) ≈  1.7616
```

逐元素乘上 `up`：

```python
output = silu(gate) * up

output[0] = -0.2689 * 3 ≈ -0.8067
output[1] =  0.0    * 4 =  0
output[2] =  1.7616 * 5 ≈  8.808
```

在真实 Qwen2-7B 中不是 3 个元素，而是：

```python
gate.shape = [T, 18944]
up.shape   = [T, 18944]
```

------

# 15）第六步：模型开始 forward

入口：

```python
class Qwen2ForCausalLM:
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        return hidden_states
```

这里暂时没有计算 logits。

它只调用：

```python
Qwen2Model.forward(...)
```

因此：

```text
Qwen2ForCausalLM.forward
只负责得到最终 hidden_states

Qwen2ForCausalLM.compute_logits
才负责得到词表 logits
```

------

# 16）第七步：Embedding 前向

`Qwen2Model.forward` 开头：

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)

    residual = None
```

在普通文本推理中通常：

```python
inputs_embeds = None
```

所以：

```python
hidden_states = self.embed_input_ids(input_ids)
```

进一步调用：

```python
def embed_input_ids(self, input_ids):
    return self.embed_tokens(input_ids)
```

假设：

```python
T = 5
input_ids.shape = [5]
positions.shape = [5]
```

则：

```python
hidden_states.shape = [5, 3584]
residual = None
```

此时的数据状态：

```python
hidden_states:
    当前 token 的 embedding 表示

residual:
    尚未建立残差流，所以为 None
```

------

# 17）第八步：进入第 0 层 DecoderLayer

循环：

```python
for idx, layer in enumerate(
    islice(self.layers, self.start_layer, self.end_layer)
):
    hidden_states, residual = layer(
        positions,
        hidden_states,
        residual,
    )
```

在 `PP=1` 时：

```python
self.start_layer = 0
self.end_layer = 28
```

执行：

```python
layer 0
layer 1
...
layer 27
```

第 0 层的输入：

```python
hidden_states.shape = [T, 3584]
residual = None
```

------

# 18）第 0 层：第一次 RMSNorm

代码：

```python
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
```

执行后：

```python
residual = 原始 embedding
residual.shape = [T, 3584]

hidden_states = RMSNorm(原始 embedding)
hidden_states.shape = [T, 3584]
```

这里没有立即执行：

```python
hidden_states = hidden_states + residual
```

因为这是第一个子层，原始 embedding 本身直接成为 residual。

RMSNorm 可近似表示为：

```python
rms = sqrt(mean(x ** 2) + 1e-6)
normalized = x / rms
output = normalized * weight
```

其中：

```python
weight.shape = [3584]
eps = 1e-6
```

RMSNorm 不减均值，这一点不同于 LayerNorm。

------

# 19）第 0 层：Self Attention

接着：

```python
hidden_states = self.self_attn(
    positions=positions,
    hidden_states=hidden_states,
)
```

输入：

```python
hidden_states.shape = [T, 3584]
positions.shape = [T]
```

内部依次执行：

```python
# 1. QKV 投影
qkv.shape = [T, 4608]

# 2. 切分
q.shape = [T, 3584]
k.shape = [T, 512]
v.shape = [T, 512]

# 3. RoPE
q.shape = [T, 3584]
k.shape = [T, 512]

# 4. Attention
attn_output.shape = [T, 3584]

# 5. o_proj
output.shape = [T, 3584]
```

返回后：

```python
hidden_states.shape = [T, 3584]
residual.shape = [T, 3584]
```

此时：

```python
hidden_states = Attention(RMSNorm(embedding))
residual = embedding
```

二者仍然分开。

------

# 20）第 0 层：第二次 RMSNorm 与融合残差

代码：

```python
hidden_states, residual = self.post_attention_layernorm(
    hidden_states,
    residual,
)
```

这是 vLLM 的融合形式，可以从逻辑上理解为：

```python
residual = residual + hidden_states
hidden_states = RMSNorm(residual)
```

也就是：

```python
residual = embedding + attention_output
hidden_states = RMSNorm(
    embedding + attention_output
)
```

形状：

```python
residual.shape = [T, 3584]
hidden_states.shape = [T, 3584]
```

注意返回的两个值角色不同：

```python
hidden_states:
    归一化之后，准备送入 MLP 的数据

residual:
    未归一化的主残差流
```

------

# 21）第 0 层：MLP

执行：

```python
hidden_states = self.mlp(hidden_states)
```

数据流：

```python
# 输入
[T, 3584]

# gate_up_proj
[T, 3584] -> [T, 37888]

# 切分
gate: [T, 18944]
up:   [T, 18944]

# silu(gate) * up
[T, 18944]

# down_proj
[T, 18944] -> [T, 3584]
```

第 0 层结束时：

```python
hidden_states = mlp_output
hidden_states.shape = [T, 3584]

residual = embedding + attention_output
residual.shape = [T, 3584]
```

此时 MLP 输出还没有加入 residual。

它将在下一层开头完成相加。

------

# 22）第 1 层到第 27 层的开头

从第 1 层开始，进入：

```python
else:
    hidden_states, residual = self.input_layernorm(
        hidden_states,
        residual,
    )
```

逻辑上相当于：

```python
residual = residual + hidden_states
hidden_states = RMSNorm(residual)
```

把上一层的 MLP 输出加入残差流：

```python
residual
= embedding
+ layer0_attention_output
+ layer0_mlp_output
```

然后再归一化，作为第 1 层 Attention 的输入。

因此 vLLM 的残差流执行方式可以整理为：

```python
# 第 0 层开头
residual = embedding
x = RMSNorm(embedding)

# 第 0 层 Attention 后
attention_output = Attention(x)

# 第 0 层 MLP 前
residual = residual + attention_output
x = RMSNorm(residual)
mlp_output = MLP(x)

# 第 1 层 Attention 前
residual = residual + mlp_output
x = RMSNorm(residual)
```

这和标准的 pre-norm Transformer 数学意义一致，只是 vLLM 为性能做了融合。

------

# 23）单层完整等价伪代码

将 vLLM 的融合写法转换为容易理解的标准写法：

```python
def decoder_layer_standard(x):
    # x.shape = [T, 3584]

    # Attention 子层
    x = x + self_attention(
        rmsnorm_1(x)
    )

    # MLP 子层
    x = x + mlp(
        rmsnorm_2(x)
    )

    return x
```

Qwen2-7B 共执行 28 次：

```python
for layer_idx in range(28):
    x = decoder_layer_standard(x)
```

但为了减少显存读写和 kernel 启动，vLLM 把它改写为：

```python
hidden_states, residual
```

双变量形式。

------

# 24）第九步：完成 28 层后执行最终 RMSNorm

循环结束：

```python
hidden_states, _ = self.norm(
    hidden_states,
    residual,
)
```

此时最后一层的 MLP 输出仍在：

```python
hidden_states
```

之前累积的残差流在：

```python
residual
```

融合 RMSNorm 内部逻辑：

```python
residual = residual + hidden_states
hidden_states = RMSNorm(residual)
```

最终：

```python
hidden_states.shape = [T, 3584]
```

这就是 Qwen2Model 的输出。

------

# 25）第十步：计算 logits

`forward()` 自身只返回 hidden states：

```python
hidden_states = model.forward(...)
```

然后推理框架调用：

```python
logits = model.compute_logits(hidden_states)
```

代码：

```python
def compute_logits(self, hidden_states):
    logits = self.logits_processor(
        self.lm_head,
        hidden_states,
    )
    return logits
```

LM Head 计算：

```python
hidden_states.shape = [T, 3584]
lm_head.weight.shape = [152064, 3584]
```

矩阵乘法：

```python
logits = hidden_states @ lm_head.weight.T
```

形状：

```python
[T, 3584] @ [3584, 152064]
    =
[T, 152064]
```

例如：

```python
T = 5

logits.shape = [5, 152064]
```

其中：

```python
logits[0]
```

是第 0 个 token 位置预测下一个 token 的词表分数；

```python
logits[4]
```

是最后一个输入 token 位置预测下一个 token 的词表分数。

在生成任务里，通常主要使用最后一个位置的 logits：

```python
next_token_logits = logits[-1]
next_token_logits.shape = [152064]
```

然后经过：

```text
temperature
top-k
top-p
repetition penalty
采样或 argmax
```

选出下一个 token。

------

# 26）一次 Prefill 的完整形状示例

假设输入长度为：

```python
T = 5
```

则完整流程如下：

```python
# 输入
input_ids.shape = [5]
positions.shape = [5]

# Embedding
hidden_states.shape = [5, 3584]

# 第 0 层 Attention
qkv.shape = [5, 4608]

q.shape = [5, 3584]
k.shape = [5, 512]
v.shape = [5, 512]

attn_output.shape = [5, 3584]
o_proj_output.shape = [5, 3584]

# 第 0 层 MLP
gate_up.shape = [5, 37888]

gate.shape = [5, 18944]
up.shape = [5, 18944]

silu_and_mul.shape = [5, 18944]
mlp_output.shape = [5, 3584]

# 重复 28 层
final_hidden_states.shape = [5, 3584]

# LM Head
logits.shape = [5, 152064]
```

------

# 27）一次 Decode 的完整形状示例

Prefill 完成后，假设再生成一个 token。

这时当前只计算一个新 token：

```python
T = 1
```

输入：

```python
input_ids.shape = [1]
positions.shape = [1]
```

Embedding：

```python
hidden_states.shape = [1, 3584]
```

每层 QKV：

```python
qkv.shape = [1, 4608]

q.shape = [1, 3584]
k.shape = [1, 512]
v.shape = [1, 512]
```

新的 K、V 被写入每一层的 KV Cache。

Attention 输出：

```python
attn_output.shape = [1, 3584]
```

MLP：

```python
gate_up.shape = [1, 37888]
mlp_output.shape = [1, 3584]
```

最终 logits：

```python
logits.shape = [1, 152064]
```

之后从这一行 logits 中采样下一个 token。

------

# 28）TP=2 时，数值会怎样变化

前面使用的是 `TP=1`。现在简要代入两张 GPU：

```python
tp_size = 2
```

Attention 初始化：

```python
total_num_heads = 28
num_heads = 28 // 2
          = 14

total_num_kv_heads = 4
num_kv_heads = 4 // 2
             = 2

head_dim = 128

q_size = 14 * 128
       = 1792

kv_size = 2 * 128
        = 256
```

单张 GPU 上：

```python
q.shape = [T, 1792]
k.shape = [T, 256]
v.shape = [T, 256]

qkv.shape = [T, 2304]
```

因为：

```python
1792 + 256 + 256 = 2304
```

两张 GPU 合起来仍然是：

```python
Q 总维度 = 1792 * 2 = 3584
K 总维度 = 256 * 2 = 512
V 总维度 = 256 * 2 = 512
```

MLP 在 TP=2 下，每张 GPU 持有大致一半的中间维度：

```python
local_intermediate_size = 18944 // 2
                        = 9472
```

每张 GPU 的 gate/up 输出：

```python
gate.shape = [T, 9472]
up.shape   = [T, 9472]

gate_up.shape = [T, 18944]
```

注意这里的 `[T, 18944]` 是单卡合并 gate 和 up 后的局部输出：

```python
9472 + 9472 = 18944
```

------

# 29）TP 不同取值对 Attention 的影响

Qwen2-7B：

```python
total_q_heads = 28
total_kv_heads = 4
head_dim = 128
```

常见结果：

```text
TP=1:
    每卡 Q heads = 28
    每卡 KV heads = 4
    q_size = 3584
    kv_size = 512

TP=2:
    每卡 Q heads = 14
    每卡 KV heads = 2
    q_size = 1792
    kv_size = 256

TP=4:
    每卡 Q heads = 7
    每卡 KV heads = 1
    q_size = 896
    kv_size = 128
```

代码要求：

```python
assert total_num_heads % tp_size == 0
```

所以 TP 必须能整除 28 个 Q 头。

对于 KV 头：

```python
if total_num_kv_heads >= tp_size:
    assert total_num_kv_heads % tp_size == 0
else:
    assert tp_size % total_num_kv_heads == 0
```

例如 `TP=8`：

```python
total_num_kv_heads = 4
tp_size = 8
```

因为 KV 头数量小于 TP 数：

```python
8 % 4 == 0
```

于是 KV 头会在部分 GPU 之间复制，而不是继续切成小数头。

------

# 30）权重加载时为什么要做映射

代码：

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_stacked={
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),

        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
)
```

Hugging Face 权重通常是分开的：

```python
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight

mlp.gate_proj.weight
mlp.up_proj.weight
```

vLLM 为减少 kernel 数量，将它们合并：

```python
q_proj + k_proj + v_proj
    ↓
qkv_proj

gate_proj + up_proj
    ↓
gate_up_proj
```

代入 Qwen2-7B：

```python
q_proj.weight.shape = [3584, 3584]
k_proj.weight.shape = [512, 3584]
v_proj.weight.shape = [512, 3584]
```

合并：

```python
qkv_proj.weight.shape
= [3584 + 512 + 512, 3584]
= [4608, 3584]
```

MLP：

```python
gate_proj.weight.shape = [18944, 3584]
up_proj.weight.shape   = [18944, 3584]
```

合并：

```python
gate_up_proj.weight.shape
= [18944 + 18944, 3584]
= [37888, 3584]
```

这种合并不改变模型数学结果，主要是减少线性层调用和 kernel 启动。

------

# 31）主要参数矩阵形状汇总

以未切分的逻辑权重为准。

```python
# Embedding
embed_tokens.weight
[152064, 3584]

# 每层 Attention
q_proj.weight
[3584, 3584]

q_proj.bias
[3584]

k_proj.weight
[512, 3584]

k_proj.bias
[512]

v_proj.weight
[512, 3584]

v_proj.bias
[512]

# vLLM 合并后
qkv_proj.weight
[4608, 3584]

qkv_proj.bias
[4608]

o_proj.weight
[3584, 3584]

# 每层 MLP
gate_proj.weight
[18944, 3584]

up_proj.weight
[18944, 3584]

gate_up_proj.weight
[37888, 3584]

down_proj.weight
[3584, 18944]

# 每层 RMSNorm
input_layernorm.weight
[3584]

post_attention_layernorm.weight
[3584]

# 最终 RMSNorm
model.norm.weight
[3584]

# LM Head
lm_head.weight
[152064, 3584]
```

------

# 32）每层主要参数量估算

## Attention 参数

QKV 权重：

```python
4608 * 3584
= 16,515,072
```

QKV bias：

```python
4608
```

输出投影：

```python
3584 * 3584
= 12,845,056
```

Attention 合计约：

```python
16,515,072
+ 4,608
+ 12,845,056
=
29,364,736
```

## MLP 参数

Gate 和 Up 合并权重：

```python
37888 * 3584
= 135,782,? 
```

精确计算：

```python
37,888 * 3,584 = 135,782,? 
```

拆开计算更清楚：

```python
gate_proj:
18,944 * 3,584 = 67,891,? 

up_proj:
18,944 * 3,584 = 67,891,?

down_proj:
3,584 * 18,944 = 67,891,?
```

三个矩阵大小相同，因此 MLP 参数量为：

```python
3 * 18,944 * 3,584
= 203,673,600
```

其中：

```python
gate_up_proj = 2 * 18,944 * 3,584
             = 135,782,400

down_proj = 18,944 * 3,584
          = 67,891,200
```

## 两个 RMSNorm

```python
2 * 3584 = 7168
```

## 单层总量

```python
29,364,736
+ 203,673,600
+ 7,168
=
233,045,504
```

约：

```python
233.05M 参数/层
```

28 层约：

```python
233,045,504 * 28
= 6,525,274,112
```

也就是约 65.25 亿个非 embedding 主干参数。

官方模型卡给出的 Qwen2-7B 非 embedding 参数约为 6.5B，总参数约为 7.6B，与上述估算一致。([Hugging Face](https://huggingface.co/Qwen/Qwen2-7B))

------

# 33）Embedding 和 LM Head 为什么占用很多参数

Embedding：

```python
152064 * 3584
= 545,? 
```

精确值：

```python
152,064 * 3,584
= 545,? 
```

计算得到：

```python
545,? ≈ 545.0M
```

更精确地拆开：

```python
152064 * 3500 = 532,224,000
152064 * 84   = 12,773,376

总计 = 544,997,376
```

所以：

```python
embedding 参数量 = 544,997,376
```

Qwen2-7B：

```python
tie_word_embeddings = False
```

因此 LM Head 还有一套同样大小的独立权重：

```python
lm_head 参数量 = 544,997,376
```

二者合计：

```python
1,089,994,752
```

约 10.9 亿参数。

这也是：

```text
非 embedding 参数约 6.5B
总参数约 7.6B
```

之间差距较大的主要原因。官方配置明确设置了 `tie_word_embeddings: false`。([Hugging Face](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json))

------

# 34）将整个 forward 压缩成带数值的注释版

```python
def qwen2_7b_forward(input_ids, positions):
    """
    假设：
        TP = 1
        PP = 1
        当前有效 token 总数 = T

    input_ids.shape = [T]
    positions.shape = [T]
    """

    # =========================================================
    # 1. Token Embedding
    # =========================================================

    # embedding weight: [152064, 3584]
    hidden_states = embed_tokens(input_ids)

    # hidden_states.shape = [T, 3584]
    residual = None

    # =========================================================
    # 2. 共执行 28 个 Decoder Layer
    # =========================================================

    for layer_idx in range(28):

        # -----------------------------------------------------
        # 2.1 Attention 前的残差处理和 RMSNorm
        # -----------------------------------------------------

        if residual is None:
            # 只会在第 0 层进入这里
            residual = hidden_states
            # residual.shape = [T, 3584]

            hidden_states = input_rmsnorm(hidden_states)
            # hidden_states.shape = [T, 3584]
        else:
            # 第 1～27 层进入这里
            #
            # 等价逻辑：
            # residual = residual + hidden_states
            # hidden_states = RMSNorm(residual)

            hidden_states, residual = input_rmsnorm(
                hidden_states,
                residual,
            )

        # -----------------------------------------------------
        # 2.2 QKV 投影
        # -----------------------------------------------------

        # qkv_proj:
        # 3584 -> 4608
        #
        # 4608 = 3584 + 512 + 512
        qkv = qkv_proj(hidden_states)

        # qkv.shape = [T, 4608]

        q, k, v = qkv.split(
            [3584, 512, 512],
            dim=-1,
        )

        # q.shape = [T, 3584]
        #   逻辑头形状：[T, 28, 128]
        #
        # k.shape = [T, 512]
        #   逻辑头形状：[T, 4, 128]
        #
        # v.shape = [T, 512]
        #   逻辑头形状：[T, 4, 128]

        # -----------------------------------------------------
        # 2.3 RoPE
        # -----------------------------------------------------

        # rope_theta = 1_000_000
        # head_dim = 128
        q, k = rotary_emb(positions, q, k)

        # 形状不变

        # -----------------------------------------------------
        # 2.4 GQA Attention
        # -----------------------------------------------------

        # 28 个 Q 头，4 个 KV 头
        # 每个 KV 头由 7 个 Q 头共享
        #
        # 缩放系数：
        # 1 / sqrt(128) ≈ 0.088388
        attn_output = attention(q, k, v)

        # attn_output.shape = [T, 3584]

        # -----------------------------------------------------
        # 2.5 Attention 输出投影
        # -----------------------------------------------------

        # o_proj: 3584 -> 3584
        hidden_states = o_proj(attn_output)

        # hidden_states.shape = [T, 3584]

        # -----------------------------------------------------
        # 2.6 Attention 残差相加 + MLP 前 RMSNorm
        # -----------------------------------------------------

        # 等价逻辑：
        # residual = residual + hidden_states
        # hidden_states = RMSNorm(residual)
        hidden_states, residual = post_attention_rmsnorm(
            hidden_states,
            residual,
        )

        # hidden_states.shape = [T, 3584]
        # residual.shape = [T, 3584]

        # -----------------------------------------------------
        # 2.7 SwiGLU MLP
        # -----------------------------------------------------

        # gate_up_proj:
        # 3584 -> 37888
        gate_up = gate_up_proj(hidden_states)

        # gate_up.shape = [T, 37888]

        gate, up = gate_up.split(
            [18944, 18944],
            dim=-1,
        )

        # gate.shape = [T, 18944]
        # up.shape   = [T, 18944]

        intermediate = silu(gate) * up

        # intermediate.shape = [T, 18944]

        # down_proj:
        # 18944 -> 3584
        hidden_states = down_proj(intermediate)

        # hidden_states.shape = [T, 3584]
        #
        # 注意：
        # 当前 MLP 输出暂时尚未加进 residual；
        # 它会在下一层 input_rmsnorm 中完成相加。

    # =========================================================
    # 3. 最后一层 MLP 残差相加 + 最终 RMSNorm
    # =========================================================

    hidden_states, _ = final_rmsnorm(
        hidden_states,
        residual,
    )

    # hidden_states.shape = [T, 3584]

    # =========================================================
    # 4. LM Head
    # =========================================================

    # lm_head:
    # 3584 -> 152064
    logits = lm_head(hidden_states)

    # logits.shape = [T, 152064]

    return logits
```

------

# 35）最终执行顺序总结

```text
1）加载 Qwen2-7B 配置
   hidden_size = 3584
   intermediate_size = 18944
   layers = 28
   Q heads = 28
   KV heads = 4
   head_dim = 128
   vocab_size = 152064

2）构造词嵌入
   152064 → 3584

3）构造 28 个 DecoderLayer

4）每层构造 Attention
   Q = 3584
   K = 512
   V = 512
   QKV 总输出 = 4608

5）每层构造 MLP
   gate = 18944
   up = 18944
   合并输出 = 37888
   down = 18944 → 3584

6）输入 token 执行 embedding
   [T] → [T, 3584]

7）执行第 0 层
   RMSNorm
   QKV 投影
   RoPE
   GQA
   o_proj
   残差 + RMSNorm
   SwiGLU MLP

8）重复执行第 1～27 层

9）最终残差相加和 RMSNorm
   [T, 3584]

10）执行 LM Head
    [T, 3584] → [T, 152064]

11）LogitsProcessor 处理输出

12）从最后一个位置的 152064 个分数中采样下一个 token
```

