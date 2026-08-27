---
title: "vllm Qwen3_5MoeForCausalLM.__init__()"
published: 2026-08-26
description: "vllm Qwen3_5MoeForCausalLM.__init__()"
image: ""
tags: ["llm_inference","vllm Qwen3_5MoeForCausalLM.__init__()"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-26T22:27:49.112.266610115Z"
---

这一行不是“读取模型权重”，而是：

>   调用 Qwen3.5 模型类的构造函数，在当前 GPU 上建立完整的 PyTorch Module 树、创建所有 Parameter，并为每个 MoE 层选择量化方法和 Experts 后端。

```
model = model_class(
    vllm_config=vllm_config,
    prefix=prefix,
)
```

对于你的模型：

```
model_class = Qwen3_5MoeForCausalLM
```

## 完整调用关系

```
Qwen3_5MoeForCausalLM.__init__()
└── Qwen3_5ForCausalLMBase.__init__()
    ├── nn.Module.__init__()
    ├── 保存vllm_config/model_config
    ├── 创建Qwen3_5Model
    │   ├── 创建Embedding
    │   ├── 创建所有本PP rank负责的DecoderLayer
    │   │   ├── Attention/Linear Attention
    │   │   ├── RMSNorm
    │   │   └── Qwen3NextSparseMoeBlock
    │   │       ├── Router Gate
    │   │       ├── Shared Expert
    │   │       └── FusedMoEFactory
    │   │           ├── MoERunner
    │   │           ├── RoutedExperts
    │   │           ├── MoeWNA16Method
    │   │           └── Oracle选择Experts
    │   └── 创建Final RMSNorm
    ├── 创建lm_head
    └── 创建LogitsProcessor
└── set_moe_parameters()
```

## 1. 进入 `Qwen3_5MoeForCausalLM`

类定义：

```
class Qwen3_5MoeForCausalLM(
    Qwen3_5ForCausalLMBase,
    QwenNextMixtureOfExperts,
):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )
        self.set_moe_parameters()
```

位于 [qwen3_5.py (line 455)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:455)。

首先调用父类 `Qwen3_5ForCausalLMBase.__init__()`，最后记录 MoE 相关参数。

## 2. 初始化最外层 CausalLM

父类构造函数位于 [qwen3_5.py (line 325)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:325)。

它首先从 `vllm_config` 中取得：

```
config = vllm_config.model_config.hf_text_config
cache_config = vllm_config.cache_config
scheduler_config = vllm_config.scheduler_config
self.quant_config = vllm_config.quant_config
```

对于你的配置，大致是：

```
模型架构：Qwen3_5MoeForCausalLM
量化配置：MoeWNA16Config
权重：GPTQ INT4
group_size：128
TP：4
DP：1
KV cache dtype：FP8
```

然后调用：

```
super().__init__()
```

这会执行 `torch.nn.Module.__init__()`，初始化 PyTorch 内部结构：

```
_modules
_parameters
_buffers
_forward_hooks
_state_dict_hooks
```

没有这一步，后续注册子模块和 Parameter 都无法工作。

## 3. 创建内部 `Qwen3_5Model`

核心代码：

```
self.model = Qwen3_5Model(
    vllm_config=vllm_config,
    prefix=maybe_prefix(prefix, "model"),
)
```

位于 [qwen3_5.py (line 342)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:342)。

如果入口 `prefix=""`，这里得到：

```
prefix = "model"
```

`prefix` 不决定计算，它主要用于：

```
checkpoint权重名称匹配
量化模块匹配
LoRA模块定位
日志和模块标识
```

例如后面的 MoE 层名称可能是：

```
model.layers.0.mlp.experts
model.layers.1.mlp.experts
...
```

## 4. 创建 Embedding

在 `Qwen3_5Model.__init__()` 中：

```
self.embed_tokens = VocabParallelEmbedding(
    self.vocab_size,
    config.hidden_size,
)
```

位于 [qwen3_5.py (line 245)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:245)。

因为使用 TP=4，词表或相关张量会按照并行策略进行分片。

此时创建的是 Parameter 存储，不是从 checkpoint 复制数据。

## 5. 创建所有 DecoderLayer

```
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    get_layer,
    prefix=f"{prefix}.layers",
)
```

位于 [qwen3_5.py (line 257)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:257)。

`make_layers()` 根据 Pipeline Parallel 决定当前 rank 创建哪些层：

```
PP=1
└── 每个rank创建所有DecoderLayer

PP>1
├── PP rank 0创建前一段
├── PP rank 1创建下一段
└── 其他层用PPMissingLayer占位
```

你的配置没有设置 PP，所以默认 PP=1，每个 TP rank 都构造全部 Transformer 层，但每层参数按 TP 规则切分。

## 6. 每层创建 Attention 和 MoE

每个 `Qwen3_5DecoderLayer` 根据 `layer_type` 创建：

```
linear_attention
或者
full_attention
```

然后判断模型类型：

```
if config.model_type == "qwen3_5_moe_text":
    self.mlp = Qwen3NextSparseMoeBlock(...)
```

因此你的每个稀疏 MLP 层都会创建 `Qwen3NextSparseMoeBlock`。

## 7. 创建 Router Gate

MoE block 中：

```
self.gate = ReplicatedLinear(
    config.hidden_size,
    config.num_experts,
    bias=False,
    quant_config=None,
)
```

位于 [qwen3_next.py (line 124)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_next.py:124)。

代入数值：

```
输入宽度 = 3072
输出宽度 = 256个专家
```

forward 时它产生：

```
router_logits.shape = [M, 256]
```

然后选出 Top-8：

```
topk_ids.shape     = [M, 8]
topk_weights.shape = [M, 8]
```

## 8. 创建 Routed Experts

核心调用：

```
self.experts = FusedMoEFactory(
    num_experts=256,
    top_k=8,
    hidden_size=3072,
    intermediate_size=config.moe_intermediate_size,
    quant_config=quant_config,
    ...
)
```

位于 [qwen3_next.py (line 165)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_next.py:165)。

Factory 创建：

```
MoERunner
├── Router
└── RoutedExperts
    ├── W13 Parameters
    ├── W2 Parameters
    ├── scales
    └── quant_method
```

## 9. 在构造过程中选择 `MoeWNA16Method`

`RoutedExperts` 创建时执行：

```
quant_method = quant_config.get_quant_method(
    self,
    prefix,
)
```

因为：

```
quant_config = MoeWNA16Config
layer        = RoutedExperts
```

所以返回：

```
MoeWNA16Method(
    quant_config,
    layer.moe_config,
)
```

这发生在模型结构构造阶段，不需要先读 checkpoint 权重。

## 10. Oracle 在这里执行

`MoeWNA16Method.__init__()` 立即调用：

```
self.wna16_backend, self.experts_cls = (
    select_wna16_moe_backend(...)
)
```

所以每创建一个 MoE 层，都会执行一次 Oracle 检查。

只是日志使用了 `logger.info_once()`，因此你通常只看到一条：

```
Using 'TRITON' WNA16 MoE backend
```

对于你的 H100 模型：

```
FlashInfer TRTLLM → 不支持SM90/group128
Marlin            → MoeWNA16 checkpoint布局被排除
Batched Marlin    → 被排除
Triton            → 支持
```

于是每个 MoE 层记录：

```
self.wna16_backend = WNA16MoEBackend.TRITON
self.experts_cls = TritonWNA16Experts
```

但此时还没有实例化最终 `TritonWNA16Experts`，也还没有组装 `FusedMoEKernel`。

## 11. 创建量化权重 Parameter

Oracle 选择结束后，`RoutedExperts` 调用：

```
self.quant_method.create_weights(...)
```

`MoeWNA16Method.create_weights()` 注册：

```
w13_qweight
w2_qweight
w13_scales
w2_scales
可能的w13_qzeros/w2_qzeros
```

对于 GPTQ INT4，权重使用 packed `uint8`：

```
w13_qweight = torch.nn.Parameter(
    torch.empty(
        num_experts,
        2 * intermediate_size_per_partition,
        hidden_size // 2,
        dtype=torch.uint8,
    ),
    requires_grad=False,
)
```

每个 `uint8` 保存两个 INT4 权重。

这些 Parameter 此时通常是：

```
已经分配存储
内容未初始化或尚未填充checkpoint值
```

它们之后会由 `load_weights()` 用 safetensors 数据填入。

## 12. 创建 lm_head

内部 Transformer 模型创建完成后：

```
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=self.quant_config,
)
```

位于 [qwen3_5.py (line 346)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:346)。

如果 checkpoint 配置了：

```
tie_word_embeddings = True
```

则共享 embedding 和 lm_head 权重。

Pipeline Parallel 非最后一个 rank 则只创建：

```
PPMissingLayer()
```

## 13. 这行执行完成后有什么

执行完：

```
model = model_class(...)
```

内存里已经有：

```
Qwen3_5MoeForCausalLM
├── model
│   ├── embed_tokens
│   ├── layers
│   │   ├── attention
│   │   ├── norm
│   │   ├── router
│   │   └── routed experts parameters
│   └── final norm
├── lm_head
└── logits_processor
```

每个 MoE 层也已经知道：

```
quant_method  = MoeWNA16Method
wna16_backend = TRITON
experts_cls   = TritonWNA16Experts
```

## 此时还没有做什么

这一行本身还没有：

```
读取39个safetensors shard
将真实checkpoint数值写入Parameter
转换/repack最终WNA16权重
实例化最终TritonWNA16Experts
组装FusedMoEKernel
分配KV cache
执行torch.compile warmup
捕获CUDA Graph
执行任何真实forward
```

后续才执行：

```
self.load_weights(model, model_config)
```

填充真实权重，然后：

```
process_weights_after_loading(...)
```

完成：

```
权重布局转换
→ 实例化TritonWNA16Experts
→ 创建Prepare/Finalize
→ 组装FusedMoEKernel
```

因此可以把这一行理解为：

```
model_class(...)
= 按VllmConfig在GPU上搭建“空模型骨架”
  + 创建参数存储
  + 为每层确定量化方法和静态后端
```

而下一步 `load_weights()` 才是往这个骨架中装入真实 checkpoint 数据。







# 为什么39个safetensors shard这么迟才进行加载，不能提前吗

39 个 safetensors shard 不是“无故延迟”，而是在最早能够正确装载的位置加载：

```
先创建目标模型结构和Parameter
→ 再把checkpoint权重写入这些Parameter
→ 最后根据权重组装kernel
```

当前顺序在 [base_loader.py (line 53)](/data/home/xli49/lxy/vllm/vllm/model_executor/model_loader/base_loader.py:53)：

```
model = initialize_model(...)      # 1. 创建目标模型

self.load_weights(model, ...)      # 2. 读取39个shard

process_weights_after_loading(...) # 3. 转换权重、组装kernel
```

## 为什么必须先创建模型

safetensors 文件本质上是：

```
参数名称 → Tensor数据
```

例如：

```
model.layers.0.mlp.experts.0.gate_proj.qweight
model.layers.0.mlp.experts.0.up_proj.qweight
model.layers.0.mlp.experts.0.down_proj.qweight
```

加载器必须先知道这些数据应该写到哪里：

```
checkpoint参数
→ 哪个vLLM module
→ 哪个Parameter
→ 当前TP rank取哪一段
→ 是否和其他权重融合
→ 是否需要转置/repack
```

这些信息只有执行：

```
model = model_class(...)
```

创建完整 Module 树以后才存在。

## 以 W13 为例

checkpoint 中 gate 和 up 通常是分开的：

```
expert.0.gate_proj.qweight
expert.0.up_proj.qweight
```

vLLM 内部却创建融合权重：

```
w13_qweight
= concat(gate_proj, up_proj)
```

而且 TP=4 后，每个 rank 只加载相应 intermediate 分片：

```
checkpoint gate/up权重
          │
          ├── TP rank 0分片
          ├── TP rank 1分片
          ├── TP rank 2分片
          └── TP rank 3分片
                    │
                    └── 写入该rank的w13_qweight
```

这个映射由构造阶段注册的自定义 `weight_loader` 完成：

```
set_weight_attrs(
    w13_qweight,
    {"weight_loader": moe_wna16_weight_loader},
)
```

如果模型还没有构造，加载器甚至不知道：

```
目标Parameter叫什么
目标shape是多少
当前rank加载哪一片
gate/up如何融合
INT4 nibble如何排列
```

## 为什么不能先把39个shard全部读进内存

理论上可以先读成一个巨大的字典：

```
checkpoint = {
    name: tensor,
    ...
}
```

但这样通常更差：

```
CPU内存里保存完整122B模型权重
+
GPU上再创建目标Parameter
+
转换时还可能产生临时副本
```

会导致：

```
CPU内存峰值很高
GPU/CPU双份权重
加载时间变长
更容易OOM
无法提前过滤其他TP/PP rank的数据
```

当前 vLLM 使用 iterator 流式加载：

```
for name, tensor in safetensors_weights_iterator(...):
    model.load_weight(name, tensor)
```

大致效果是：

```
打开shard 1
├── 读取当前rank需要的tensor
├── 写入模型Parameter
└── 释放临时tensor

打开shard 2
...

打开shard 39
```

所以进度条中的 `39 shards` 不代表同时把 39 个文件全部放进内存。

## 哪些东西其实已经提前读取

在创建模型前，vLLM 已经读取了少量元数据：

```
config.json
quantize_config.json
tokenizer配置
模型architecture
hidden_size
num_hidden_layers
num_experts
topk
GPTQ bits/group_size/desc_act
```

这些足够完成：

```
选择Qwen3_5MoeForCausalLM
选择MoeWNA16Config
计算Parameter形状
选择Oracle Experts候选
```

但不需要读取巨大的 safetensors 数据。

## 为什么 `Using TritonWNA16Experts` 在 shard 后才打印

这里要区分“选择”和“实例化”。

### 模型构造时：选择类

```
self.wna16_backend, self.experts_cls = (
    select_wna16_moe_backend(...)
)
```

这时只保存：

```
backend    = TRITON
experts_cls = TritonWNA16Experts
```

### 权重加载后：实例化 Experts

```
process_weights_after_loading()
└── MoeWNA16Method._setup_kernel()
    └── make_wna16_moe_kernel()
        └── TritonWNA16Experts(...)
```

日志：

```
Using MoEPrepareAndFinalizeNoDPEPModular
Using TritonWNA16Experts
```

来自真正实例化 Experts 的阶段，所以出现在：

```
Loading safetensors checkpoint shards: 100%
```

之后是正常的。

原因是 `TritonWNA16Experts` 构造时需要已经加载好的：

```
w13_scales
w2_scales
zero points
block_shape
最终权重布局
```

## 能否提前或并行读取

可以优化“I/O 准备”，但不能颠倒逻辑依赖。

可以提前：

```
下载模型到本地
下载safetensors index
打开/mmap shard
预取下一个shard
多线程读取不同文件
```

当前 loader 已经支持类似：

```
enable_multithread_load
num_threads
safetensors_prefetch_num_threads
safetensors_prefetch_block_size
```

相关代码在 [default_loader.py (line 266)](/data/home/xli49/lxy/vllm/vllm/model_executor/model_loader/default_loader.py:266)。

但是逻辑顺序仍然必须是：

```
至少先创建对应Parameter
→ 才能把预取的数据写进去
```

更激进的 layer-wise loading 可以做到：

```
创建第0层 → 加载第0层
创建第1层 → 加载第1层
...
```

但会让：

```
模型构造
TP/PP映射
权重名称匹配
量化后处理
参数共享
compile元数据
```

复杂很多，当前默认 loader 没采用这种整体交错方式。

所以当前顺序是合理的：

```
先读取轻量配置
→ 创建rank-local模型骨架
→ 流式读取39个shard
→ 写入已注册Parameter
→ WNA16权重转换
→ 实例化Experts和kernel
```

真正能够提前的是下载和 I/O prefetch；真实权重绑定不能早于模型 Parameter 和权重映射的创建。





## 怎么写入的权重的

对你的 `MoeWNA16 + GPTQ INT4 + TP=4`，真实写入过程是：

```
safetensors文件
→ 产生loaded_weight
→ 根据checkpoint名称找到目标Parameter
→ 转成GPU Tensor
→ GPTQ格式转换
→ 取当前TP rank的切片
→ 定位expert和W13/W2区域
→ param.data切片.copy_(loaded_weight)
```

真正完成写入的核心最终就是：

```
expert_data.copy_(loaded_weight)
```

或：

```
param.data[...] = loaded_weight
```

后者底层同样执行 Tensor copy。

## 1. 从 safetensors 产生权重迭代器

Default loader 创建：

```
safetensors_weights_iterator(...)
```

然后不断产生：

```
(weight_name, loaded_weight)
```

例如：

```
weight_name:
model.layers.0.mlp.experts.0.gate_proj.qweight

loaded_weight:
当前checkpoint tensor
```

这个 iterator 是流式的：

```
读取一个权重
→ 写入模型
→ 再读取下一个权重
```

不会先构造一个包含所有权重的大字典。

## 2. Default loader 调模型的 `load_weights`

入口位于 [default_loader.py (line 415)](/data/home/xli49/lxy/vllm/vllm/model_executor/model_loader/default_loader.py:415)：

```
loaded_weights = model.load_weights(
    self.get_all_weights(model_config, model)
)
```

这里：

```
self.get_all_weights(...)
```

返回：

```
Iterable[(weight_name, loaded_weight)]
```

然后调用 Qwen3.5 自己的：

```
Qwen3_5MoeForCausalLM.load_weights()
```

代码在 [qwen3_5.py (line 438)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/qwen3_5.py:438)：

```
def load_weights(self, weights):
    loader = AutoWeightsLoader(self)
    return loader.load_weights(
        weights,
        mapper=self.hf_to_vllm_mapper,
    )
```

## 3. AutoWeightsLoader 根据名称递归寻找目标 Module

`AutoWeightsLoader` 会把名称逐层拆开：

```
model
└── layers
    └── 0
        └── mlp
            └── experts
                └── 0
                    └── gate_proj
                        └── qweight
```

核心逻辑在 [utils.py (line 345)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/utils.py:345)。

它判断名称对应：

```
子Module？
Parameter？
或者当前Module有自定义load_weights()？
```

普通 Parameter 最终调用：

```
weight_loader = getattr(
    param,
    "weight_loader",
    default_weight_loader,
)

weight_loader(param, weight_data)
```

位于 [utils.py (line 308)](/data/home/xli49/lxy/vllm/vllm/model_executor/models/utils.py:308)。

但是 MoE 的 checkpoint 权重名称和 vLLM Parameter 不是一一对应，所以 `RoutedExperts` 使用自己的 `load_weights()`。

## 4. RoutedExperts 将 checkpoint 名称映射到 W13/W2

进入 [routed_experts.py (line 878)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:878)。

它通过 `expert_mapping` 将名称映射：

```
checkpoint gate_proj → w13_qweight，shard_id="w1"
checkpoint up_proj   → w13_qweight，shard_id="w3"
checkpoint down_proj → w2_qweight， shard_id="w2"
```

例如：

```
model.layers.0.mlp.experts.7.gate_proj.qweight
```

被解析为：

```
目标Parameter：w13_qweight
expert_id：7
shard_id："w1"
```

然后取得已经创建好的 Parameter：

```
param = getattr(self, param_name)
```

例如：

```
param = self.w13_qweight
```

最后调用绑定在 Parameter 上的 loader：

```
param.weight_loader(
    param=param,
    loaded_weight=loaded_expert,
    weight_name=weight_name,
    shard_id=shard_id,
    expert_id=expert_id,
)
```

位于 [routed_experts.py (line 944)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:944)。

## 5. WNA16 自定义 loader 把 tensor 放到 GPU

`MoeWNA16Method` 给 W13/W2 Parameter 安装了一个包装后的 loader：

```
moe_wna16_weight_loader
```

位于 [moe_wna16.py (line 566)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:566)。

核心代码：

```
device = get_tp_group().device
tp_rank = get_tensor_model_parallel_rank()

loaded_weight = loaded_weight.to(device)
```

这里是最明确的设备传输。

如果 safetensors tensor 当前在 CPU：

```
CPU loaded_weight
    │
    │ cudaMemcpy HtoD
    ▼
GPU loaded_weight
```

如果加载策略已经让 `loaded_weight` 位于 GPU，那么 `.to(device)` 可能不产生实际复制。

## 6. 转换 GPTQ INT4 格式

你的 checkpoint 是 GPTQ INT4，因此执行：

```
if "weight" in weight_name:
    loaded_weight = (
        loaded_weight.T
        .contiguous()
        .view(torch.uint8)
    )
```

位于 [moe_wna16.py (line 594)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/moe_wna16.py:594)。

逻辑是：

```
GPTQ checkpoint布局
→ transpose
→ contiguous
→ 按uint8重新解释
```

INT4 每两个权重装在一个 `uint8` 中：

```
一个uint8
├── 低4 bit：权重0
└── 高4 bit：权重1
```

这里只是布局转换，还没有写入最终 Parameter。

## 7. 委托给 RoutedExperts 原始 loader

WNA16 做完设备移动和 GPTQ 格式转换后：

```
return weight_loader(
    param,
    loaded_weight,
    weight_name,
    shard_id,
    expert_id,
)
```

这里的 `weight_loader` 是：

```
RoutedExperts.weight_loader
```

它负责：

```
专家编号映射
TP切片
W13 gate/up拼接
写入目标Parameter
```

## 8. W13 如何写入

W13 融合：

```
W13
├── 前半部分：gate_proj，即w1
└── 后半部分：up_proj，即w3
```

进入 `_load_w13()`：

```
loaded_per_rank = (
    loaded_weight.shape[shard_dim] // tp_size
)

start_offset = loaded_per_rank * tp_rank

loaded_weight = loaded_weight.narrow(
    shard_dim,
    start_offset,
    loaded_per_rank,
)
```

位于 [routed_experts.py (line 477)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:477)。

这一步选择当前 TP rank 的切片。

假设完整 intermediate size 是 1024：

```
TP=4
每rank intermediate=256

rank0 → [0:256]
rank1 → [256:512]
rank2 → [512:768]
rank3 → [768:1024]
```

然后定位目标 Parameter 的 W13 区域：

```
if shard_id == "w1":
    # gate写入前半部分
    expert_data = expert_data.narrow(
        shard_dim,
        0,
        shard_size,
    )
else:
    # up写入后半部分
    expert_data = expert_data.narrow(
        shard_dim,
        shard_size,
        shard_size,
    )
```

最后真正写入：

```
expert_data.copy_(loaded_weight)
```

位于 [routed_experts.py (line 526)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:526)。

代入 expert 7：

```
gate_proj
→ w13_qweight[7, 0:256, :]
                  ↑
                  copy_

up_proj
→ w13_qweight[7, 256:512, :]
                    ↑
                    copy_
```

最终每卡：

```
w13_qweight.shape
≈ [256 experts, 512 gate+up, 1536 packed hidden]
```

## 9. W2 如何写入

Down projection 进入 `_load_w2()`：

```
loaded_per_rank = (
    loaded_weight.shape[shard_dim] // tp_size
)

start_offset = loaded_per_rank * tp_rank

loaded_weight = loaded_weight.narrow(
    shard_dim,
    start_offset,
    loaded_per_rank,
)
```

W2 是 Row Parallel，所以沿输入 intermediate 维度切分。

最后：

```
expert_data.copy_(loaded_weight)
```

位于 [routed_experts.py (line 561)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py:561)。

代入 expert 7：

```
checkpoint expert 7 down_proj
→ 选择当前TP rank的intermediate切片
→ w2_qweight[7, :, :]
               ↑
               copy_
```

最终每卡大致：

```
w2_qweight.shape
≈ [256 experts, 3072 hidden, 128 packed intermediate]
```

`128 packed` 对应：

```
256个INT4值
÷ 每个uint8装2个INT4
= 128个uint8
```

## 10. Scale 如何写入

`w13_scales` 和 `w2_scales` 也通过同一个路径处理。

根据 Parameter 上的：

```
quant_method = "group"
```

进入 group scale loader：

```
self._load_model_weight_or_group_weight_scale(...)
```

它最终同样调用：

```
_load_w13(...)
```

或：

```
_load_w2(...)
```

然后通过：

```
expert_data.copy_(loaded_weight)
```

写入对应 GPU scale Parameter。

## 11. `copy_()` 到底做什么

假设：

```
param.device == cuda:2
loaded_weight.device == cuda:2
```

那么：

```
param.data.copy_(loaded_weight)
```

执行的是 GPU 到 GPU 的复制：

```
GPU临时loaded_weight
        │
        │ device-to-device copy/kernel
        ▼
GPU Parameter storage
```

在此之前：

```
loaded_weight.to(device)
```

可能已经完成：

```
CPU safetensors tensor
        │
        │ host-to-device copy
        ▼
GPU临时loaded_weight
```

完整数据流：

```
磁盘shard
  │
  ▼
safetensors mmap/CPU Tensor
  │
  │ loaded_weight.to(cuda)
  ▼
GPU临时Tensor
  │
  ├── transpose
  ├── contiguous
  ├── uint8 view
  └── TP narrow
  │
  │ expert_data.copy_(loaded_weight)
  ▼
GPU中的w13_qweight/w2_qweight Parameter
```

## 普通权重的最简路径

对于不需要 MoE 融合、TP 切片或量化转换的普通 Parameter，最终使用 [weight_utils.py (line 1222)](/data/home/xli49/lxy/vllm/vllm/model_executor/model_loader/weight_utils.py:1222)：

```
def default_weight_loader(param, loaded_weight):
    assert param.size() == loaded_weight.size()
    param.data.copy_(loaded_weight)
```

所以无论普通权重还是 WNA16 MoE 权重，最终落点都是：

```
目标_parameter.data.copy_(处理后的_loaded_weight)
```

WNA16 只是写入前多做了名称映射、专家定位、GPTQ转换、W13融合和TP切片。
