---
title: "vllm quatization config"
published: 2026-08-26
description: "vllm quatization config"
image: ""
tags: ["llm_inference","vllm quatization config"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-26T23:17:51.571.896903363Z"
---

这三个对象不是 Oracle 临时创建出来轮流尝试的。它们来自三种不同的 checkpoint 量化配置解析路径：

```
GPTQ checkpoint
└── AutoGPTQConfig.from_config()

AWQ checkpoint
└── AutoAWQConfig.from_config()

Compressed-Tensors checkpoint
└── QuantizationArgs.model_validate()
```

一次模型加载通常只会走其中一种主路径。

## 1. `AutoGPTQConfig` 怎么构造

假设 checkpoint 的 `quantize_config.json`：

```
{
  "quant_method": "gptq",
  "bits": 4,
  "group_size": 128,
  "desc_act": false,
  "sym": true,
  "lm_head": false
}
```

vLLM 识别：

```
quant_method == "gptq"
```

然后选择：

```
AutoGPTQConfig
```

兼容性判断在 [auto_gptq.py (line 219)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:219)：

```
if quant_method != "gptq":
    return None

if user_quant is None or user_quant in (
    "gptq",
    "gptq_marlin",
    "auto_gptq",
    "marlin",
):
    return "auto_gptq"
```

随后调用：

```
AutoGPTQConfig.from_config(config)
```

代码在 [auto_gptq.py (line 195)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:195)：

```
weight_bits = config["bits"]
group_size = config["group_size"]
desc_act = config["desc_act"]
is_sym = config["sym"]
lm_head_quantized = config.get("lm_head", False)
dynamic = config.get("dynamic", {})

return AutoGPTQConfig(
    weight_bits,
    group_size,
    desc_act,
    is_sym,
    lm_head_quantized,
    dynamic,
    config,
)
```

代入数值：

```
AutoGPTQConfig(
    weight_bits=4,
    group_size=128,
    desc_act=False,
    is_sym=True,
    lm_head_quantized=False,
    dynamic={},
    full_config=config,
)
```

构造函数再计算：

```
self.pack_factor = 32 // weight_bits
```

所以：

```
pack_factor = 32 / 4 = 8
```

表示一个 `int32` 中打包 8 个 INT4。

最终对象大致是：

```
AutoGPTQConfig
├── weight_bits = 4
├── group_size = 128
├── desc_act = False
├── is_sym = True
├── pack_factor = 8
├── quant_type = uint4b8
└── full_config = 原始quantize_config
```

### 对 RoutedExperts 的处理

模型构造到 MoE 层时：

```
AutoGPTQConfig.get_quant_method(
    layer=RoutedExperts,
    prefix=...,
)
```

位于 [auto_gptq.py (line 240)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/auto_gptq.py:240)。

它先检查 MoE Marlin 是否支持当前 shape：

```
if not check_moe_marlin_supports_layer(...):
    return MoeWNA16Config.from_config(
        self.full_config
    ).get_quant_method(layer, prefix)
```

如果支持：

```
return AutoGPTQMoEMethod(
    cloned_auto_gptq_config,
    layer.moe_config,
)
```

因此有两个结果：

```
AutoGPTQConfig
├── 当前MoE shape支持Marlin
│   └── AutoGPTQMoEMethod
│       └── Oracle收到AutoGPTQConfig
│
└── 当前MoE shape不支持Marlin
    └── 转成MoeWNA16Config
        └── MoeWNA16Method
```

------

## 2. `AutoAWQConfig` 怎么构造

AWQ checkpoint 配置可能是：

```
{
  "quant_method": "awq",
  "w_bit": 4,
  "q_group_size": 128,
  "zero_point": true,
  "lm_head": false
}
```

有的 checkpoint 使用：

```
{
  "bits": 4,
  "group_size": 128
}
```

`AutoAWQConfig.from_config()` 同时支持两套字段名。

代码在 [auto_awq.py (line 238)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/auto_awq.py:238)：

```
weight_bits = cls.get_from_keys(
    config,
    ["w_bit", "bits"],
)

group_size = cls.get_from_keys(
    config,
    ["q_group_size", "group_size"],
)

zero_point = config["zero_point"]
lm_head_quantized = config.get("lm_head", False)
modules_to_not_convert = config.get(
    "modules_to_not_convert",
    None,
)

return AutoAWQConfig(
    weight_bits,
    group_size,
    zero_point,
    lm_head_quantized,
    modules_to_not_convert,
    full_config,
)
```

代入典型数值：

```
AutoAWQConfig(
    weight_bits=4,
    group_size=128,
    zero_point=True,
    lm_head_quantized=False,
    modules_to_not_convert=[],
    full_config=config,
)
```

构造函数同样计算：

```
self.pack_factor = 32 // 4
```

得到：

```
AutoAWQConfig
├── weight_bits = 4
├── group_size = 128
├── zero_point = True
├── pack_factor = 8
├── quant_type = uint4
└── full_config = 原始配置
```

### 对 MoE 层的处理

进入 `RoutedExperts` 后：

```
AutoAWQConfig.get_quant_method(...)
```

如果 Marlin 支持：

```
return AutoAWQMoEMethod(
    self,
    layer.moe_config,
)
```

如果不支持：

```
return MoeWNA16Config.from_config(
    self.full_config
).get_quant_method(layer, prefix)
```

所以也存在：

```
AutoAWQConfig
├── Marlin支持 → AutoAWQMoEMethod
└── Marlin不支持 → MoeWNA16Method
```

------

## 3. `QuantizationArgs` 怎么构造

`QuantizationArgs` 不是 vLLM 自己定义的 `QuantizationConfig` 子类，而是来自：

```
compressed_tensors.quantization.QuantizationArgs
```

它用于 compressed-tensors checkpoint。

这类 checkpoint 的 `config.json` 中通常包含：

```
{
  "quantization_config": {
    "format": "pack-quantized",
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "weights": {
          "num_bits": 4,
          "type": "int",
          "symmetric": true,
          "strategy": "group",
          "group_size": 128,
          "dynamic": false
        },
        "input_activations": null
      }
    }
  }
}
```

首先构造：

```
CompressedTensorsConfig.from_config(config)
```

代码在 [compressed_tensors.py (line 230)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:230)。

然后遍历 `config_groups`：

```
for _, quant_config in config_groups.items():
    for target in quant_config["targets"]:
        target_scheme_map[target]["weights"] = (
            QuantizationArgs.model_validate(
                quant_config["weights"]
            )
        )
```

位于 [compressed_tensors.py (line 300)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:300)。

也就是通过 Pydantic 验证并构造：

```
weight_quant = QuantizationArgs.model_validate(
    {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
    }
)
```

生成的对象大致为：

```
QuantizationArgs
├── num_bits = 4
├── type = INT
├── symmetric = True
├── strategy = GROUP
├── group_size = 128
├── actorder = None
└── dynamic = False
```

它被存入：

```
target_scheme_map["Linear"]["weights"]
```

创建 `RoutedExperts` 时：

```
weight_quant = scheme_dict["weights"]
input_quant = scheme_dict["input_activations"]
```

然后构造：

```
CompressedTensorsWNA16MoEMethod(
    weight_quant,
    input_quant,
    layer.moe_config,
)
```

代码在 [compressed_tensors_moe.py (line 128)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:128)。

这个 Method 再把 `QuantizationArgs` 传给 Oracle：

```
self.wna16_backend, self.experts_cls = (
    select_wna16_moe_backend(
        config=self.moe,
        weight_key=weight_key,
        quant_config=self.weight_quant,
        ...
    )
)
```

位于 [compressed_tensors_moe_wna16.py (line 108)](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py:108)。

------

## 4. 你的模型实际构造了哪个

你显式指定：

```
--quantization moe_wna16
```

因此全局首先构造的是：

```
MoeWNA16Config.from_config(...)
```

而不是把 MoE 层配置成 `AutoGPTQConfig`。

对于不同层：

```
MoeWNA16Config.get_quant_method()
│
├── 普通Linear层
│   └── AutoGPTQConfig.from_config(full_config)
│       └── 使用GPTQ/Marlin线性层实现
│
└── RoutedExperts MoE层
    └── MoeWNA16Method
        └── Oracle收到MoeWNA16Config
```

关键代码：

```
if isinstance(layer, LinearBase):
    return AutoGPTQConfig.from_config(
        self.full_config
    ).get_quant_method(layer, prefix)

elif isinstance(layer, RoutedExperts):
    return MoeWNA16Method(
        self,
        layer.moe_config,
    )
```

因此同一个模型内部可能同时有：

```
普通dense Linear
└── AutoGPTQConfig

MoE RoutedExperts
└── MoeWNA16Config
```

但你现在讨论的 WNA16 MoE Oracle 收到的是：

```
quant_config = MoeWNA16Config
```

所以：

```
allow_marlin = not isinstance(
    quant_config,
    MoeWNA16Config,
)
```

结果必定是：

```
allow_marlin = False
```

## 三条路径总结

```
GPTQ + 默认/gptq
└── AutoGPTQConfig
    ├── MoE支持Marlin → AutoGPTQMoEMethod
    └── 不支持       → MoeWNA16Method fallback

AWQ + 默认/awq
└── AutoAWQConfig
    ├── MoE支持Marlin → AutoAWQMoEMethod
    └── 不支持       → MoeWNA16Method fallback

Compressed-Tensors
└── CompressedTensorsConfig
    └── QuantizationArgs
        └── CompressedTensorsWNA16MoEMethod

显式 --quantization moe_wna16
└── MoeWNA16Config
    ├── Dense Linear → AutoGPTQ/AutoAWQ方法
    └── MoE          → MoeWNA16Method
```

所以 `AutoGPTQConfig`、`AutoAWQConfig`、`QuantizationArgs` 是三种不同 checkpoint 集成产生的配置对象，不是 Oracle 自己创建的三个候选后端。
