---
title: "vLLM Kimi3"
published: 2026-08-12
description: "vLLM Kimi3"
image: ""
tags: ["llm_inference","vLLM Kimi3"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-12T20:51:49.658.332927739Z"
---



# KimiK3Config

三个类是 **一个外壳、两个零件**。K3 是多模态模型，HF `config.json` 顶层是 `KimiK3Config`；里面嵌两块：语言模型用 `KimiLinearConfig`，视觉塔用 `KimiK3VisionConfig`。

```
KimiK3Config                    ← 整机（多模态）
├── text_config: KimiLinearConfig     ← 语言模型（MLA / MoE / KDA）
└── vision_config: KimiK3VisionConfig ← ViT + projector
```

`KimiLinearConfig` 还能单独当纯文本模型用。`KimiK3VisionConfig` 不能单独 serve。

---

## 1. `KimiK3Config`：顶层外壳

`model_type = "kimi_k3"`。构造时把 `text_config` / `vision_config` 从 dict 转成上面两个类：

```86:111:vllm/transformers_utils/configs/kimi_k3.py
class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(
        self,
        text_config: dict | KimiLinearConfig | None = None,
        vision_config: dict | KimiK3VisionConfig | None = None,
        ...
    ):
        if text_config is None:
            self.text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            self.text_config = KimiLinearConfig(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = KimiK3VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = KimiK3VisionConfig(**vision_config)
```

它自己几乎不存模型超参，只存多模态胶水字段：`media_placeholder_token_id`、`image_placeholder`、`ignore_index`。`hidden_size` / `vocab_size` 直接转发给 text：

```133:139:vllm/transformers_utils/configs/kimi_k3.py
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
```

唯一的跨模块约束：projector 输出维必须等于 LM hidden size，对不上就改 vision 那边：

```113:122:vllm/transformers_utils/configs/kimi_k3.py
        # K3's vision projector output must match the text model's hidden size.
        if self.vision_config.text_hidden_size != self.text_config.hidden_size:
            ...
            self.vision_config.text_hidden_size = self.text_config.hidden_size
```

模型侧：`KimiK3ForConditionalGeneration` 拿整份 `KimiK3Config`，再拆开用：

```1561:1630:vllm/models/kimi_k3/nvidia/model.py
        config: KimiK3Config = model_config.hf_config
        ...
        self.vision_tower = MoonViT3dPretrainedModel(config.vision_config, ...)
        self.mm_projector = KimiK25MultiModalProjector(config=config.vision_config, ...)
        self.language_model = init_vllm_registered_model(
            ...,
            hf_config=config.text_config,
            architectures=["KimiLinearForCausalLM"],
        )
```

也就是：外壳 → 视觉用 `vision_config`，语言用 `text_config` 去实例化一个 `KimiLinearForCausalLM`。

---

## 2. `KimiK3VisionConfig`：ViT + projector

`model_type = "kimi_k3_vision"`。全是视觉超参，分两摊：

**ViT（`vt_*`）**

- `patch_size=14`，位置编码 `init_pos_emb_*` / `pos_emb_type`
- `vt_num_hidden_layers=27`、`vt_hidden_size=1024`、`vt_num_attention_heads=12`
- `merge_kernel_size=(2,2)`、`merge_type`、`video_attn_type`

**Projector**

- `mm_projector_type="patchmergerv2"`
- `mm_hidden_size`（默认等于 `vt_hidden_size`）
- `text_hidden_size=2304`（要对齐 LM，上面会被外壳改掉）

后面几行是给 K2.5 视觉实现用的别名，所以同一套 ViT 代码既能读 `vt_hidden_size` 也能读 `hidden_size`：

```79:83:vllm/transformers_utils/configs/kimi_k3.py
        self.num_attention_heads = vt_num_attention_heads
        self.num_hidden_layers = vt_num_hidden_layers
        self.hidden_size = vt_hidden_size
        self.intermediate_size = vt_intermediate_size
```

它**不是**独立 HF 模型。`config.py` 只按顶层 `model_type` 注册了 `kimi_k3 → KimiK3Config`，没有 `kimi_k3_vision` 这条 serve 入口。

---

## 3. `KimiLinearConfig`：语言模型本体

`model_type = "kimi_linear"`。这是真正的 LLM 超参，和视觉无关：

- 通用 Transformer：`vocab_size`、`hidden_size`、`num_hidden_layers`、RoPE
- **MLA**：`q_lora_rank` / `kv_lora_rank` / `qk_*_head_dim`（`is_mla`）
- **MoE**：`num_experts`、`moe_intermediate_size`、`first_k_dense_replace`（`is_moe`）
- **线性注意力 KDA**：`linear_attn_config` 里哪些层是 KDA、哪些是 full attn（`is_kda_layer`）

```131:161:vllm/transformers_utils/configs/kimi_linear.py
    def is_mla(self): ...
    def is_moe(self): ...
    def is_linear_attn(self) -> bool: ...
    def is_kda_layer(self, layer_idx: int): ...
```

两种用法：

1. **嵌在 K3 里**：`KimiK3Config.text_config`，给 `language_model` 用。层实现（MLA / KDA / MTP）拿的都是它，例如 `KimiLinearModel` 读 `hf_text_config`。
2. **单独当纯文本模型**：registry 里有 `KimiLinearForCausalLM`，`model_type="kimi_linear"` 直接映射到这个 config。没有 vision。

MTP 也说明了嵌套关系：K3 的 MTP 层数在 `text_config` 上，不在顶层：

```364:368:vllm/config/speculative.py
        if hf_config.model_type == "kimi_k3":
            # Kimi-K3 keeps the text-model fields (incl. the MTP layer count)
            # nested under ``text_config`` (a KimiLinearConfig).
            text_config = getattr(hf_config, "text_config", hf_config)
            n_predict = getattr(text_config, "num_nextn_predict_layers", None)
```

---

## 三者关系

|                      | 角色            | 谁拥有它                                 | 谁消费它                                                 |
| -------------------- | --------------- | ---------------------------------------- | -------------------------------------------------------- |
| `KimiK3Config`       | 多模态整机      | HF 顶层 `config.json`                    | `KimiK3ForConditionalGeneration`                         |
| `KimiLinearConfig`   | 语言模型        | `KimiK3Config.text_config`，或自己当顶层 | `KimiLinearForCausalLM` / MLA / MoE / KDA                |
| `KimiK3VisionConfig` | ViT + projector | 只作为 `KimiK3Config.vision_config`      | `MoonViT3dPretrainedModel`、`KimiK25MultiModalProjector` |

连接点只有一个：`vision_config.text_hidden_size == text_config.hidden_size`，这样图像特征才能塞进语言模型。





