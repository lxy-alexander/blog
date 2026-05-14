---
title: "LoRA"
published: 2026-05-13
description: "LoRA"
image: ""
tags: ["llm","LoRA"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-13T22:14:44.088.457187561Z"
---

# LoRA in Large Language Model Applications

LoRA (Low-Rank Adaptation, 低秩适配) is ==a lightweight fine-tuning (微调) method== that freezes the original model weights and only trains a small number of additional parameters. It usually applied to q_proj and v_proj. training becomes cheaper and faster.

-   q: What does the current token want to query. 
-   k: What features does the current token have that can be matched by others. 
-   v: The information content that the current token actually provides to others

==Q and K are used to compute the attention score to determine who to focus on.==

==V is used for weighted summation to determine what content is obtained(决定拿到什么内容=什么内容被拿到).==

<br>

## 1. Core Idea

LoRA hypothesizes that the weight update (权重更新) during fine-tuning has a low intrinsic rank (内在秩), so $\Delta W$ can be approximated by the product of two small matrices instead of training the full matrix.

The full weight update is decomposed as:

$$ W = W_0 + \Delta W = W_0 + BA $$

Where $W_0 \in \mathbb{R}^{d \times k}$ is the frozen (冻结) pretrained weight, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable, and $r \ll \min(d, k)$ is the rank (秩).<br>

## 2. Why LoRA Works

Fine-tuning large language models updates billions of parameters, but research shows the update matrix $\Delta W$ is intrinsically low-rank, meaning most useful adaptation lives in a small subspace (子空间).

Key advantages in interview tone: **no inference latency (推理延迟)** because $BA$ can be merged into $W_0$, **10000x fewer trainable parameters**, and **task switching** by swapping small adapter weights instead of full models.

<br>

## 3. Initialization Strategy

$A$ is initialized with a random Gaussian (高斯分布) distribution and $B$ is initialized to zero, so $\Delta W = BA = 0$ at the start of training, ensuring the model begins identical to the pretrained baseline.

$$ A \sim \mathcal{N}(0, \sigma^2), \quad B = 0 ;\Rightarrow; \Delta W_{\text{init}} = 0 $$

<br>

## 4. Scaling Factor Alpha

LoRA scales the update by $\frac{\alpha}{r}$ so that tuning $r$ does not require retuning the learning rate (学习率); $\alpha$ acts like a learning rate multiplier on the adapter.

$$ h = W_0 x + \frac{\alpha}{r} B A x $$

In practice, $\alpha$ is often set to $2r$ (e.g., $r=8, \alpha=16$) as a stable default.

<br>

## 5. Target Modules

LoRA is typically applied to the attention (注意力) projection matrices $W_q$, $W_k$, $W_v$, $W_o$ in Transformer (变换器) blocks, since these dominate the parameter count and adaptation impact.

The original paper found that **applying LoRA to $W_q$ and $W_v$** gives the best parameter-performance trade-off (权衡), while applying it to all four is best when budget allows.<br>

## 6. Parameter Count Comparison

For a single linear layer with input dim $d$ and output dim $k$, full fine-tuning trains $d \times k$ parameters, while LoRA trains only $r \times (d + k)$ parameters.

$$ \text{Reduction Ratio} = \frac{r(d + k)}{dk} $$

For example, with $d = k = 4096$ and $r = 8$, LoRA trains $65{,}536$ params vs. full fine-tuning's $16{,}777{,}216$ — a **~256x reduction (减少)** in trainable parameters.

<br>

## 7. Merge for Zero Inference Latency

After training, the adapter can be folded into the base weights via $W' = W_0 + BA$, so the deployed model has the **exact same architecture and speed** as the original — there is no extra matrix multiplication at inference time.

$$ W'_{\text{merged}} = W_0 + \frac{\alpha}{r} BA $$

<br>

## 8. LoRA Training Workflow

The end-to-end workflow involves freezing the base model, injecting adapters into target modules, training only the adapters, and optionally merging them back for deployment (部署).<br>

## 9. Code Example with PEFT

The Hugging Face `peft` library provides a one-liner to attach LoRA adapters to any `transformers` model, dramatically simplifying the workflow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load base model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                         # Rank (秩)
    lora_alpha=16,               # Alpha scaling factor
    lora_dropout=0.05,           # Dropout (随机失活)
    target_modules=["c_attn"],   # Target attention projections
    bias="none",
)

# 3. Wrap model with PEFT
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364

# 4. Inspect what's trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
        break
# Output: base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight torch.Size([8, 768])

# 5. Forward pass works as usual
inputs = tokenizer("LoRA is", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.logits.shape)
# Output: torch.Size([1, 3, 50257])

# 6. Save only adapter weights (small file)
model.save_pretrained("./lora_adapter")
# Output: Saves ~1MB instead of ~500MB full model

# 7. Merge for inference
merged_model = model.merge_and_unload()
print(type(merged_model).__name__)
# Output: GPT2LMHeadModel
```

<br>

## 10. LoRA vs. Other PEFT Methods

LoRA differs from adapters (适配器) and prefix tuning (前缀微调) in *where* and *how* it modifies the model, with each method offering different trade-offs in parameters, latency, and quality.

| Method               | Trainable Params | Inference Latency     | Where Applied                    |
| -------------------- | ---------------- | --------------------- | -------------------------------- |
| **Full fine-tuning** | 100%             | None                  | All weights                      |
| **LoRA**             | ~0.1–1%          | None (after merge)    | Linear layers (mostly attention) |
| **Adapters**         | ~0.5–8%          | Adds extra layers     | Between Transformer blocks       |
| **Prefix tuning**    | ~0.1%            | Slight (extra tokens) | Input/key-value prefix           |
| **Prompt tuning**    | ~0.01%           | Slight (extra tokens) | Input embeddings only            |

<br>

## 11. Variants and Extensions

Several follow-up works extend LoRA: **QLoRA** quantizes (量化) the frozen base to 4-bit so a 65B model fits on one GPU, **DoRA** decomposes weights into magnitude and direction for better quality, and **AdaLoRA** dynamically allocates rank budget across layers based on importance.

In interview tone: when GPU memory is tight, reach for **QLoRA**; when you want LoRA-level params with full-FT quality, try **DoRA**; when different layers need different capacity, use **AdaLoRA**.

<br>

## 12. Practical Tips

Choose $r$ between 4 and 64 — larger ranks help for harder tasks but with diminishing returns (收益递减); keep $\alpha = 2r$ as a safe default; always target $W_q$ and $W_v$ first before expanding; and use `lora_dropout=0.05–0.1` to prevent overfitting (过拟合) on small datasets.

<br> <br>

## 8. Common Interview Questions

### 1) Why Is LoRA Efficient?

Because it trains only low-rank matrices instead of updating all model weights.

### 2) Why Freeze Original Parameters?

Freezing reduces memory usage and preserves pretrained knowledge.

### 3) Why Usually Apply LoRA on Attention Layers?

Attention layers contribute most to model capability.

### 4) Difference Between LoRA and Adapter

LoRA modifies weight updates mathematically, while Adapter adds extra neural layers.

<br> <br>
