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

# LoRA 

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



## 2. Common Interview Questions

### 1) Why Is LoRA Efficient?

Because it trains only low-rank matrices instead of updating all model weights.

### 2) Why Freeze Original Parameters?

Freezing reduces memory usage and preserves pretrained knowledge.

### 3) Why Usually Apply LoRA on Attention Layers?

Attention layers contribute most to model capability.

### 4) Difference Between LoRA and Adapter

LoRA modifies weight updates mathematically, while Adapter adds extra neural layers.

<br> <br>



# LoRA Full Workflow

LoRA training can be understood as: **load the base model → freeze original weights → inject LoRA adapters → train only adapters → save adapters → load or merge adapters for inference**.

## 1. Load the Base Model

First, load a pretrained large language model, such as Qwen, LLaMA, or Mistral.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
)
```

At this point, the model is still the original Transformer model.

```text
Base Model
├── Transformer Layer 0
│   ├── Self-Attention
│   │   ├── q_proj
│   │   ├── k_proj
│   │   ├── v_proj
│   │   └── o_proj
│   └── MLP / FFN
│       ├── gate_proj
│       ├── up_proj
│       └── down_proj
├── Transformer Layer 1
└── ...
```

## 2. Freeze the Base Model

The key idea of LoRA is to keep the original model weights frozen and train only small additional matrices.

Full fine-tuning updates:

```text
Wq, Wk, Wv, Wo, MLP weights, LayerNorm, Embedding, ...
```

LoRA fine-tuning updates:

```text
Only LoRA A and LoRA B matrices.
```

The original Linear layer is:

$$
y = Wx
$$

After LoRA injection:

$$
y = Wx + BAx
$$

Where:

| Symbol | Meaning                          |
| ------ | -------------------------------- |
| $W$    | Frozen original weight matrix    |
| $A$    | Trainable down-projection matrix |
| $B$    | Trainable up-projection matrix   |
| $BA$   | Low-rank update                  |

## 3. Define LoRA Configuration

The LoRA configuration tells PEFT where to insert adapters and how large the low-rank update should be.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
```

| Parameter        | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| `r`              | LoRA rank, which controls the bottleneck dimension.          |
| `lora_alpha`     | Scaling factor that controls the strength of LoRA updates.   |
| `lora_dropout`   | Dropout used to reduce overfitting.                          |
| `bias`           | Whether to train bias parameters, usually set to `"none"`.   |
| `task_type`      | Task type, and causal language modeling uses `"CAUSAL_LM"`.  |
| `target_modules` | The Linear layers where LoRA adapters will be inserted.      |
| q_proj           | Query projection (查询投影): decides what each token looks for. |
| k_proj           | Key projection (键投影): decides what each token offers for matching. |
| v_proj           | Value projection (值投影): stores information passed through Attention. |
| o_proj           | Output projection (输出投影): maps Attention output back to hidden size. |
| gate_proj        | Gate projection (门控投影): controls gated activation in the MLP/FFN block. |
| up_proj          | Up projection (升维投影): expands hidden states to a larger dimension. |
| down_proj        | Down projection (降维投影): compresses hidden states back to model dimension. |

## 4. Inject LoRA Adapters

This line inserts LoRA adapters into the selected target modules.

```python
from peft import get_peft_model

model = get_peft_model(base_model, lora_config)
```

Before LoRA:

```text
q_proj = Linear(hidden_size, hidden_size)
```

After LoRA:

```text
q_proj = Linear(hidden_size, hidden_size)
       + lora_A: Linear(hidden_size, r)
       + lora_B: Linear(r, hidden_size)
```

The computation becomes:

```text
Before LoRA:

x ──> W ──> y


After LoRA:

        ┌──> frozen W ─────────────┐
x ──────┤                          ├──> y
        └──> trainable A -> B ─────┘
```

Adapters are injected into the selected Transformer modules.

```text
Transformer Layer
├── Self-Attention
│   ├── q_proj  ← add lora_A + lora_B
│   ├── k_proj  ← add lora_A + lora_B
│   ├── v_proj  ← add lora_A + lora_B
│   └── o_proj  ← add lora_A + lora_B
└── MLP / FFN
    ├── gate_proj ← add lora_A + lora_B
    ├── up_proj   ← add lora_A + lora_B
    └── down_proj ← add lora_A + lora_B
```

You can verify the adapter locations with:

```python
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name)
```

Example output:

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default
base_model.model.model.layers.0.self_attn.v_proj.lora_A.default
base_model.model.model.layers.0.self_attn.v_proj.lora_B.default
base_model.model.model.layers.0.mlp.gate_proj.lora_A.default
base_model.model.model.layers.0.mlp.gate_proj.lora_B.default
```

## 5. Forward Pass

During forward propagation, the frozen Linear layer and the trainable LoRA branch work together.

```text
Input text
↓
Tokenizer
↓
Token IDs
↓
Embedding
↓
Transformer layers
↓
For each target Linear layer:
    output = frozen_linear(x) + lora_B(lora_A(x)) * scale
↓
Logits
↓
Loss
```

The full LoRA formula is:

$$
y = Wx + \frac{\alpha}{r}BAx
$$

Where:

| Term         | Meaning                        |
| ------------ | ------------------------------ |
| $W$          | Frozen base weight             |
| $A$          | Trainable LoRA down-projection |
| $B$          | Trainable LoRA up-projection   |
| $\alpha / r$ | Scaling factor                 |

LoRA does not replace the original Linear layer; it adds a trainable side branch to the frozen layer.

## 6. Backward Pass

During backpropagation, only LoRA adapter parameters are updated.

```text
Loss
↓
Backward
↓
Update lora_A and lora_B
↓
Keep original W frozen
```

Updated parameters:

```text
lora_A
lora_B
```

Frozen parameters:

```text
q_proj.weight
k_proj.weight
v_proj.weight
o_proj.weight
gate_proj.weight
up_proj.weight
down_proj.weight
embedding weights
most original model weights
```

You can check trainable parameters with:

```python
model.print_trainable_parameters()
```

Example output:

```text
trainable params: 4,xxx,xxx || all params: 500,xxx,xxx || trainable%: 0.8
```

This is why LoRA saves GPU memory and training cost.

## 7. Train the Model

The training dataset usually contains instruction-response pairs.

```python
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

train_data = [
    {
        "text": "### Instruction:\nExplain LoRA.\n\n### Response:\nLoRA freezes the base model and trains low-rank adapters."
    }
]

dataset = Dataset.from_list(train_data) # convert a normal Python list into a Hugging Face Dataset object.

training_args = TrainingArguments(
    output_dir="./qwen_lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=20,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()
```

During training, the base model stays frozen and only the LoRA adapter weights change.

## 8. Save the Adapter

After training, the common practice is to save only the LoRA adapter.

```python
model.save_pretrained("./qwen_lora_adapter")
tokenizer.save_pretrained("./qwen_lora_adapter")
```

The output folder usually looks like this:

```text
qwen_lora_adapter
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

| File                        | Purpose                    |
| --------------------------- | -------------------------- |
| `adapter_config.json`       | Stores LoRA configuration. |
| `adapter_model.safetensors` | Stores LoRA A/B weights.   |
| Tokenizer files             | Store tokenizer settings.  |

The adapter is small because it does not contain the full base model.

## 9. Inference with LoRA Adapter

For inference, load the base model first, then attach the LoRA adapter.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen_lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "Explain LoRA in one sentence."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Output example:
# Explain LoRA in one sentence.
# LoRA fine-tunes a frozen large model by training small low-rank adapter matrices.
```

## 10. Merge LoRA for Deployment

For deployment, LoRA can be merged into the base model weights.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "./qwen_lora_adapter"
merged_model_path = "./qwen_lora_merged"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)

merged_model = model.merge_and_unload()

merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

# Output:
# The merged model is saved to ./qwen_lora_merged
```

After merging:

```text
Before merge:
output = Wx + LoRA(x)

After merge:
W_new = W + LoRA_weight
output = W_new x
```

So inference no longer needs a separate LoRA adapter branch.

## 11. Final Interview Summary

LoRA freezes the original large language model, injects small trainable low-rank adapters into selected Linear layers, trains only those adapters, and optionally merges them back into the base model for deployment.

```text
Base Model
↓
Freeze Original Weights
↓
Inject LoRA A/B Matrices
↓
Train Only LoRA Parameters
↓
Save Adapter
↓
Load Adapter or Merge for Deployment
```

