---
title: "Hugging Face"
published: 2026-05-13
description: "Hugging Face"
image: ""
tags: ["llm","Hugging Face"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-13T20:44:19.530.457029945Z"
---

# Hugging Face Complete API Usage

Hugging Face provides a unified ecosystem for machine learning, including the `transformers`, `datasets`, `tokenizers`, `huggingface_hub`, and `accelerate` libraries for model training, inference, and sharing.

## 1. Installation and Setup

Install the core libraries via pip and authenticate with your Hugging Face account token to access private models and push artifacts.

```bash
pip install transformers datasets tokenizers huggingface_hub accelerate evaluate
# Authenticate with Hugging Face Hub
from huggingface_hub import login

login(token="hf_xxxxxxxxxxxxxxxxxxxx")  # Replace with your token (令牌)
# Output: Token is valid. Your token has been saved.
```

<br>

## 2. Pipeline API

The `pipeline` API is the highest-level abstraction (抽象) for quick inference (推理) on common tasks with a single line of code.

```python
from transformers import pipeline

# Sentiment analysis (情感分析)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love using Hugging Face!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# Text generation (文本生成)
generator = pipeline("text-generation", model="gpt2")
output = generator("Once upon a time", max_length=20, num_return_sequences=1)
print(output)
# Output: [{'generated_text': 'Once upon a time, there was a ...'}]

# Zero-shot classification (零样本分类)
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = zsc("This is a tutorial about AI", candidate_labels=["education", "sports", "politics"])
print(result["labels"][0])
# Output: education
```

<br>

## 3. AutoClasses

`AutoModel` and `AutoTokenizer` automatically load the correct architecture (架构) based on the model name, avoiding hard-coded class imports.

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Load tokenizer (分词器) and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize input
inputs = tokenizer("Hello, Hugging Face!", return_tensors="pt", padding=True, truncation=True)
print(inputs["input_ids"].shape)
# Output: torch.Size([1, 7])

# Forward pass (前向传播)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# Output: torch.Size([1, 7, 768])
```

<br>

## 4. Tokenizer API

The tokenizer handles text-to-token conversion with padding (填充), truncation (截断), and special token (特殊标记) insertion.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode single sentence
encoded = tokenizer("Hello world", add_special_tokens=True)
print(encoded)
# Output: {'input_ids': [101, 7592, 2088, 102], 'attention_mask': [1, 1, 1, 1], ...}

# Batch encoding with padding
batch = tokenizer(
    ["Short text", "A much longer text for padding test"],
    padding=True,
    truncation=True,
    max_length=10,
    return_tensors="pt"
)
print(batch["input_ids"].shape)
# Output: torch.Size([2, 10])

# Decode tokens back to text
decoded = tokenizer.decode([101, 7592, 2088, 102], skip_special_tokens=True)
print(decoded)
# Output: hello world
```

<br>

## 5. Datasets API

The `datasets` library provides efficient memory-mapped (内存映射) loading and streaming access to thousands of datasets from the Hub.

```python
from datasets import load_dataset, Dataset

# Load from Hub
dataset = load_dataset("imdb", split="train[:100]")
print(dataset[0]["text"][:50])
# Output: I rented I AM CURIOUS-YELLOW from my video store...

# Map a preprocessing function
def add_length(example):
    example["length"] = len(example["text"])
    return example

dataset = dataset.map(add_length)
print(dataset[0]["length"])
# Output: 1640

# Create dataset from dict
custom = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
print(custom)
# Output: Dataset({features: ['text', 'label'], num_rows: 2})

# Streaming mode (流式模式) for large datasets
stream = load_dataset("c4", "en", split="train", streaming=True)
first = next(iter(stream))
print(list(first.keys()))
# Output: ['text', 'timestamp', 'url']
```

<br>

## 6. Trainer API

The `Trainer` class provides a high-level training loop with built-in support for mixed precision (混合精度), gradient accumulation (梯度累积), and distributed training (分布式训练).

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

dataset = load_dataset("imdb", split="train[:200]")
def tokenize_fn(ex):
    return tokenizer(ex["text"], padding="max_length", truncation=True, max_length=128)
dataset = dataset.map(tokenize_fn, batched=True)

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
# Output: {'loss': 0.6932, 'learning_rate': 1.8e-05, 'epoch': 0.4} ...
```

<br>

## 7. Generation API

The `generate()` method supports multiple decoding strategies (解码策略) including greedy, beam search, and sampling for autoregressive (自回归) models.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("The future of AI is", return_tensors="pt")

# Greedy decoding (贪心解码)
greedy = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(greedy[0], skip_special_tokens=True))
# Output: The future of AI is uncertain, but it is...

# Beam search (束搜索)
beam = model.generate(**inputs, max_new_tokens=10, num_beams=4, early_stopping=True)
print(tokenizer.decode(beam[0], skip_special_tokens=True))
# Output: The future of AI is uncertain, but one thing...

# Top-k / Top-p sampling (采样)
sampled = model.generate(
    **inputs, max_new_tokens=10, do_sample=True,
    top_k=50, top_p=0.95, temperature=0.8
)
print(tokenizer.decode(sampled[0], skip_special_tokens=True))
# Output: The future of AI is bright and full of...
```

<br>

## 8. Hugging Face Hub API

The `huggingface_hub` library lets you programmatically download, upload, and manage repositories (仓库) on the Hub.

```python
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

api = HfApi()

# Download a single file
file_path = hf_hub_download(repo_id="bert-base-uncased", filename="config.json")
print(file_path)
# Output: /root/.cache/huggingface/hub/.../config.json

# Download entire repo snapshot (快照)
local_dir = snapshot_download(repo_id="bert-base-uncased", allow_patterns=["*.json"])
print(local_dir)
# Output: /root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/...

# Create a repo and upload a file
api.create_repo(repo_id="my-username/my-model", exist_ok=True, private=False)
api.upload_file(
    path_or_fileobj="./model.bin",
    path_in_repo="model.bin",
    repo_id="my-username/my-model",
)
# Output: CommitInfo(commit_url='https://huggingface.co/...')
```

<br>

## 9. Accelerate API

`Accelerate` abstracts away device placement (设备放置) and distributed training boilerplate, letting the same script run on CPU, single GPU, or multi-GPU.

```python
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, TensorDataset

accelerator = Accelerator()

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
loader = DataLoader(dataset, batch_size=8)

# Prepare for distributed/mixed-precision
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

for x, y in loader:
    optimizer.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.cross_entropy(pred, y)
    accelerator.backward(loss)  # Replaces loss.backward()
    optimizer.step()
    break
print(f"Loss: {loss.item():.4f}")
# Output: Loss: 0.7321
```

<br>

## 10. Evaluate API

The `evaluate` library provides standardized metrics (指标) such as accuracy, F1, BLEU, and ROUGE with a unified interface.

```python
import evaluate

# Load a metric
accuracy = evaluate.load("accuracy")
result = accuracy.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 0])
print(result)
# Output: {'accuracy': 0.75}

# Combine multiple metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
result = clf_metrics.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 0])
print(result)
# Output: {'accuracy': 0.75, 'f1': 0.667, 'precision': 0.5, 'recall': 1.0}
```

<br>

## 11. PEFT API

`PEFT` (Parameter-Efficient Fine-Tuning, 参数高效微调) enables techniques like LoRA (低秩适配) to fine-tune large models with minimal trainable parameters.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                  # Rank (秩)
    lora_alpha=32,        # Scaling factor
    lora_dropout=0.1,
    target_modules=["c_attn"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364
```

<br>

## 12. Inference API (Serverless)

The `InferenceClient` calls hosted models via HTTP, removing the need to load weights locally.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_xxxxxxxxxxxxxxxx")

# Text generation
result = client.text_generation(
    "The capital of France is",
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=10,
)
print(result)
# Output:  Paris. It is located in the north-central part...

# Image classification
result = client.image_classification("cat.jpg", model="google/vit-base-patch16-224")
print(result[0])
# Output: ImageClassificationOutputElement(label='tabby cat', score=0.92)

# Chat completion (OpenAI-compatible interface)
messages = [{"role": "user", "content": "Hello!"}]
response = client.chat_completion(messages=messages, model="meta-llama/Llama-3.1-8B-Instruct", max_tokens=20)
print(response.choices[0].message.content)
# Output: Hello! How can I help you today?
```

<br>

## 13. Saving and Loading Models

Models, tokenizers, and configs can be persisted locally with `save_pretrained()` and reloaded with `from_pretrained()` for offline (离线) inference.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Save locally
tokenizer.save_pretrained("./my_bert")
model.save_pretrained("./my_bert")
# Output: Files saved: config.json, pytorch_model.bin, tokenizer.json, vocab.txt

# Reload from local path
tokenizer2 = AutoTokenizer.from_pretrained("./my_bert")
model2 = AutoModel.from_pretrained("./my_bert")
print(type(model2).__name__)
# Output: BertModel

# Push to Hub
model.push_to_hub("my-username/my-bert")
tokenizer.push_to_hub("my-username/my-bert")
# Output: CommitInfo(commit_url='https://huggingface.co/my-username/my-bert/commit/...')
```

<br> <br>
