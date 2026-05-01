---
title: "GGUF"
published: 2026-04-30
description: "GGUF"
image: ""
tags: ["llm","GGUF"]
category: llm
draft: false
lang: ""
createdAt: "2026-04-30T21:03:19.324.256981938Z"
---

# GGUF (GPT-Generated Unified Format)

GGUF is a binary file format designed for storing and loading quantized large language models efficiently for inference.

## 1. What is GGUF

GGUF (GPT-Generated Unified Format) is a single-file binary format introduced by the llama.cpp team in 2023, designed to store model weights, metadata (元数据), and tokenizer (分词器) information together for fast loading and CPU/GPU inference (推理).

<br>

## 2. What does it look like

A GGUF file is a single `.gguf` binary file with three main sections: a **header** (魔数 magic number `GGUF` + version), a **metadata key-value section** (storing architecture, hyperparameters, tokenizer vocab), and a **tensor data section** (storing quantized (量化) weights like Q4_K_M, Q8_0).

The structure looks like:

```
┌─────────────────────────────┐
│  Header: "GGUF" + version   │
├─────────────────────────────┤
│  Metadata (KV pairs)        │
│  - general.architecture     │
│  - llama.context_length     │
│  - tokenizer.ggml.tokens    │
├─────────────────────────────┤
│  Tensor Info (name, shape)  │
├─────────────────────────────┤
│  Tensor Data (quantized)    │
└─────────────────────────────┘
```

<br>

## 3. Example

A typical filename looks like `llama-3-8b-instruct.Q4_K_M.gguf`, where `Q4_K_M` means 4-bit quantization (4位量化) with K-type mixed precision (混合精度), reducing an 8B model from ~16GB (FP16) down to ~4.5GB so it can run on a consumer laptop via `llama.cpp` or `Ollama`.

```cpp
// Loading a GGUF model in llama.cpp
#include "llama.h"

llama_model_params params = llama_model_default_params();
llama_model* model = llama_load_model_from_file(
    "llama-3-8b-instruct.Q4_K_M.gguf", params
);
```

<br> <br>

# GGUF Example 

A concrete walkthrough of what's actually inside a real GGUF file.

## 1. File Example: Llama-3-8B-Instruct Q4_K_M

Take the popular file `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` (~4.92 GB), downloaded from Hugging Face — this is a 4-bit quantized (4位量化) version of Meta's Llama 3 8B model.

<br>

## 2. Header (文件头)

The first bytes of the file in hexadecimal (十六进制):

```
47 47 55 46    → "GGUF" magic number (魔数)
03 00 00 00    → version = 3
```

This tells the loader: "I am a GGUF v3 file."

<br>

## 3. Metadata (元数据)

Real key-value pairs you would see when inspecting this file:

```
general.architecture           = "llama"
general.name                   = "Meta-Llama-3-8B-Instruct"
general.file_type              = 15                    (Q4_K_M)
llama.context_length           = 8192                  (上下文窗口)
llama.embedding_length         = 4096                  (隐藏维度 hidden dim)
llama.block_count              = 32                    (Transformer 层数)
llama.attention.head_count     = 32                    (注意力头数)
llama.attention.head_count_kv  = 8                     (GQA, 分组查询注意力)
llama.rope.freq_base           = 500000.0              (RoPE 位置编码)
tokenizer.ggml.model           = "gpt2"                (BPE tokenizer)
tokenizer.ggml.tokens          = ["!", "\"", "#", ...]  (128256 tokens)
tokenizer.ggml.bos_token_id    = 128000
tokenizer.ggml.eos_token_id    = 128009
```

This **fully describes** how to rebuild the Llama 3 architecture without any external `config.json`.

<br>

## 4. Tensor Info (张量信息)

A list of all 291 tensors with their names, shapes (形状), and offsets (偏移量):

```
token_embd.weight              shape=[4096, 128256]   type=Q4_K   offset=0
blk.0.attn_q.weight            shape=[4096, 4096]     type=Q4_K   offset=...
blk.0.attn_k.weight            shape=[1024, 4096]     type=Q4_K   offset=...
blk.0.attn_v.weight            shape=[1024, 4096]     type=Q6_K   offset=...
blk.0.attn_output.weight       shape=[4096, 4096]     type=Q4_K   offset=...
blk.0.ffn_gate.weight          shape=[14336, 4096]    type=Q4_K   offset=...
blk.0.ffn_down.weight          shape=[4096, 14336]    type=Q6_K   offset=...
blk.0.ffn_up.weight            shape=[14336, 4096]    type=Q4_K   offset=...
... (repeats for blk.1 ~ blk.31)
output_norm.weight             shape=[4096]           type=F32
output.weight                  shape=[4096, 128256]   type=Q6_K
```

The loader uses these offsets to **memory-map (mmap 内存映射)** any weight directly without reading the whole file.

<br>

## 5. Tensor Data (张量数据)

The remaining ~4.9 GB is the raw quantized (量化) weight bytes — for example, Q4_K_M means weights are stored in 4-bit blocks of 32 values, each block sharing one FP16 scale (缩放因子) and minimum, achieving ~4.5 bits per parameter on average.

<br>

## 6. How to Use It (使用方式)

```bash
# Run with llama.cpp
./llama-cli -m Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
            -p "What is GGUF?" -n 128

# Or with Ollama
ollama run llama3:8b-instruct-q4_K_M
```

The model loads in **~2 seconds** via mmap, runs on CPU or GPU, and uses only ~5 GB RAM instead of 16 GB (FP16 original).

<br> <br>
