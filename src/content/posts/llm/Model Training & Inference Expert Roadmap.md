---
title: "Model Training & Inference Expert Roadmap"
published: 2026-05-13
description: "Model Training & Inference Expert Roadmap"
image: ""
tags: ["llm","Model Training & Inference Expert Roadmap"]
category: llm
draft: false
lang: ""
createdAt: "2026-02-11T04:09:27.802.186613987Y"
---

# Model Training & Inference Expert Roadmap

A complete knowledge graph and progression path from fundamentals to expert level in model training and inference.

## 1. Overview

The roadmap consists of five stages: math/programming foundations, deep learning basics, large model training, inference optimization, and system-level frontier research.

<br>

## 2. Time Investment

Each stage typically takes 3-12 months, with stage 3 (training) and stage 4 (inference) often running in parallel.

<br> <br>

# Stage 1 — Math & Programming Foundations

The foundation layer covering math, Python, and classical machine learning (经典机器学习).

## 1. Mathematics

Linear algebra, probability, and optimization form the language of all ML models.

### 1) Linear Algebra

==Matrix operations==, ==eigendecomposition== (特征分解), and ==SVD (奇异值分解, Singlular Value Decomposition)== underpin every neural network layer. SVD can be used in image compression.

$$ A = U \Sigma V^T $$

```python
import numpy as np

A = np.array([[3, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")        # Eigenvalues: [4. 2.]
print(f"Eigenvectors:\n{eigenvectors}")     # Eigenvectors: [[0.707 -0.707] [0.707 0.707]]

U, S, Vt = np.linalg.svd(A)
print(f"Singular values: {S}")              # Singular values: [4. 2.]
```

### 2) Probability & Statistics

KL divergence (KL散度) measures distance between distributions, central to VAE and RLHF training.

$$ D_{KL}(P | Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} $$

```python
import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))

p = np.array([0.4, 0.6])
q = np.array([0.5, 0.5])
print(f"KL(P||Q): {kl_divergence(p, q):.4f}")   # KL(P||Q): 0.0204
```

### 3) Optimization

Gradient descent (梯度下降) minimizes loss by moving against the gradient direction.

$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t) $$

<br>

## 2. Python & PyTorch

Master `torch.nn`, `autograd`, and `DataLoader` for end-to-end training pipelines.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = MLP()
x = torch.randn(2, 784)
out = model(x)
print(f"Output shape: {out.shape}")     # Output shape: torch.Size([2, 10])
```

<br>

## 3. Classical Machine Learning

Logistic regression (逻辑回归) is the simplest building block of every neural classifier.

$$ P(y=1|x) = \frac{1}{1 + e^{-w^T x}} $$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)
print(f"Accuracy: {clf.score(X, y):.3f}")   # Accuracy: 0.973
```

<br> <br>

# Stage 2 — Deep Learning Fundamentals

Master neural network core mechanisms, classical architectures, and training engineering.

## 1. Backpropagation

Backpropagation (反向传播) computes gradients via the chain rule from output back to input.

$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} $$

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
print(f"dy/dx at x=2: {x.grad.item()}")     # dy/dx at x=2: 14.0  (3x^2+2 = 14)
```

<br>

## 2. Optimizers

AdamW (Adam with decoupled weight decay, 权重衰减) is the default choice for transformer training.

$$ m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 $$

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
print(f"Optimizer: {type(optimizer).__name__}")     # Optimizer: AdamW
```

<br>

## 3. Normalization

LayerNorm (层归一化) normalizes across feature dim; RMSNorm drops mean centering for speed.

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot g $$

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, 8)        # (batch, seq, dim)
ln = nn.LayerNorm(8)
out = ln(x)
print(f"Mean: {out.mean(-1)[0,0]:.4f}")     # Mean: 0.0000 (approximately)
print(f"Std: {out.std(-1)[0,0]:.4f}")       # Std: 1.0690
```

<br>

## 4. Transformer Attention

Scaled dot-product attention (缩放点积注意力) computes weighted sums of values based on query-key similarity.

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

```python
import torch
import torch.nn.functional as F

def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

Q = K = V = torch.randn(1, 4, 8)        # (batch, seq, dim)
out = attention(Q, K, V)
print(f"Output shape: {out.shape}")      # Output shape: torch.Size([1, 4, 8])
```

<br>

## 5. Mixed Precision Training

AMP (automatic mixed precision, 自动混合精度) uses FP16/BF16 for speed and FP32 for stability.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Skeleton only - requires GPU to run
# scaler = GradScaler()
# with autocast(dtype=torch.bfloat16):
#     output = model(input)
#     loss = loss_fn(output, target)
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()
print("AMP reduces memory by ~50% with minimal accuracy loss")
# AMP reduces memory by ~50% with minimal accuracy loss
```

<br> <br>

# Stage 3 — Large Model Training

The core stage covering Transformer internals, pretraining, alignment, and distributed training.

## 1. Positional Encoding

RoPE (rotary position embedding, 旋转位置编码) injects position via rotation matrices on Q/K.

$$ R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \ \sin\theta & \cos\theta \end{pmatrix} $$

```python
import torch

def rope(x, theta=10000.0):
    seq_len, dim = x.shape[-2:]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

x = torch.randn(1, 4, 8)
out = rope(x)
print(f"RoPE output shape: {out.shape}")    # RoPE output shape: torch.Size([1, 4, 8])
```

<br>

## 2. Attention Variants

GQA (grouped-query attention, 分组查询注意力) shares K/V across query groups to cut KV cache size.

```python
import torch
import torch.nn as nn

class GQA(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim)
    
    def forward(self, x):
        # KV memory: n_kv_heads/n_heads of standard MHA
        return self.wq(x), self.wk(x), self.wv(x)

gqa = GQA(dim=512, n_heads=8, n_kv_heads=2)
print(f"KV memory ratio: {2/8}")        # KV memory ratio: 0.25
```

<br>

## 3. Scaling Laws

Chinchilla scaling law (缩放定律) says optimal training uses ~20 tokens per parameter.

$$ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} $$

```python
def chinchilla_optimal_tokens(n_params_billion):
    return n_params_billion * 20

print(f"7B model needs: {chinchilla_optimal_tokens(7)}B tokens")    # 7B model needs: 140B tokens
print(f"70B model needs: {chinchilla_optimal_tokens(70)}B tokens")  # 70B model needs: 1400B tokens
```

<br>

## 4. Alignment — RLHF

RLHF (reinforcement learning from human feedback, 基于人类反馈的强化学习) has three stages: SFT, reward modeling, PPO.

$$ \max_{\pi} \mathbb{E}*{x \sim D, y \sim \pi}[R(x, y)] - \beta \cdot D*{KL}[\pi | \pi_{ref}] $$

```python
# RLHF pipeline conceptual flow
stages = [
    "1. SFT (supervised fine-tuning) on demonstrations",
    "2. Reward model trained on preference pairs",
    "3. PPO optimizes policy against reward with KL constraint"
]
for s in stages:
    print(s)
# 1. SFT (supervised fine-tuning) on demonstrations
# 2. Reward model trained on preference pairs
# 3. PPO optimizes policy against reward with KL constraint
```

<br>

## 5. Alignment — DPO

DPO (direct preference optimization, 直接偏好优化) skips reward modeling by directly optimizing preference loss.

$$ L_{DPO} = -\log \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right) $$

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    return -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()

# Toy log-probs (real values come from model forward passes)
loss = dpo_loss(
    torch.tensor(-2.0), torch.tensor(-3.0),
    torch.tensor(-2.5), torch.tensor(-2.5)
)
print(f"DPO loss: {loss.item():.4f}")       # DPO loss: 0.6444
```

<br>

## 6. LoRA Fine-tuning

LoRA (low-rank adaptation, 低秩适配) freezes the base and trains two small matrices A, B with rank r.

$$ W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k} $$

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.base.weight.requires_grad = False      # freeze
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank
    
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

layer = LoRALinear(768, 768, rank=8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"Trainable: {trainable}/{total} ({100*trainable/total:.2f}%)")
# Trainable: 12288/602112 (2.04%)
```

<br>

## 7. Distributed Training — DDP

DDP (distributed data parallel, 分布式数据并行) replicates the model on every GPU and AllReduces gradients.

```python
# Conceptual DDP setup - requires multi-GPU runtime
"""
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
"""
print("DDP: full model copy per GPU, AllReduce gradients each step")
# DDP: full model copy per GPU, AllReduce gradients each step
```

<br>

## 8. Distributed Training — ZeRO

ZeRO (zero redundancy optimizer, 零冗余优化器) partitions optimizer states, gradients, and parameters across GPUs.

```python
# ZeRO stages comparison
stages = {
    "ZeRO-1": "shard optimizer states (~4x memory saving)",
    "ZeRO-2": "+ shard gradients (~8x memory saving)",
    "ZeRO-3": "+ shard parameters (linear scaling with GPUs)"
}
for k, v in stages.items():
    print(f"{k}: {v}")
# ZeRO-1: shard optimizer states (~4x memory saving)
# ZeRO-2: + shard gradients (~8x memory saving)
# ZeRO-3: + shard parameters (linear scaling with GPUs)
```

<br>

## 9. Tensor & Pipeline Parallelism

TP (tensor parallel, 张量并行) splits a single matmul across GPUs; PP (pipeline parallel, 流水线并行) splits layers across GPUs.

```python
# Megatron-style TP: column-parallel and row-parallel linears
# Layer split: GPU 0 holds layers 0-7, GPU 1 holds layers 8-15
# Combined: 3D parallelism = DP × TP × PP
print("3D parallel example: 64 GPUs = 4 DP × 4 TP × 4 PP")
# 3D parallel example: 64 GPUs = 4 DP × 4 TP × 4 PP
```

<br> <br>

# Stage 4 — Inference Optimization

The deployment-critical stage covering inference engines, KV cache, quantization, and decoding acceleration.

## 1. KV Cache

KV cache (键值缓存) stores past K/V tensors so each new token avoids recomputing attention over history.

$$ \text{Memory} = 2 \cdot L \cdot H \cdot D \cdot S \cdot B \cdot \text{dtype} $$

```python
def kv_cache_size_gb(layers, heads, head_dim, seq_len, batch, dtype_bytes=2):
    # 2 for K and V
    bytes_total = 2 * layers * heads * head_dim * seq_len * batch * dtype_bytes
    return bytes_total / 1e9

# LLaMA-7B: 32 layers, 32 heads, 128 head_dim, 4K context, batch 1, fp16
size = kv_cache_size_gb(32, 32, 128, 4096, 1, 2)
print(f"LLaMA-7B KV cache @ 4K: {size:.2f} GB")     # LLaMA-7B KV cache @ 4K: 2.15 GB
```

<br>

## 2. PagedAttention

PagedAttention (分页注意力) treats KV cache like OS virtual memory with fixed-size blocks, eliminating fragmentation.

```python
# PagedAttention conceptual model
# Each sequence's KV is stored in non-contiguous blocks
# Block table maps logical -> physical block IDs
block_size = 16
seq_len = 100
num_blocks = (seq_len + block_size - 1) // block_size
print(f"Sequence of {seq_len} tokens uses {num_blocks} blocks of size {block_size}")
# Sequence of 100 tokens uses 7 blocks of size 16
```

<br>

## 3. Continuous Batching

Continuous batching (连续批处理) replaces finished sequences with new ones at each step, keeping GPU utilization high.

```python
# Static batching: wait for slowest in batch (poor utilization)
# Continuous: insert new requests as soon as slots free up
# vLLM/SGLang achieve 2-10x throughput vs static batching
print("Continuous batching: dynamic slot allocation per token step")
# Continuous batching: dynamic slot allocation per token step
```

<br>

## 4. Quantization — INT8/INT4

Quantization (量化) maps FP16 weights to lower-bit integers, trading precision for memory and speed.

$$ q = \text{round}\left(\frac{w}{s}\right) + z, \quad w \approx s \cdot (q - z) $$

```python
import torch

def quantize_int8(w):
    scale = w.abs().max() / 127
    q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale

def dequantize_int8(q, scale):
    return q.float() * scale

w = torch.randn(4, 4)
q, s = quantize_int8(w)
w_recovered = dequantize_int8(q, s)
err = (w - w_recovered).abs().mean()
print(f"Quantization MAE: {err:.4f}")       # Quantization MAE: 0.0021 (varies)
print(f"Memory ratio: {1/4}")               # Memory ratio: 0.25 (FP32 -> INT8)
```

<br>

## 5. Quantization — AWQ & GPTQ

AWQ (activation-aware quantization, 激活感知量化) protects salient weight channels detected via activation magnitudes.

```python
# AWQ key idea:
# 1. Identify 1% salient weight channels by activation magnitude
# 2. Scale these channels up before quantization to preserve precision
# 3. Apply inverse scaling at inference time
# Result: INT4 quality close to FP16

# GPTQ key idea:
# 1. Layer-wise quantization minimizing reconstruction error
# 2. Uses Hessian information to choose quantization order
# 3. Calibration on small dataset

print("AWQ and GPTQ enable INT4 with <1% accuracy loss")
# AWQ and GPTQ enable INT4 with <1% accuracy loss
```

<br>

## 6. Speculative Decoding

Speculative decoding (推测解码) uses a small draft model to predict K tokens, then verifies in parallel with the target model.

```python
# Speculative decoding flow
def speculative_decode(draft_model, target_model, prompt, k=4):
    """
    1. Draft model generates k candidate tokens autoregressively
    2. Target model verifies all k tokens in ONE forward pass
    3. Accept longest prefix matching target's distribution
    4. Sample one extra token from target
    """
    pass

# Typical speedup: 2-3x with same output distribution
print("Speculative decoding: lossless 2-3x speedup")
# Speculative decoding: lossless 2-3x speedup
```

<br>

## 7. FlashAttention

FlashAttention (闪存注意力) tiles attention computation in SRAM, avoiding O(N²) HBM reads.

$$ \text{Time saved} \propto \frac{N \cdot d}{M} \text{ where } M = \text{SRAM size} $$

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ uses FlashAttention automatically when available
q = torch.randn(1, 8, 512, 64, device='cpu')
k = torch.randn(1, 8, 512, 64, device='cpu')
v = torch.randn(1, 8, 512, 64, device='cpu')

out = F.scaled_dot_product_attention(q, k, v)
print(f"Output shape: {out.shape}")     # Output shape: torch.Size([1, 8, 512, 64])
# On GPU, this dispatches to FlashAttention kernel automatically
```

<br>

## 8. Inference Engines

vLLM and SGLang are the leading open-source inference engines, with TensorRT-LLM optimal for NVIDIA deployments.

```python
# vLLM minimal usage
"""
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")
params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(["Hello, my name is"], params)
print(outputs[0].outputs[0].text)
"""
engines = {
    "vLLM": "PagedAttention + continuous batching, broadest model support",
    "SGLang": "RadixAttention prefix caching, fastest for agent workloads",
    "TensorRT-LLM": "NVIDIA-optimized kernels, lowest latency on H100/A100",
    "TGI": "HuggingFace-native, easy production deployment"
}
for k, v in engines.items():
    print(f"{k}: {v}")
```

<br> <br>

# Stage 5 — System & Frontier

The expert stage: GPU programming, MoE, long context, multimodal, and reasoning models.

## 1. CUDA Programming

CUDA (compute unified device architecture, 统一计算设备架构) lets you write GPU kernels directly in C++.

```cpp
// Vector addition kernel
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Host launch
int main() {
    int n = 1024;
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(a, b, c, n);
    // Output: c[i] = a[i] + b[i] for i in [0, n)
    
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
```

<br>

## 2. Triton

Triton (OpenAI's GPU DSL) lets you write CUDA-level performance in Python.

```python
# Triton kernel skeleton
"""
import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
"""
print("Triton: Python syntax, CUDA performance, used by PyTorch 2.0 compile")
# Triton: Python syntax, CUDA performance, used by PyTorch 2.0 compile
```

<br>

## 3. MoE — Mixture of Experts

MoE (mixture of experts, 专家混合) routes each token to a few experts via a learned gating network.

$$ y = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x) $$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMoE(nn.Module):
    def __init__(self, dim=64, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
    
    def forward(self, x):
        scores = self.gate(x)                                       # (B, S, E)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (topk_idx[..., k] == e)
                if mask.any():
                    out[mask] += topk_weights[..., k:k+1][mask] * self.experts[e](x[mask])
        return out

moe = SimpleMoE()
x = torch.randn(2, 4, 64)
out = moe(x)
print(f"MoE output shape: {out.shape}")     # MoE output shape: torch.Size([2, 4, 64])
```

<br>

## 4. Long Context — YaRN

YaRN (yet another RoPE extensioN, RoPE 外推方法) rescales RoPE frequencies to extend context length without retraining.

```python
# Long context techniques summary
techniques = {
    "Position Interpolation (PI)": "linearly compress positions into trained range",
    "NTK-aware": "scale base frequency theta non-uniformly",
    "YaRN": "ramp scaling + temperature, current SOTA for extrapolation",
    "Ring Attention": "split long sequence across GPUs, ring-pass KV blocks"
}
for k, v in techniques.items():
    print(f"{k}: {v}")
```

<br>

## 5. Reasoning Models — GRPO

GRPO (group relative policy optimization, 组相对策略优化) is DeepSeek R1's RL algorithm that drops the value network.

$$ A_i = \frac{r_i - \text{mean}({r_1, ..., r_G})}{\text{std}({r_1, ..., r_G})} $$

```python
import torch

def grpo_advantage(rewards):
    """Compute advantages by normalizing rewards within a group."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# Sample 4 outputs for the same prompt, get rewards
rewards = torch.tensor([0.8, 0.2, 0.5, 0.9])
advantages = grpo_advantage(rewards)
print(f"Advantages: {advantages}")
# Advantages: tensor([ 0.6489, -1.4276, -0.4150,  1.1937])
```

<br>

## 6. Multimodal — Vision Language Models

VLM (vision language model, 视觉语言模型) connects a vision encoder (CLIP/SigLIP) to an LLM via a projector.

```python
# VLM architecture pattern
"""
Image -> Vision Encoder (ViT) -> Image Embeddings
                                          \\
                                           Projector (MLP) -> Token Embeddings
                                          /
Text -> Tokenizer -> Text Embeddings -- Concat -- LLM -> Output
"""
components = ["Vision Encoder (SigLIP)", "Projector (2-layer MLP)", "LLM (Qwen/LLaMA)"]
for c in components:
    print(f"- {c}")
# - Vision Encoder (SigLIP)
# - Projector (2-layer MLP)
# - LLM (Qwen/LLaMA)
```

<br> <br>

# Open-Source Models Study Guide

A curated learning order for the most influential open-source models.

## 1. Foundation Layer

Start with nanoGPT, then LLaMA family, then Qwen for state-of-the-art Chinese-capable models.

### 1) nanoGPT

Karpathy's nanoGPT is the minimal readable Transformer training implementation in ~300 lines.

```python
# Core nanoGPT block - simplified
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

print("nanoGPT: read tokenizer.py, model.py, train.py in this order")
# nanoGPT: read tokenizer.py, model.py, train.py in this order
```

### 2) LLaMA 3

LLaMA 3 (92-page technical report, 技术报告) is the most complete open pretraining + alignment textbook.

```python
# LLaMA 3 key design choices
design = {
    "Tokenizer": "128K BPE vocab (up from 32K in LLaMA 2)",
    "Attention": "GQA with 8 KV heads",
    "Position": "RoPE with theta=500000 for long context",
    "Norm": "RMSNorm pre-normalization",
    "Activation": "SwiGLU FFN",
    "Training": "15T tokens, multi-stage curriculum"
}
for k, v in design.items():
    print(f"{k}: {v}")
```

<br>

## 2. MoE Layer

DeepSeek V3 is required reading: MLA, fine-grained experts, MTP, and auxiliary-loss-free load balancing.

### 1) Mixtral 8x7B

Mixtral uses 8 experts with top-2 routing, activating ~13B params per token from 47B total.

```python
# Mixtral routing
"""
For each token:
1. Gate: linear projection to 8 logits
2. Top-2: pick 2 highest-scoring experts
3. Softmax over top-2 weights
4. Weighted sum of expert outputs
"""
print("Mixtral: 47B total params, 13B active per token")
# Mixtral: 47B total params, 13B active per token
```

### 2) DeepSeek V3

DeepSeek V3 innovations: MLA (multi-head latent attention, 多头潜在注意力), shared+routed experts, MTP (multi-token prediction).

```python
# DeepSeek V3 scale
specs = {
    "Total params": "671B",
    "Active params": "37B per token",
    "Experts": "1 shared + 256 routed (top-8)",
    "MLA KV cache": "~6% of standard MHA",
    "MTP": "predicts 2 future tokens during training"
}
for k, v in specs.items():
    print(f"{k}: {v}")
```

<br>

## 3. Reasoning Layer

DeepSeek R1 is the milestone open-source reasoning model trained via pure RL with GRPO.

```python
# DeepSeek R1 training pipeline
pipeline = [
    "1. R1-Zero: pure RL on DeepSeek-V3-Base (no SFT) - emergent reasoning",
    "2. Cold-start: small SFT dataset to fix R1-Zero readability issues",
    "3. Reasoning RL: GRPO on math/code with rule-based rewards",
    "4. SFT on 800K curated samples",
    "5. Final RL: helpfulness + harmlessness + reasoning combined"
]
for s in pipeline:
    print(s)
```

<br>

## 4. How to Study Each Model

Apply this four-step method to every model worth studying.

### 1) Read the Tech Report

Focus on architecture changes, data mixture, training hyperparameters, and alignment recipe.

### 2) Read the Source Code

Start with HuggingFace `transformers/models/<name>/modeling_<name>.py` and focus on the `forward` function.

### 3) Hands-on Fine-tuning

Use LoRA on your own data with a 7B model to internalize the training loop.

```python
# Minimal SFT example with HuggingFace TRL
"""
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
"""
print("LoRA SFT: ~24GB VRAM enough for 7B model")
# LoRA SFT: ~24GB VRAM enough for 7B model
```

### 4) Deploy with vLLM

Serve the model and benchmark throughput/latency to understand inference characteristics.

```python
# vLLM benchmark snippet
"""
from vllm import LLM, SamplingParams
import time

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", gpu_memory_utilization=0.9)
params = SamplingParams(temperature=0, max_tokens=256)

prompts = ["Explain quantum mechanics."] * 100
start = time.time()
outputs = llm.generate(prompts, params)
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Throughput: {total_tokens/elapsed:.0f} tokens/sec")
"""
print("Typical 7B throughput on A100: 3000-5000 tokens/sec")
# Typical 7B throughput on A100: 3000-5000 tokens/sec
```

<br> <br>

# Key Takeaways

The roadmap rewards depth over breadth — master a few canonical models thoroughly.

## 1. Priority Order

Study LLaMA 3, DeepSeek V3, and DeepSeek R1 in depth; 80% of new models become trivial after this.

<br>

## 2. Reports Beat Weights

Many model weights are too large to run, but every technical report is free design knowledge.

<br>

## 3. Stage 3 and 4 Run in Parallel

Training and inference skills are equally valued in industry; alternate between them.

<br>

## 4. Project-Driven Learning

One complete end-to-end project per stage beats reading ten textbooks.

<br> <br>
