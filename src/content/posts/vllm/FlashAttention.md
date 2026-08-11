---
title: "FlashAttention"
published: 2026-08-09
description: "FlashAttention"
image: ""
tags: ["vllm","FlashAttention"]
category: vllm
draft: false
lang: ""
createdAt: "2026-08-09T21:40:04.328.815708170Z"
---

# FlashAttention for LLM Inference/Infra —— 5 天速成 + 代码实战

>   目标：不啃 CUDA kernel 源码，只学到能看懂"为什么 vLLM/SGLang 都用 FlashAttention/FlashInfer 做 attention 后端"、以及 `flash_attn_varlen_func` 这些 API 参数是什么意思。 位置感：FlashAttention 跟你之前学的东西是同一条链路上的邻居——

```text
Kubernetes / Ray     ← 调度：Pod/Actor 放在哪
NCCL                  ← 通信：多卡之间怎么传数据
FlashAttention        ← 计算：单卡上，attention 这一步怎么算得快、省显存   ← 这一份
```

前三份关心"多卡怎么协作"，这一份关心"单张卡上最贵的那个算子（attention）怎么优化"。它不解决通信问题，解决的是**显存带宽**问题。

------

## Day 0：环境准备 + 一句话理解 FlashAttention

```bash
# 需要真实 GPU（Ampere 架构以上，比如 A100/A10/4090/H100）

conda create -n flashattn python=3.11 -y
conda activate flashattn

conda install cuda -c nvidia/label/cuda-13.0.0

pip install torch
pip install -U setuptools wheel packaging psutil ninja numpy
pip install flash-attn --no-build-isolation




python -c "import flash_attn; print(flash_attn.__version__)"
```

**一句话理解**：标准 attention 实现要先算出完整的 `QK^T` 矩阵（形状是 `[seq_len, seq_len]`）存到显存里，再做 softmax，再乘 `V`。这个中间矩阵在长序列下巨大，而且要在显存（HBM）和计算单元之间来回搬运，**真正的瓶颈不是算力，是显存带宽**。FlashAttention 的核心思路是：**分块计算，never materialize 那个完整的中间矩阵**，把数据放在 GPU 的高速片上内存（SRAM）里算完就扔，只往显存写最终结果。

------

## Day 1：先跑通"慢但直观"的标准 Attention，理解它为什么慢

### 1.1 标准（naive）实现

```python
# naive_attention.py
import torch
import math

def naive_attention(q, k, v, causal=False):
    """
    q, k, v: [batch, num_heads, seq_len, head_dim]
    这是最直白的实现：完整算出 attention matrix，显存开销 O(seq_len^2)
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)   # [B, H, seq, seq] —— 这就是那个巨大的中间矩阵

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)       # 又是一次 O(seq^2) 的显存读写
    output = attn_weights @ v
    return output

# 测试
B, H, S, D = 1, 8, 2048, 64
q = torch.randn(B, H, S, D, device="cuda")
k = torch.randn(B, H, S, D, device="cuda")
v = torch.randn(B, H, S, D, device="cuda")

torch.cuda.reset_peak_memory_stats()
out = naive_attention(q, k, v, causal=True)
print("naive attention 峰值显存 (MB):", torch.cuda.max_memory_allocated() / 1024**2)
print("中间矩阵 scores 大小 (MB):", B*H*S*S*4 / 1024**2)   # float32 每个元素 4 字节
python naive_attention.py
# seq_len=2048 时，光是那个 attention matrix 就要 128MB (单个 head)
# seq_len 翻倍到 4096，这个矩阵直接变成 512MB —— O(n^2) 增长，这就是长上下文推理的显存噩梦
```

### 1.2 显存开销随序列长度的增长（直观感受 O(n²)）

```python
# memory_scaling.py
import torch

for seq_len in [512, 1024, 2048, 4096, 8192]:
    size_mb = 1 * 8 * seq_len * seq_len * 4 / 1024**2   # B=1, H=8, float32
    print(f"seq_len={seq_len:6d}  attention matrix ≈ {size_mb:10.1f} MB")
seq_len=   512  attention matrix ≈        8.0 MB
seq_len=  1024  attention matrix ≈       32.0 MB
seq_len=  2048  attention matrix ≈      128.0 MB
seq_len=  4096  attention matrix ≈      512.0 MB
seq_len=  8192  attention matrix ≈     2048.0 MB
```

这就是为什么长上下文推理（32K、128K context）如果用 naive attention 根本跑不起来——这一份要解决的就是这个问题。

------

## Day 2：FlashAttention 的核心技巧——分块 + 在线 Softmax

不用真的去实现 CUDA kernel，用纯 PyTorch 写一个"教学版 FlashAttention"就能看懂算法思路（真实 kernel 是用 CUDA 手写在 SRAM 里做的，这里只是复现数学逻辑）。

### 2.1 关键难点：Softmax 是"全局"的，怎么分块算？

标准 softmax 需要看到一整行的所有分数才能算分母（归一化项）。FlashAttention 用**在线 softmax（online softmax）**技巧：每处理一个新的分块，就用一个数学上等价的公式更新之前累积的结果，不需要等看完整行。

```python
# flash_attention_educational.py —— 教学版实现，展示分块 + 在线 softmax 的逻辑
# 注意：这是纯 PyTorch 写的，速度不会比 naive 快，只是为了看懂算法，不是真实优化实现
import torch
import math

def flash_attention_educational(q, k, v, block_size=128, causal=False):
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)

    output = torch.zeros_like(q)
    row_max = torch.full((B, H, S, 1), float("-inf"), device=q.device)   # 每行当前见过的最大值
    row_sum = torch.zeros((B, H, S, 1), device=q.device)                  # 每行当前的 softmax 分母（在线累积）

    num_blocks = (S + block_size - 1) // block_size

    for j in range(num_blocks):     # 外层循环：遍历 K/V 的每一个分块（对应 kernel 里的"外循环加载 K,V 到 SRAM"）
        k_start, k_end = j * block_size, min((j + 1) * block_size, S)
        k_block = k[:, :, k_start:k_end, :]
        v_block = v[:, :, k_start:k_end, :]

        # 当前分块的局部分数
        scores = q @ k_block.transpose(-2, -1) * scale   # [B, H, S, block_size]

        if causal:
            q_idx = torch.arange(S, device=q.device).unsqueeze(1)
            k_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
            mask = q_idx < k_idx
            scores = scores.masked_fill(mask, float("-inf"))

        block_max = scores.max(dim=-1, keepdim=True).values           # 这个分块内的局部最大值
        new_max = torch.maximum(row_max, block_max)                    # 更新全局最大值（数值稳定性用，防止 exp 溢出）

        # 在线 softmax 的核心公式：用新旧最大值的差值，重新缩放之前累积的结果
        exp_scores = torch.exp(scores - new_max)
        correction = torch.exp(row_max - new_max)                      # 修正之前累积的分母和输出

        row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
        output = output * correction + exp_scores @ v_block             # 累积输出，还没除以分母

        row_max = new_max

    output = output / row_sum      # 最后统一除以累积的分母，得到最终结果
    return output


# 验证正确性：跟 naive attention 结果应该一致（在数值精度范围内）
from naive_attention import naive_attention   # 复用 Day1 的实现

B, H, S, D = 1, 4, 1024, 64
q = torch.randn(B, H, S, D, device="cuda")
k = torch.randn(B, H, S, D, device="cuda")
v = torch.randn(B, H, S, D, device="cuda")

out_naive = naive_attention(q, k, v, causal=True)
out_flash = flash_attention_educational(q, k, v, block_size=128, causal=True)

print("最大误差:", (out_naive - out_flash).abs().max().item())   # 应该是 1e-5 量级的浮点误差
python flash_attention_educational.py
# 输出应该显示误差极小，证明"分块 + 在线 softmax"这套数学是等价的
```

**这就是 FlashAttention 论文的核心贡献**：证明了 softmax 可以增量式地分块计算，从而让整个 attention 计算可以在 GPU 的 SRAM（比 HBM 快一个数量级，但容量小得多）里完成，全程不用把 `[seq, seq]` 那个大矩阵写回显存。真实的 CUDA kernel 把上面这套逻辑用 tiling + shared memory 手写实现，还做了大量算子融合（QK^T、mask、softmax、乘V 全部融合成一个 kernel），所以比 naive 实现快好几倍、显存从 O(n²) 降到 O(n)。

------

## Day 3：用真实的 `flash-attn` 库

教学版只是帮你理解算法，生产用的是官方 CUDA 实现：

```python
# real_flash_attn.py
import torch
from flash_attn import flash_attn_func

B, H, S, D = 1, 8, 4096, 64
# 注意：flash_attn_func 要求的输入形状是 [batch, seq_len, num_heads, head_dim]（跟 naive 实现的维度顺序不一样，容易踩坑）
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

torch.cuda.reset_peak_memory_stats()
out = flash_attn_func(q, k, v, causal=True)
print("flash-attn 峰值显存 (MB):", torch.cuda.max_memory_allocated() / 1024**2)
print("output shape:", out.shape)
python real_flash_attn.py
```

对比同样 seq_len=4096 下 naive 实现和 flash-attn 的峰值显存，差距会非常明显——这就是为什么长上下文模型能跑起来。

### 3.1 推理场景专用：变长序列（varlen）版本

推理服务里同一个 batch 里各请求的 prompt 长度经常不一样（不像训练时会 pad 到统一长度），FlashAttention 提供了专门处理变长序列的 API，这也是 vLLM/SGLang 实际调用的接口：

```python
# varlen_flash_attn.py
import torch
from flash_attn import flash_attn_varlen_func

# 模拟一个 batch 里有 3 个请求，长度分别是 100, 250, 80（不用 pad，直接拼成一条长序列）
seqlens = [100, 250, 80]
total_tokens = sum(seqlens)
H, D = 8, 64

q = torch.randn(total_tokens, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(total_tokens, H, D, device="cuda", dtype=torch.float16)
v = torch.randn(total_tokens, H, D, device="cuda", dtype=torch.float16)

# cu_seqlens：累积长度数组，用来告诉 kernel 每条序列的边界在哪
# 例如 [0, 100, 350, 430] 表示第一条是 [0:100)，第二条是 [100:350)，第三条是 [350:430)
cu_seqlens = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(seqlens), dim=0).tolist()),
    dtype=torch.int32, device="cuda",
)
max_seqlen = max(seqlens)

out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
print("varlen output shape:", out.shape)   # [total_tokens, H, D]
```

**这就是你在 vLLM/SGLang 源码里看到 `cu_seqlens` 这个变量名时应该联想到的东西**：它是"把一个 batch 里长短不一的多条序列拼成一条长序列后，标记每条序列边界"的数组，避免了 padding 浪费算力。

------

## Day 4：推理场景的关键变体——KV Cache 与 Paged Attention

训练时 Q、K、V 都是新算出来的；**推理的 decode 阶段每次只有 1 个新 token 的 Q，但要跟之前所有 token 的 K/V 做 attention**——这些历史 K/V 就是 KV cache。

### 4.1 decode 阶段的 attention 长什么样

```python
# decode_attention_concept.py —— 概念示意：decode 阶段 Q 只有 1 个 token，K/V 是完整历史
import torch
from flash_attn import flash_attn_with_kvcache

B, H, D = 1, 8, 64
max_seq_len = 4096
current_len = 500          # 已经生成了 500 个 token

# KV cache：预先分配好的大 buffer，随着生成不断往里写新的 K/V
k_cache = torch.randn(B, max_seq_len, H, D, device="cuda", dtype=torch.float16)
v_cache = torch.randn(B, max_seq_len, H, D, device="cuda", dtype=torch.float16)

# 新的一步只有 1 个 token 的 Q
q_new = torch.randn(B, 1, H, D, device="cuda", dtype=torch.float16)
k_new = torch.randn(B, 1, H, D, device="cuda", dtype=torch.float16)   # 新 token 自己的 K
v_new = torch.randn(B, 1, H, D, device="cuda", dtype=torch.float16)

out = flash_attn_with_kvcache(
    q_new,
    k_cache, v_cache,          # 完整的历史 cache（函数内部会把 k_new/v_new 写进 cache 里对应位置）
    k=k_new, v=v_new,
    cache_seqlens=torch.tensor([current_len], device="cuda", dtype=torch.int32),
)
print("decode step output shape:", out.shape)   # [B, 1, H, D]
```

### 4.2 为什么 vLLM 又搞了个 PagedAttention（跟 FlashAttention 是什么关系）

FlashAttention 解决的是"attention 计算本身怎么快、怎么省显存"；**PagedAttention（vLLM 提出）解决的是另一个正交问题：KV cache 怎么在显存里管理，避免碎片和浪费**。

```text
FlashAttention  → 优化"怎么算"（计算 kernel 层面，省的是计算过程中的显存带宽）
PagedAttention  → 优化"怎么存"（内存管理层面，省的是 KV cache 本身占用的显存）
```

两者不冲突，是互补的——vLLM/SGLang 实际用的是"分页存储的 KV cache" + "FlashAttention/FlashInfer 风格的分块计算 kernel"组合在一起（比如 `flash-attn` 后来也加了对 paged KV cache layout 的支持，`flash_attn_with_kvcache` 也能接收分页的 cache）。

------

## Day 5：接回 vLLM / SGLang，知道该看哪些地方

### 5.1 vLLM 里怎么选 attention backend

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    # vLLM 会自动探测硬件选最优 backend，也可以强制指定
)
# 用环境变量强制指定 attention backend，排障/对比性能时常用
export VLLM_ATTENTION_BACKEND=FLASH_ATTN     # 或 FLASHINFER / XFORMERS，取决于 vLLM 版本支持情况
vllm serve Qwen/Qwen2.5-7B-Instruct
```

### 5.2 源码里该找哪些关键词

读 vLLM/SGLang 的 attention 相关代码时，按这个顺序找会比较快：

```text
1. 找 "attention_backend" 或 "AttentionBackend" 这个抽象类
      → 理解它怎么在 FlashAttention / FlashInfer / XFormers 之间切换
2. 找调用 flash_attn_varlen_func / flash_attn_with_kvcache 的地方
      → 对照 Day3、Day4 的参数（cu_seqlens, cache_seqlens）理解在传什么
3. 找 KV cache 的分配逻辑（通常叫 kv_cache_manager / block_manager）
      → 这是 PagedAttention 那一层，跟 FlashAttention 是两码事，别混在一起看
4. 如果是 SGLang，额外找 RadixAttention 相关代码
      → 这是在 PagedAttention 基础上加了"前缀复用"的调度逻辑，同样是正交的优化
```

### 5.3 一个简单的 benchmark，直观感受差距

```python
# benchmark_flash_vs_naive.py
import torch, time
from flash_attn import flash_attn_func
from naive_attention import naive_attention

B, H, S, D = 1, 8, 4096, 64

def bench(fn, *args, warmup=3, iters=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000   # ms

q_naive = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
k_naive = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
v_naive = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

q_fa = q_naive.transpose(1, 2).contiguous()   # flash_attn_func 要 [B,S,H,D]
k_fa = k_naive.transpose(1, 2).contiguous()
v_fa = v_naive.transpose(1, 2).contiguous()

t_naive = bench(lambda: naive_attention(q_naive, k_naive, v_naive, causal=True))
t_flash = bench(lambda: flash_attn_func(q_fa, k_fa, v_fa, causal=True))

print(f"naive attention:  {t_naive:.2f} ms")
print(f"flash attention:  {t_flash:.2f} ms")
print(f"speedup: {t_naive / t_flash:.1f}x")
python benchmark_flash_vs_naive.py
# seq_len=4096 时通常能看到几倍的速度差，序列越长差距越大
```

------

## 参考资料

-   FlashAttention 论文（v1）：https://arxiv.org/abs/2205.14135
-   FlashAttention-2 论文：https://arxiv.org/abs/2307.08691
-   FlashAttention GitHub：https://github.com/Dao-AILab/flash-attention
-   PagedAttention / vLLM 论文：https://arxiv.org/abs/2309.06180
-   FlashInfer（vLLM/SGLang 现在也常用的另一套推理 attention kernel 库）：https://github.com/flashinfer-ai/flashinfer
-   SGLang RadixAttention 论文：https://arxiv.org/abs/2312.07104
