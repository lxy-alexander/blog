---
title: "FlashAttention V1"
published: 2026-03-09
description: "FlashAttention V1"
image: ""
tags: ["llm","FlashAttention V1"]
category: llm
draft: false
lang: ""
---

# **I. FlashAttention v1 — A Complete Learning Manual**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> FlashAttention v1 is an <strong>IO-aware exact attention algorithm</strong> (IO感知的精确注意力算法) that rewrites the standard Scaled Dot-Product Attention (缩放点积注意力) to be <em>memory-efficient and fast</em> by exploiting GPU memory hierarchy (GPU内存层次结构). Rather than materializing the full N×N attention matrix in High Bandwidth Memory (高带宽内存, HBM), FlashAttention tiles the computation across SRAM (静态随机存取存储器) on-chip blocks — reducing HBM reads/writes from <strong>O(N²)</strong> to <strong>O(N)</strong> and achieving wall-clock speedups of 2–4× on modern GPUs. </div>

------

## 1. Background (背景)

### 1) The Transformer Bottleneck (Transformer瓶颈)

The <span style="color:#2980B9">core operation</span> of every Transformer (变换器) is:

$$ \text{Attention}(Q, K, V) = \text{softmax}!\left(\frac{QK^\top}{\sqrt{d}}\right)V $$

where <span style="color:#E8600A;font-weight:700">Q, K, V ∈ ℝ^{N×d}</span> are the Query (查询矩阵), Key (键矩阵), and Value (值矩阵), N is the Sequence Length (序列长度), and d is the Head Dimension (头维度).

| Step | Operation               | Memory Footprint (内存占用) |
| ---- | ----------------------- | --------------------------- |
| ①    | Compute S = QKᵀ / √d    | O(N²) — the **bottleneck**  |
| ②    | P = softmax(S) row-wise | O(N²)                       |
| ③    | O = PV                  | O(Nd)                       |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> For N = 16 384 (a 16 K context), the attention matrix S alone requires <strong>16 384² × 4 bytes ≈ 1 GB</strong> of HBM per head. Modern LLMs (大语言模型) stack 32–96 heads, making vanilla attention prohibitively expensive.</div>

### 2) GPU Memory Hierarchy (GPU内存层次结构)

```
┌─────────────────────────────────────────────────────────────┐
│  GPU Die                                                     │
│  ┌──────────────────────────────────┐                       │
│  │  SM (Streaming Multiprocessor)   │  ← Execution Unit     │
│  │  ┌──────────────┐               │                       │
│  │  │  Registers   │  ~256 KB/SM   │  ← Fastest            │
│  │  ├──────────────┤               │                       │
│  │  │  SRAM/L1     │  ~192 KB/SM   │  ← ~19 TB/s bandwidth │
│  │  └──────────────┘               │                       │
│  └──────────────────────────────────┘                       │
│                                                             │
│  ┌──────────────────────────────────┐                       │
│  │  HBM (High Bandwidth Memory)     │  ~40–80 GB           │
│  │                                  │  ~2 TB/s bandwidth    │
│  │  Stores: Q, K, V, O, S, P …      │  ← 10× slower SRAM   │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

<span style="color:#2980B9">Key insight:</span> Standard attention is **memory-bandwidth bound** (内存带宽瓶颈), not compute-bound. Arithmetic Intensity (算术强度) of vanilla attention is very low — most time is spent moving data, not computing.

------

## 2. Motivation (动机)

### 1) Why Standard Attention Is Slow

<span style="color:#E8600A">1.</span> **HBM round-trips**: Each matrix S and P must be written to and read back from HBM.
 <span style="color:#E8600A">2.</span> **Memory wall**: N² growth means 4× longer sequences = 16× more memory.
 <span style="color:#E8600A">3.</span> **Approximate methods fail**: Sparse / Low-rank approximations (稀疏/低秩近似) sacrifice accuracy and are hard to implement efficiently.

### 2) The FlashAttention Insight

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Goal:</strong> Compute exact softmax attention <em>without ever materializing the full N×N matrix in HBM</em>.<br> <strong>Strategy:</strong> Tile (分块) the Q, K, V matrices into blocks that fit in SRAM, compute a numerically stable running softmax (在线softmax) on-the-fly, and accumulate the output O — all in one fused kernel (融合核函数). </div>

------

## 3. Mathematical Foundations (数学基础)

### 1) Online Softmax (在线Softmax) — The Key Trick

Standard softmax for a row vector **x** ∈ ℝ^N:

$$ \text{softmax}(x)*i = \frac{e^{x_i}}{\sum*{j=1}^{N} e^{x_j}} $$

<span style="color:#C0392B;font-weight:600">Pitfall: Computing this naively requires two passes — one to find max for numerical stability, one for the exponentials. We can't do two passes if we never store the full row.</span>

**Solution — 3-variable online update** across tiles:

For each new tile of scores, maintain a running tuple <span style="color:#E8600A;font-weight:700">(m, ℓ, O)</span>:

| Variable                                                     | Meaning                             | Update Rule                                              |
| ------------------------------------------------------------ | ----------------------------------- | -------------------------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">m</code> | Running row-max (行最大值)          | m_new = max(m_old, tile_max)                             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ℓ</code> | Running normalizer (累积归一化因子) | ℓ_new = e^(m_old−m_new)·ℓ_old + Σ e^(x_j−m_new)          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">O</code> | Running weighted output (累积输出)  | O_new = e^(m_old−m_new)·O_old + exp(S_tile−m_new)·V_tile |

After processing all tiles, the final output is:

$$ O_{\text{final}} = \frac{O}{\ell} $$

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> This is mathematically equivalent to the standard softmax — it is <strong>exact</strong>, not approximate. The correction factor <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">e^(m_old − m_new)</code> rescales previously accumulated values when a new, larger maximum is discovered.</div>

### 2) Block Size Analysis (块大小分析)

For SRAM of size <span style="color:#E8600A;font-weight:700">M</span> bytes, and vectors of dimension <span style="color:#E8600A;font-weight:700">d</span>:

$$ B_c = \left\lfloor \frac{M}{4d} \right\rfloor \quad \text{(Key/Value block size)} $$ $$ B_r = \min!\left(\left\lfloor \frac{M}{4d} \right\rfloor,\ d\right) \quad \text{(Query block size)} $$

Number of blocks: <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">T_r = ⌈N/B_r⌉</code> query blocks, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">T_c = ⌈N/B_c⌉</code> KV blocks.

------

## 4. Algorithm Walkthrough (算法详解)

### 1) Visual Overview (可视化概览)

```
HBM (High Bandwidth Memory)                SRAM (On-chip)
┌─────────────────────────┐               ┌──────────────────┐
│  Q  [N × d]             │──── load ────▶│  Q_i  [Br × d]   │
│  K  [N × d]             │──── load ────▶│  K_j  [Bc × d]   │
│  V  [N × d]             │──── load ────▶│  V_j  [Bc × d]   │
│                         │               │                  │
│  O  [N × d]  (output)   │◀─── write ───│  O_i  [Br × d]   │
│  ℓ  [N]      (norm)     │◀─── write ───│  ℓ_i  [Br]       │
│  m  [N]      (row-max)  │◀─── write ───│  m_i  [Br]       │
└─────────────────────────┘               └──────────────────┘

Outer loop (i = 1 … T_r):  iterate over Q blocks  ← parallelized across GPU SMs
  Inner loop (j = 1 … T_c): iterate over K/V blocks
        ┌──────────────────────────────────────────┐
        │  1. S_ij  = Q_i · K_j^T / √d            │
        │  2. m̃_ij = rowmax(S_ij)                 │
        │  3. P̃_ij = exp(S_ij − m̃_ij)            │
        │  4. ℓ̃_ij = rowsum(P̃_ij)                │
        │  5. Update (m_i, ℓ_i, O_i) online       │
        └──────────────────────────────────────────┘
  Write O_i = O_i / ℓ_i  back to HBM
```

### 2) Full Algorithm (完整算法) — Step by Step

**Inputs:** Q, K, V ∈ ℝ^{N×d} in HBM. SRAM size M.

**Step 0 — Initialize:**

-   Divide Q into T_r blocks {Q₁, …, Q_{T_r}}, each of size B_r × d
-   Divide K, V into T_c blocks {K₁, …, K_{T_c}}, {V₁, …, V_{T_c}}, each B_c × d
-   Initialize O = (0), ℓ = (0), m = (−∞) in HBM

**Step 1 — Outer loop over query blocks** <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">for i = 1 to T_r</code>:

1.  Load Q_i, O_i, ℓ_i, m_i from HBM into SRAM

**Step 2 — Inner loop over KV blocks** <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">for j = 1 to T_c</code>: 2. Load K_j, V_j from HBM into SRAM 3. Compute <span style="color:#E8600A;font-weight:700">S_ij = Q_i K_jᵀ / √d</span> ∈ ℝ^{B_r × B_c} (on-chip) 4. Compute tile statistics: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">m̃ = rowmax(S_ij)</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">P̃ = exp(S_ij − m̃)</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ℓ̃ = rowsum(P̃)</code> 5. Online update (numerically stable):

$$ m_i^{\text{new}} = \max(m_i,\ \tilde{m}),\quad \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{\tilde{m} - m_i^{\text{new}}} \tilde{\ell} $$

$$ O_i^{\text{new}} = \text{diag}(e^{m_i - m_i^{\text{new}}})^{-1} O_i + e^{\tilde{m} - m_i^{\text{new}}} \tilde{P} V_j $$

**Step 3 — Write back:** 6. Compute <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">O_i = diag(ℓ_i)⁻¹ O_i</code> and write O_i, ℓ_i, m_i to HBM

**Output:** O ∈ ℝ^{N×d}

### 3) Complexity Analysis (复杂度分析)

| Metric                      | Standard Attention | FlashAttention v1                                            |
| --------------------------- | ------------------ | ------------------------------------------------------------ |
| HBM reads/writes (IO复杂度) | O(Nd + N²)         | <span style="color:#E8600A;font-weight:700">O(N² d² / M)</span> |
| Memory footprint (内存占用) | O(N²)              | <span style="color:#E8600A;font-weight:700">O(N)</span>      |
| Compute FLOPs (计算量)      | O(N²d)             | O(N²d) — same                                                |
| Speed (wall-clock)          | 1×                 | **2–4× faster**                                              |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Since d ≪ M (typical: d = 64, M ~ 100 KB as scalars), the IO complexity <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">O(N²d²/M)</code> is <em>strictly less</em> than O(N²). The authors prove this is optimal up to log factors for any algorithm that avoids materializing the N×N matrix.</div>

------

## 5. Backward Pass (反向传播)

### 1) Recomputation Strategy (重计算策略)

Standard attention saves P (N×N) for the backward pass — <span style="color:#C0392B;font-weight:600">this is what consumes O(N²) memory</span>.

FlashAttention v1 avoids storing P by **recomputing** it during the backward pass from the saved {O, ℓ, m} (only O(N) storage):

```
Backward Input:  dO  [N × d]   (gradient of output)
Saved from fwd:  O   [N × d],  ℓ  [N],  m  [N]

For each block:
  1. Reload Q_i, K_j, V_j from HBM  ← pay HBM read cost again
  2. Recompute S_ij, P_ij on-chip
  3. Compute dV_j += P_ijᵀ · dO_i
  4. Compute dP_ij = dO_i · V_jᵀ
  5. Compute dS_ij = P_ij ⊙ (dP_ij − (dO_i ⊙ O_i)·1ᵀ)
  6. Accumulate dQ_i, dK_j
```

<span style="color:#2980B9">Trade-off:</span> More FLOPs (recomputation overhead ~33%) but dramatically less memory.

------

## 6. Code Implementation (代码实现)

### 1) Minimal Python Reference (纯Python参考实现)

```python
import torch
import math

def flash_attention_v1_reference(Q, K, V, block_size=64):
    """
    Pure-Python FlashAttention v1 reference.
    Q, K, V: [N, d]  (single head, batch=1 for clarity)
    Returns: O [N, d]
    """
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # Output accumulator & online softmax statistics
    O = torch.zeros(N, d, dtype=Q.dtype, device=Q.device)
    ell = torch.zeros(N,    dtype=Q.dtype, device=Q.device)   # running normalizer
    m   = torch.full((N,), float('-inf'),
                     dtype=Q.dtype, device=Q.device)          # running row-max

    Br = block_size   # query block size
    Bc = block_size   # key/value block size
    Tr = math.ceil(N / Br)
    Tc = math.ceil(N / Bc)

    for i in range(Tr):                      # outer: query blocks
        qi_start = i * Br
        qi_end   = min(qi_start + Br, N)
        Qi = Q[qi_start:qi_end]              # [Br, d]
        Oi = O[qi_start:qi_end]              # view
        li = ell[qi_start:qi_end]
        mi = m[qi_start:qi_end]

        for j in range(Tc):                  # inner: KV blocks
            kj_start = j * Bc
            kj_end   = min(kj_start + Bc, N)
            Kj = K[kj_start:kj_end]          # [Bc, d]
            Vj = V[kj_start:kj_end]          # [Bc, d]

            # Step 1: tile attention scores
            Sij = scale * (Qi @ Kj.T)        # [Br, Bc]

            # Step 2: tile statistics
            m_tilde = Sij.max(dim=-1).values  # [Br]
            P_tilde = torch.exp(Sij - m_tilde.unsqueeze(-1))  # [Br, Bc]
            l_tilde = P_tilde.sum(dim=-1)     # [Br]

            # Step 3: online update (numerically stable rescaling)
            m_new = torch.maximum(mi, m_tilde)
            alpha  = torch.exp(mi - m_new)    # rescale old
            beta   = torch.exp(m_tilde - m_new)  # rescale new

            li_new = alpha * li + beta * l_tilde
            Oi_new = (alpha.unsqueeze(-1) * Oi
                      + beta.unsqueeze(-1) * (P_tilde @ Vj))

            # Update in-place
            O[qi_start:qi_end] = Oi_new
            ell[qi_start:qi_end] = li_new
            m[qi_start:qi_end]   = m_new
            Oi = Oi_new
            li = li_new
            mi = m_new

        # Normalize output block
        O[qi_start:qi_end] = O[qi_start:qi_end] / ell[qi_start:qi_end].unsqueeze(-1)

    return O


# ──── Correctness check ───────────────────────────────────────
def standard_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = scale * (Q @ K.T)
    P = torch.softmax(S, dim=-1)
    return P @ V

if __name__ == "__main__":
    torch.manual_seed(42)
    N, d = 256, 64
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)

    out_flash = flash_attention_v1_reference(Q, K, V, block_size=32)
    out_std   = standard_attention(Q, K, V)

    print(f"Max diff: {(out_flash - out_std).abs().max():.2e}")
    # Expected: < 1e-5  (numerical precision differences only)
```

### 2) Triton Kernel Sketch (Triton GPU核函数草图)

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    N, d,
    BLOCK_M: tl.constexpr,   # Br — query block size
    BLOCK_N: tl.constexpr,   # Bc — KV block size
    SCALE:   tl.constexpr,
):
    # Each program handles one query block (row-block of Q)
    start_m = tl.program_id(0)

    # Pointers to this query block
    offs_m  = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_N)

    # Load Q block into SRAM registers
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qn
                       + offs_d[None, :] * stride_qd)  # [Bm, d]

    # Initialize accumulators
    m_i   = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i   = tl.zeros([BLOCK_M],              dtype=tl.float32)
    acc   = tl.zeros([BLOCK_M, BLOCK_N],     dtype=tl.float32)

    # Inner loop over KV blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K, V tiles
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn
                           + offs_d[None, :] * stride_kd)   # [Bn, d]
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn
                           + offs_d[None, :] * stride_vd)   # [Bn, d]

        # Attention scores
        s = tl.dot(q, tl.trans(k)) * SCALE                  # [Bm, Bn]

        # Online softmax update
        m_ij  = tl.max(s, axis=1)
        p     = tl.exp(s - m_ij[:, None])
        l_ij  = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - tl.maximum(m_i, m_ij))
        m_i   = tl.maximum(m_i, m_ij)
        l_i   = alpha * l_i + tl.exp(m_ij - m_i) * l_ij
        acc   = alpha[:, None] * acc + tl.exp(m_ij - m_i)[:, None] * tl.dot(p, v)

    # Normalize and store output
    acc = acc / l_i[:, None]
    tl.store(O_ptr + offs_m[:, None] * stride_on
                   + offs_d[None, :] * stride_od, acc)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In practice, use <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">flash_attn</code> from the official <a href="https://github.com/Dao-AILab/flash-attention">Dao-AILab/flash-attention</a> repo, or <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">torch.nn.functional.scaled_dot_product_attention</code> (PyTorch ≥ 2.0) which uses FlashAttention internally. The Triton sketch above omits causal masking (因果掩码), dropout, and multi-head batching for clarity.</div>

### 3) PyTorch Drop-in Usage (PyTorch即插即用)

```python
import torch
import torch.nn.functional as F

# PyTorch >= 2.0 automatically dispatches to FlashAttention
# when inputs are on CUDA and dtype is float16/bfloat16

B, H, N, d = 2, 8, 2048, 64

Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.bfloat16)
K = torch.randn(B, H, N, d, device='cuda', dtype=torch.bfloat16)
V = torch.randn(B, H, N, d, device='cuda', dtype=torch.bfloat16)

# is_causal=True enables causal masking (autoregressive decoding)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print(out.shape)  # [2, 8, 2048, 64]
```

------

## 7. Comparison Table (对比总结)

| Dimension                      | Vanilla Attention (原始注意力) | FlashAttention v1                                            |
| ------------------------------ | ------------------------------ | ------------------------------------------------------------ |
| HBM Accesses (HBM访问)         | O(N² + Nd)                     | O(N²d/M)                                                     |
| Peak Memory (峰值内存)         | O(N²)                          | <span style="color:#E8600A;font-weight:700">O(N)</span>      |
| Numerical Exactness (精确性)   | Exact                          | <span style="color:#E8600A;font-weight:700">Exact</span>     |
| Kernel Fusion (核融合)         | ✗ (3 separate kernels)         | <span style="color:#E8600A;font-weight:700">✓ (1 fused kernel)</span> |
| Backward memory (反向内存)     | O(N²) saved P                  | O(N) — recomputes P                                          |
| Causal mask support (因果掩码) | ✓                              | ✓                                                            |
| Speedup on A100 (实测加速)     | 1×                             | ~3× (fp16, N=2048)                                           |
| Max context (最大上下文)       | ~2 K (40 GB GPU)               | ~16–64 K                                                     |

------

## 8. Limitations & What FlashAttention v2 Fixed (局限性)

<span style="color:#C0392B;font-weight:600">Known limitations of v1:</span>

-   <span style="color:#C0392B;font-weight:600">Work partitioning (工作划分)</span>: v1 parallelizes only over batch and heads, leaving GPU underutilized for small batch sizes
-   <span style="color:#C0392B;font-weight:600">Non-optimal GEMM tiling</span>: Inner-loop sequential over KV blocks limits warp-level parallelism (线程束并行)
-   <span style="color:#C0392B;font-weight:600">No GQA support</span>: Grouped-Query Attention (分组查询注意力) required additional modifications

FlashAttention v2 addresses these with: parallelism over sequence dimension, reduced non-GEMM FLOPs, and better warp partitioning — achieving ~2× further speedup over v1.

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> FlashAttention v1 achieves <strong>exact, fast, memory-efficient attention</strong> by tiling Q/K/V into SRAM-sized blocks and maintaining a 3-variable online running state (m, ℓ, O) — eliminating the O(N²) HBM bottleneck without any approximation.</div>
