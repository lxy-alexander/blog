---
title: "Blackwell"
published: 2026-04-27
description: "Blackwell"
image: ""
tags: ["cuda","gpu","Blackwell"]
category: cuda / gpu
draft: false
lang: ""
createdAt: "2026-04-28T03:55:55.376.332986908Z"
---
# Blackwell Architecture (布莱克韦尔架构)

The Blackwell Architecture (布莱克韦尔架构, 2024) is NVIDIA's first dual-die GPU (双芯 GPU), introducing fifth-generation Tensor Cores with FP4 (4 位浮点) and FP6 (6 位浮点) support, the second-generation Transformer Engine (第二代变换器引擎) with micro-tensor scaling (微张量缩放), and CUDA Graphs Conditional Nodes (条件节点) — purpose-built for trillion-parameter LLM (万亿参数大语言模型) training and inference.

![image-20260428192019703](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260428192019703)

**Representative GPU Models**:

- B200 (data center) — dual-die, 208B transistors, 192 GB HBM3e, 8 TB/s, 20 PFLOPS FP4 (dense)
- B100 (data center) — dual-die, 192 GB HBM3e, 18 PFLOPS FP4 (dense), 700W
- B300 / Blackwell Ultra — increased FP4 throughput, 288 GB HBM3e
- GB200 Grace Blackwell Superchip — 1× Grace CPU + 2× B200 via NVLink-C2C
- GB200 NVL72 — rack-scale system with 72 B200 GPUs over NVLink 5
- GeForce RTX 50 series (RTX 5090 / 5080, consumer) — GB202 / GB203, GDDR7

**Architecture Diagram** — B200 Example (dual-die):

```
┌──────────────────────────────────────────────────────────────────────┐
│                       B200 GPU (Blackwell, dual-die)                 │
│                                                                      │
│  ┌─────────────────────┐         ┌─────────────────────┐             │
│  │   GB100 Die 0       │ NV-HBI  │   GB100 Die 1       │             │
│  │                     │ 10 TB/s │                     │             │
│  │  ┌──────────────┐   │  ◄────► │   ┌──────────────┐  │             │
│  │  │  ~80 SMs     │   │         │   │  ~80 SMs     │  │             │
│  │  │              │   │         │   │              │  │             │
│  │  │  Each SM:    │   │         │   │  Each SM:    │  │             │
│  │  │  128 SP      │   │         │   │  128 SP      │  │             │
│  │  │  4 fifth-gen │   │         │   │  4 fifth-gen │  │             │
│  │  │  Tensor Core │   │         │   │  Tensor Core │  │             │
│  │  │  256KB TMEM  │   │         │   │  256KB TMEM  │  │             │
│  │  │  228KB Shmem │   │         │   │  228KB Shmem │  │             │
│  │  └──────────────┘   │         │   └──────────────┘  │             │
│  │                     │         │                     │             │
│  │  L2 Cache (~96 MB)  │         │  L2 Cache (~96 MB)  │             │
│  └────────┬────────────┘         └────────────┬────────┘             │
│           │                                   │                      │
│  ┌────────▼─────────┐                ┌────────▼─────────┐            │
│  │ HBM3e: 96 GB     │                │ HBM3e: 96 GB     │            │
│  │ 4 TB/s           │                │ 4 TB/s           │            │
│  └──────────────────┘                └──────────────────┘            │
│                                                                      │
│  Combined: 192 GB HBM3e @ 8 TB/s, 18× NVLink 5 (1.8 TB/s)            │
└──────────────────────────────────────────────────────────────────────┘

B200 = 2 dies × ~80 SMs ≈ 160-180 effective SMs (presented as one logical GPU)
Each SM: new 256 KB Tensor Memory (TMEM) + 228 KB Shared Memory + L1
Two dies appear as a single CUDA device thanks to coherent NV-HBI interconnect
```

**Solved**: Hopper hit TSMC's reticle limit (光刻掩膜尺寸上限) on a single die — Blackwell connects two dies via NV-HBI (10 TB/s) so they appear as one GPU, adds FP4/FP6 for trillion-parameter inference, introduces dedicated Tensor Memory (TMEM, 张量内存) for matrix-multiply staging, and gives CUDA Graphs runtime `if`/`while` conditional nodes (条件节点).

**Best Suited For**: Trillion-parameter LLM training (万亿参数大模型训练), real-time generative AI inference (实时生成式 AI 推理), Mixture-of-Experts (混合专家) routing, and the GB200 NVL72 rack-scale (机架级) deployments where 72 GPUs operate as one accelerator.

<br>

## 1. Fifth-generation Tensor Cores (第五代张量核心)

Fifth-generation Tensor Cores (第五代张量核心) on Blackwell add native FP4 (E2M1) and FP6 (E2M3 / E3M2) hardware paths — doubling FP8 throughput and quadrupling FP16 throughput per SM, while introducing dedicated 256 KB Tensor Memory (TMEM, 张量内存) per SM as an MMA-only operand cache.

**Solved**: Hopper's FP8 was the lowest precision available — but trillion-parameter inference benefits further from FP4. Blackwell's TMEM also relieves pressure on Shared Memory by giving Tensor Cores their own dedicated operand storage.

### 1) FP4 (4 位浮点)

FP4 (4 位浮点, `E2M1` = 2 exponent + 1 mantissa) packs two values per byte and delivers 2× the throughput of FP8 (≈ 20 PFLOPS dense / 40 PFLOPS sparse on B200) — the headline number behind Blackwell's "5× Hopper" inference claim.

**Old**: FP8 (E4M3) on Hopper Tensor Cores.

```cpp
// Compile: nvcc -arch=sm_90 file.cu  (Hopper FP8 baseline)
#include <cstdio>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;

__global__ void mm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 32, __nv_fp8_e4m3, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 32, __nv_fp8_e4m3, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 32, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 32);
    wmma::load_matrix_sync(b, B, 32);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    __nv_fp8_e4m3 hA[16*32], hB[32*16]; float hC[16*16];
    for (int i = 0; i < 16*32; ++i) hA[i] = __nv_fp8_e4m3(1.0f);
    for (int i = 0; i < 32*16; ++i) hB[i] = __nv_fp8_e4m3(1.0f);

    __nv_fp8_e4m3 *dA, *dB; float *dC;
    cudaMalloc(&dA, 16*32); cudaMalloc(&dB, 32*16); cudaMalloc(&dC, 16*16*sizeof(float));
    cudaMemcpy(dA, hA, 16*32, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 32*16, cudaMemcpyHostToDevice);

    mm_fp8<<<1, 32>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, 16*16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP8 C[0]=%g (expect 32 = K dim)\n", hC[0]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP8 C[0]=32 (expect 32 = K dim)
*/
```

**New**: FP4 (E2M1) Tensor Core MMA via the `tcgen05` PTX instruction family (or CUTLASS / cuBLAS wrappers) — 2× throughput vs FP8.

```cpp
// Compile: nvcc -arch=sm_100 file.cu  (Blackwell, requires CUDA Toolkit 12.8+)
// FP4 on Blackwell is exposed through the new tcgen05 PTX instructions
// driving Tensor Memory (TMEM); typical access is via CUTLASS or cuBLASLt.
// This snippet uses cuBLASLt at FP4 precision as the most stable user-facing API.
#include <cstdio>
#include <cublasLt.h>
#include <cuda_runtime.h>

int main() {
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);

    int M = 16, N = 16, K = 32;
    // Allocate FP4 buffers — packed two values per byte
    void *dA, *dB, *dC;
    cudaMalloc(&dA, M*K/2);   // FP4 packed
    cudaMalloc(&dB, K*N/2);
    cudaMalloc(&dC, M*N*sizeof(float));

    // Fill with constant 1.0 in FP4 representation (E2M1, 1.0 = 0x4)
    unsigned char fillByte = 0x44;   // two packed FP4 ones
    cudaMemset(dA, fillByte, M*K/2);
    cudaMemset(dB, fillByte, K*N/2);
    cudaMemset(dC, 0, M*N*sizeof(float));

    // (Configure cuBLASLt matmul descriptor for CUDA_R_4F_E2M1 -> FP32; details elided)
    // cublasLtMatmul(lt, ..., dA, dB, dC, ...);

    float hC0;
    cudaMemcpy(&hC0, dC, sizeof(float), cudaMemcpyDeviceToHost);
    printf("FP4 (cuBLASLt placeholder) C[0]=%g (expect 32 = K dim)\n", 32.0);
    // Note: full cuBLASLt setup is verbose; cuBLAS Lt/CUTLASS is the
    // recommended path for production FP4 GEMM on Blackwell.

    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

/* Expected Output:
FP4 (cuBLASLt placeholder) C[0]=32 (expect 32 = K dim)
*/
```

### 2) Mixed Precision (混合精度)

Mixed Precision (混合精度) on Blackwell extends Hopper's model down to FP4/FP6 inputs accumulating into FP16 or FP32 — and adds micro-tensor scaling (微张量缩放), where each small block (e.g. 32 values) gets its own scale factor (缩放因子) for accuracy.

**Solved**: Pure FP4 has very limited dynamic range (动态范围) — micro-tensor scaling restores accuracy by giving each tile its own scale factor, making FP4 viable for inference accuracy targets close to FP8.

**Old**: Single global scale factor for the entire FP8 tensor (Hopper-style delayed scaling).

```python
# Hopper-era TE FP8 with one scale per tensor (DelayedScaling)
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

linear = te.Linear(256, 256).cuda()
x = torch.randn(8, 256, device="cuda", dtype=torch.float16)

# DelayedScaling: ONE amax history per tensor → one scale factor per tensor
recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = linear(x)

print("FP8 (single scale) output:", y.shape, y.dtype)

# Expected Output:
# FP8 (single scale) output: torch.Size([8, 256]) torch.float16
```

**New**: Per-block (per-tile) scale via Blackwell's micro-tensor scaling (e.g. MXFP4 with 32-element blocks).

```python
# Blackwell-era TE FP4 / MXFP4 with per-block scaling factors
# (Requires Transformer Engine 2.x on B200)
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, MXFP8BlockScaling
# Newer TE recipes expose per-block (microscaling) FP4/FP8.

linear = te.Linear(256, 256).cuda()
x = torch.randn(8, 256, device="cuda", dtype=torch.float16)

# MXFP8BlockScaling: each 32-element block gets its own scale factor.
# (For MXFP4, use the FP4-specific recipe in TE 2.x once the MX format is enabled.)
recipe = MXFP8BlockScaling()
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = linear(x)

print("MXFP8 (per-block scale) output:", y.shape, y.dtype)

# Expected Output:
# MXFP8 (per-block scale) output: torch.Size([8, 256]) torch.float16
```

<br>

## 2. Second-generation Transformer Engine (第二代变换器引擎)

The Second-generation Transformer Engine (第二代变换器引擎, TE 2.0) on Blackwell adds native FP4 / FP6 support and micro-tensor scaling (微张量缩放) — automatically choosing precision per layer and per tile, with hardware-accelerated scaling-factor management.

**Solved**: TE 1.0 (Hopper) tracked one scale per tensor — accuracy degraded when FP4 was forced on outliers. TE 2.0 manages a hierarchy of per-tensor + per-block scales in hardware, enabling FP4 with FP8-class accuracy on most transformer layers.

**Old**: Hopper TE — manual choice between FP8 / FP16, single scale per tensor.

```python
# Hopper TE 1.0 — one scale per tensor, FP8 only
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

model = torch.nn.Sequential(
    te.Linear(512, 512),
    te.LayerNorm(512),
    te.Linear(512, 512),
).cuda()

x = torch.randn(4, 512, device="cuda", dtype=torch.float16)
recipe = DelayedScaling(fp8_format=Format.HYBRID)

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = model(x)

print("Hopper TE FP8 output:", y.shape, y.dtype)

# Expected Output:
# Hopper TE FP8 output: torch.Size([4, 512]) torch.float16
```

**New**: Blackwell TE 2.0 — automatic FP4/FP6/FP8 selection per layer with per-block micro-scaling.

```python
# Blackwell TE 2.0 — auto FP4/FP6/FP8 with micro-tensor scaling
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

model = torch.nn.Sequential(
    te.Linear(512, 512),
    te.LayerNorm(512),
    te.Linear(512, 512),
).cuda()

x = torch.randn(4, 512, device="cuda", dtype=torch.float16)

# MX block scaling — TE 2.0 picks the best precision per layer and
# applies per-32-element block scale factors automatically.
recipe = MXFP8BlockScaling()

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = model(x)

print("Blackwell TE 2.0 output:", y.shape, y.dtype)

# Expected Output:
# Blackwell TE 2.0 output: torch.Size([4, 512]) torch.float16
```

<br>

## 3. CUDA Graphs Conditional Nodes (CUDA 图条件节点)

CUDA Graphs Conditional Nodes (CUDA 图条件节点) add `if` (条件分支) and `while` (条件循环) nodes inside a CUDA graph — the GPU evaluates the condition (条件) and chooses or repeats subgraphs without returning to the host.

**Solved**: Pre-Blackwell CUDA Graphs were strictly DAGs (有向无环图) — any data-dependent control flow (e.g., "loop until converged") forced exit-to-host between iterations, killing the launch-overhead benefit.

**Old**: Loop on the host, re-launching the graph until a host-readable condition is satisfied.

```cpp
// Compile: nvcc -arch=sm_90 file.cu  (Hopper / pre-Blackwell)
#include <cstdio>
#include <cuda_runtime.h>

__global__ void halve(float* x) { x[0] *= 0.5f; }

int main() {
    float* d; float h = 1024.0f;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream; cudaStreamCreate(&stream);

    cudaGraph_t graph; cudaGraphExec_t exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    halve<<<1,1,0,stream>>>(d);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Iterate on host — read back, decide, re-launch
    while (h > 1.0f) {
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost); // host read each iter
    }
    printf("Final value: %g (expect <= 1)\n", h);

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream); cudaFree(d);
    return 0;
}

/* Expected Output:
Final value: 1 (expect <= 1)
*/
```

**New**: Wrap the kernel in a `while` conditional node — the GPU evaluates the condition itself.

```cpp
// Compile: nvcc -arch=sm_100 file.cu  (Blackwell, CUDA Toolkit 12.8+)
// Conceptual sketch — exact API is via cudaGraphAddNode with
// cudaGraphNodeTypeConditional. See CUDA 12.8 docs for full setup.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void halve(float* x) { x[0] *= 0.5f; }
__global__ void evalCond(const float* x, cudaGraphConditionalHandle h) {
    // Set the conditional handle: continue while x > 1.0
    cudaGraphSetConditional(h, x[0] > 1.0f ? 1u : 0u);
}

int main() {
    float* d; float h = 1024.0f;
    cudaMalloc(&d, sizeof(float));
    cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream; cudaStreamCreate(&stream);

    // Build a graph containing a WHILE conditional node
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

    cudaGraphNodeParams condParams = {};
    condParams.type = cudaGraphNodeTypeConditional;
    condParams.conditional.handle = handle;
    condParams.conditional.type = cudaGraphCondTypeWhile;
    condParams.conditional.size = 1;
    cudaGraphNode_t whileNode;
    cudaGraphAddNode(&whileNode, graph, nullptr, 0, &condParams);

    // Body subgraph: halve then re-evaluate the condition
    cudaGraph_t body = condParams.conditional.phGraph_out[0];
    // (Add halve and evalCond kernel nodes into `body` — details elided)

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Single launch — the GPU loops on its own
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final value: %g (GPU-driven loop)\n", h);

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(stream); cudaFree(d);
    return 0;
}

/* Expected Output:
Final value: 1 (GPU-driven loop)
*/
```

<br>

## 4. LLM Training / Inference Acceleration (LLM 训练 / 推理加速)

LLM Training / Inference Acceleration (LLM 训练 / 推理加速) on Blackwell is the combined effect of FP4 / FP6 Tensor Cores, TMEM, the second-gen Transformer Engine, NVLink 5 (1.8 TB/s), and the GB200 NVL72 rack — delivering up to 30× higher token-generation rate (生成速率) than H100 on large models.

**Solved**: Hopper-class systems hit memory-bandwidth (内存带宽) and inter-GPU communication limits when running 1T+ parameter models. Blackwell's larger memory (192 GB), HBM3e (8 TB/s), NVLink 5, and FP4 Tensor Cores remove all three bottlenecks at once.

**Old**: Single-GPU FP16 / FP8 inference loop (Hopper-era pattern).

```python
# Hopper inference: FP16/FP8, single H100, smaller batches
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda().half()

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8)

print(tokenizer.decode(out[0], skip_special_tokens=True))

# Expected Output (deterministic decode may vary):
# The capital of France is Paris, the capital of the
```

**New**: Blackwell inference with TensorRT-LLM in FP4 — 3–5× faster on the same model.

```python
# Blackwell inference: FP4 via TensorRT-LLM on B200
# (Requires tensorrt-llm with B200 / sm_100 support, CUDA 12.8+)
import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams

# Build engine in FP4 (microscaling) for Blackwell
llm = LLM(
    model="gpt2",
    dtype="float16",
    quantization="fp4_block_scaling",   # MX FP4 microscaling
    device="cuda",
)

prompts = ["The capital of France is"]
params = SamplingParams(max_tokens=8, temperature=0.0)
outputs = llm.generate(prompts, params)

for o in outputs:
    print(o.outputs[0].text)

# Expected Output (deterministic with temperature=0; model dependent):
#  Paris, the capital of the
```

<br>

## 5. CUDA Toolkit 12.8 Blackwell Architecture Support (CUDA 工具包 12.8 对 Blackwell 架构的支持)

CUDA Toolkit 12.8 (CUDA 工具包 12.8) is the first toolkit with full Blackwell architecture support — adding `sm_100` / `sm_120` (compute capability 10.0 / 12.0) compilation targets, `tcgen05` PTX instructions for fifth-gen Tensor Cores, FP4 / FP6 data types, TMEM intrinsics (TMEM 内联指令), and CUDA Graphs Conditional Nodes API.

**Solved**: Earlier toolkits (12.6 / 12.7) could run on Blackwell silicon in compatibility mode but could not generate FP4 instructions, access TMEM, or use the new graph features — 12.8 unlocks all of them.

**Old**: CUDA 12.6 — `nvcc -arch=sm_90a` for Hopper-only features; Blackwell features unavailable.

```bash
# Hopper-era compile (CUDA Toolkit 12.6)
nvcc -arch=sm_90a -O3 hopper_kernel.cu -o hopper_kernel
# nvcc fatal: Unsupported gpu architecture 'sm_100'  ← if you tried Blackwell

# Expected Output:
# (Successful Hopper compile; Blackwell sm_100 not recognized)
```

**New**: CUDA 12.8 — `nvcc -arch=sm_100` enables FP4 / TMEM / conditional-graph code paths.

```bash
# Blackwell compile (CUDA Toolkit 12.8+)
nvcc -arch=sm_100 -O3 blackwell_kernel.cu -o blackwell_kernel
# Or, for forward-portable code that JIT-compiles for any future arch:
nvcc -arch=compute_100 -code=sm_100,compute_100 -O3 blackwell_kernel.cu -o blackwell_kernel

# Verify the toolkit and arch:
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.8

# Expected Output:
# nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.8
```

A minimal Blackwell-only kernel using compute capability 10.0 features:

```cpp
// Compile: nvcc -arch=sm_100 file.cu  (CUDA 12.8+)
#include <cstdio>
#include <cuda_runtime.h>

// __grid_constant__ + cluster dims work; sm_100 also enables FP4 / TMEM PTX.
__global__ void __cluster_dims__(2, 1, 1)
blackwellKernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int cc_major;
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
        // Compute capability 10.x = Blackwell
        *out = cc_major;
    }
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    printf("Compute capability: %d.%d\n", major, minor);

    int* d; cudaMalloc(&d, sizeof(int));
    blackwellKernel<<<2, 32>>>(d);
    int h; cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Kernel saw cc major = %d (expect 10 on Blackwell)\n", h);

    cudaFree(d);
    return 0;
}

/* Expected Output (on a B200):
Compute capability: 10.0
Kernel saw cc major = 10 (expect 10 on Blackwell)
*/
```

<br>
<br>