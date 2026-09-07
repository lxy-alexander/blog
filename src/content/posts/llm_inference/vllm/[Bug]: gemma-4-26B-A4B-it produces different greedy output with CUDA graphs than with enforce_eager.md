---
title: "[Bug]: gemma-4-26B-A4B-it produces different greedy output with CUDA graphs than with enforce_eager"
published: 2026-09-04
description: "[Bug]: gemma-4-26B-A4B-it produces different greedy output with CUDA graphs than with enforce_eager"
image: ""
tags: ["llm_inference","vllm","[Bug]: gemma-4-26B-A4B-it produces different greedy output with CUDA graphs than with enforce_eager"]
category: llm_inference / vllm
draft: false
lang: ""
createdAt: "2026-09-04T22:39:49.616.832682862Z"
---

在仓库根目录 `/data/home/xli49/lxy/vllm` 执行。GPU 物理编号 1–4 会被 `CUDA_VISIBLE_DEVICES` 映射成进程内的 0–3。

### 1. 确认环境与 GPU

```
cd /data/home/xli49/lxy/vllm

git rev-parse HEAD

.venv/bin/python -c '
import torch, vllm
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
'

nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
```

本次使用 commit：

```
e35298628fceb490d4ca2a21c0a6e8f63d3c6740
```

### 2. 检查复现脚本

脚本：

[gemma4_compile_consistency.py](/data/home/xli49/lxy/vllm/benchmarks/gemma4_compile_consistency.py)

先做语法检查：

```python
#!/usr/bin/env python3
"""Compare Gemma 4 outputs across vLLM execution modes."""

import argparse
import json
from pathlib import Path


MODEL = "google/gemma-4-26B-A4B-it"
PROMPT = "The capital of France is Paris, and the capital of Japan is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("eager", "eager_native", "compile", "graph")
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--routed-experts", action="store_true")
    return parser.parse_args()


def serialize_logprobs(logprobs):
    if logprobs is None:
        return None
    return {
        str(token_id): {
            "logprob": float(item.logprob),
            "rank": item.rank,
            "decoded_token": item.decoded_token,
        }
        for token_id, item in logprobs.items()
    }


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": MODEL,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.mode in {"eager", "eager_native"},
        "max_model_len": 2496,
        "max_num_batched_tokens": 2496,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.85,
        "seed": 0,
        "disable_custom_all_reduce": True,
        "enable_return_routed_experts": args.routed_experts,
    }
    if args.mode == "compile":
        llm_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
    if args.mode == "eager_native":
        llm_kwargs["ir_op_priority"] = {
            "rms_norm": ["native"],
            "fused_add_rms_norm": ["native"],
        }

    llm = LLM(**llm_kwargs)
    result = llm.generate(
        [PROMPT],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
            prompt_logprobs=20,
            routed_experts_prompt_start=0,
        ),
    )[0]

    payload = {
        "mode": args.mode,
        "model": MODEL,
        "prompt": PROMPT,
        "prompt_token_ids": result.prompt_token_ids,
        "next_token_ids": list(result.outputs[0].token_ids),
        "output_logprobs": [
            serialize_logprobs(logprobs) for logprobs in result.outputs[0].logprobs
        ],
        "prompt_logprobs": [
            serialize_logprobs(logprobs)
            for logprobs in (result.prompt_logprobs or [])
        ],
        "llm_kwargs": llm_kwargs,
        "routed_experts": (
            result.outputs[0].routed_experts.tolist()
            if result.outputs[0].routed_experts is not None
            else None
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

```



```
.venv/bin/python -m py_compile \
  benchmarks/gemma4_compile_consistency.py
```

脚本固定了：

```
model                 google/gemma-4-26B-A4B-it
prompt                The capital of France is Paris, and the capital of Japan is
dtype                 BF16
temperature           0
max_tokens            1
prompt_logprobs       20
logprobs              20
tensor_parallel_size  4
max_model_len         2496
max_num_batched_tokens 2496
disable_custom_all_reduce True
```

这里使用 `2496`，是因为当前版本对 Gemma 4 多模态预算有校验：

```
max_tokens_per_mm_item = 2496
```

设置成 2048 会在加载前被配置校验拒绝。对于当前 13-token 纯文本 prompt，不会改变实际输入长度。

### 3. 公共环境变量

每次都使用：

```
CUDA_VISIBLE_DEVICES=1,2,3,4
VLLM_USE_FLASHINFER_SAMPLER=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

每种模式必须用独立 Python 进程运行，避免模型状态、CUDA context 和 KV cache 相互污染。

### 4. 运行默认 eager

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  eager \
  /tmp/gemma4-a5000-tp4-eager-2496.json \
  --tensor-parallel-size 4
```

该模式对应：

```
torch.compile = off
CUDA Graph    = off
RMSNorm       = vllm_c 优先
```

结果：

```
next token = 992
```

### 5. 运行 eager-native

这一步用于排除 RMSNorm kernel 不同的影响：

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  eager_native \
  /tmp/gemma4-a5000-tp4-eager-native.json \
  --tensor-parallel-size 4
```

对应：

```
torch.compile = off
CUDA Graph    = off
RMSNorm       = native
```

结果：

```
next token = 9079
```

### 6. 运行 compile-only

compile-only 开启 Inductor，但显式关闭 CUDA Graph：

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  compile \
  /tmp/gemma4-a5000-tp4-compile.json \
  --tensor-parallel-size 4
```

结果：

```
next token = 9079
```

再独立运行一次，验证 compile 自身是否稳定：

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  compile \
  /tmp/gemma4-a5000-tp4-compile-repeat.json \
  --tensor-parallel-size 4
```

两次 compile 结果逐项完全一致。

### 7. 运行默认 CUDA Graph

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  graph \
  /tmp/gemma4-a5000-tp4-graph.json \
  --tensor-parallel-size 4
```

再运行一次：

```
CUDA_VISIBLE_DEVICES=1,2,3,4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python benchmarks/gemma4_compile_consistency.py \
  graph \
  /tmp/gemma4-a5000-tp4-graph-repeat.json \
  --tensor-parallel-size 4
```

结果：

```
graph 1 next token = 9079
graph 2 next token = 9079
```

但第一次 Graph 的部分 logprob 与 compile 相差最多 `4.4588 nats`，第二次则与 compile 完全一致。

### 8. 比较结果

```
.venv/bin/python - <<'PY'
import json

paths = {
    "eager": "/tmp/gemma4-a5000-tp4-eager-2496.json",
    "eager_native": "/tmp/gemma4-a5000-tp4-eager-native.json",
    "compile1": "/tmp/gemma4-a5000-tp4-compile.json",
    "compile2": "/tmp/gemma4-a5000-tp4-compile-repeat.json",
    "graph1": "/tmp/gemma4-a5000-tp4-graph.json",
    "graph2": "/tmp/gemma4-a5000-tp4-graph-repeat.json",
}

data = {name: json.load(open(path)) for name, path in paths.items()}


def values(logprobs):
    return {
        str(token): float(item["logprob"])
        for token, item in logprobs.items()
    }


def compare(left, right):
    flips = 0
    count = 0
    worst = 0.0
    worst_at = None
    exact = True

    pairs = zip(
        data[left]["prompt_logprobs"],
        data[right]["prompt_logprobs"],
    )

    for position, (pa, pb) in enumerate(pairs):
        if pa is None or pb is None:
            continue

        count += 1
        a = values(pa)
        b = values(pb)

        if max(a, key=a.get) != max(b, key=b.get):
            flips += 1

        exact &= a == b

        for token in set(a) & set(b):
            delta = abs(a[token] - b[token])
            if delta > worst:
                worst = delta
                worst_at = (position, token)

    print(
        f"{left:13} vs {right:13}",
        f"tokens={data[left]['next_token_ids']}/"
        f"{data[right]['next_token_ids']}",
        f"flips={flips}/{count}",
        f"worst={worst:.6f}",
        f"at={worst_at}",
        f"exact={exact}",
    )


compare("eager", "eager_native")
compare("eager_native", "compile1")
compare("eager", "compile1")
compare("compile1", "compile2")
compare("compile1", "graph1")
compare("compile1", "graph2")
compare("graph1", "graph2")
PY
```

预期输出核心部分：

```
eager vs eager_native:
tokens=[992]/[9079]
flips=6/12
worst=7.192223

eager_native vs compile1:
tokens=[9079]/[9079]
flips=4/12
worst=8.647717

eager vs compile1:
tokens=[992]/[9079]
flips=6/12
worst=7.306273

compile1 vs compile2:
flips=0/12
worst=0
exact=True

compile1 vs graph1:
flips=0/12
worst=4.458825

compile1 vs graph2:
flips=0/12
worst=0
exact=True
```

整体流程就是：

```
默认 eager
    ↓
eager-native：隔离 RMSNorm 实现差异
    ↓
compile-only ×2：隔离 Inductor，并验证重复稳定性
    ↓
CUDA Graph ×2：检查 graph 相对 compile 的影响及跨运行稳定性
    ↓
比较 greedy token、prompt argmax、最大 logprob delta
```







# 结果

可以，而且已经用 GPU 1–4 的 4×RTX A5000、TP=4 完成验证。这个 issue 在 A5000 上复现得比单卡测试更明显。

### 结果

固定条件：

-   vLLM commit：`e35298628fceb490d4ca2a21c0a6e8f63d3c6740`
-   PyTorch：`2.13.0+cu130`
-   BF16，TP=4
-   Triton Attention、Triton MoE
-   TP 通信：PyNCCL
-   temperature=0
-   禁用 FlashInfer sampler 和 custom all-reduce

| 模式                         | 下一个 token | 对比                    | Prompt argmax 翻转 | 最大差      |
| ---------------------------- | ------------ | ----------------------- | ------------------ | ----------- |
| eager，默认 `vllm_c` RMSNorm | 992          | eager vs compile        | 6/12               | 7.3063 nats |
| eager，强制 native RMSNorm   | 9079         | native-eager vs compile | 4/12               | 8.6477 nats |
| compile-only，第 1 次        | 9079         | compile1 vs compile2    | 0/12               | 0           |
| compile-only，第 2 次        | 9079         | 完全一致                | 0/12               | 0           |
| CUDA Graph，第 1 次          | 9079         | compile vs graph1       | 0/12               | 4.4588 nats |
| CUDA Graph，第 2 次          | 9079         | compile vs graph2       | 0/12               | 0           |

所以原报告中的：

```
eager: 992
CUDA Graph: 9079
```

在 4×A5000 上完整复现。

### 新发现

问题不只有一个来源。

1.  RMSNorm kernel 选择差异很重要

默认 eager 使用：

```
rms_norm = vllm_c
```

compile 使用：

```
rms_norm = native / Inductor
```

仅在 eager 中把 RMSNorm 改成 native，token 就从：

```
992 → 9079
```

所以在 A5000 TP=4 上，最终 greedy token 翻转很大一部分来自 eager 与 compile 使用不同 RMSNorm 实现。

1.  `torch.compile` 融合问题仍然独立存在

即使 eager 和 compile 都强制使用 native RMSNorm：

```
eager-native vs compile
argmax 翻转：4/12
最大差：8.6477 nats
```

说明 Router scaling、residual/scalar 等 Inductor 融合和 BF16 舍入边界变化仍然会显著改变分布。并不能只修 RMSNorm kernel 选择。

1.  CUDA Graph 路径自身存在跨运行不稳定

两次 compile-only 逐项完全相同，排除了普通 TP/NCCL 跨进程随机性。

但是两次 Graph：

```
graph1：与 compile 最大相差 4.4588 nats
graph2：与 compile 逐项完全一致
```

这和 issue 中“CUDA Graph 两次运行也不一致”的描述吻合。虽然两次 Graph 的 top-1 都保持 `9079`，但概率分布不稳定。

### 当前结论

这个问题至少有三层：

```
eager 与 compile 的 RMSNorm kernel 不同
                +
Inductor 改变 BF16 融合/舍入边界
                +
CUDA Graph 路径存在跨运行状态或捕获差异
                ↓
MoE top-k 放大差异
                ↓
最终 greedy token 可能翻转
```

因此需要修的不只是一个 CUDA Graph 开关，也不只是简单提高精度。

诊断脚本保存在：

[gemma4_compile_consistency.py](/data/home/xli49/lxy/vllm/benchmarks/gemma4_compile_consistency.py)

主要结果：

-   [eager](/tmp/gemma4-a5000-tp4-eager-2496.json)
-   [eager-native](/tmp/gemma4-a5000-tp4-eager-native.json)
-   [compile 1](/tmp/gemma4-a5000-tp4-compile.json)
-   [compile 2](/tmp/gemma4-a5000-tp4-compile-repeat.json)
-   [CUDA Graph 1](/tmp/gemma4-a5000-tp4-graph.json)
-   [CUDA Graph 2](/tmp/gemma4-a5000-tp4-graph-repeat.json)

没有修改 vLLM 生产代码，只增加了诊断脚本。脚本通过 `py_compile` 和 `git diff --check`；当前环境没有安装 `ruff`。
