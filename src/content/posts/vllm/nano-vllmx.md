---
title: "nano-vllm"
published: 2026-03-20
description: "nano-vllm"
image: ""
tags: ["vllm","nano-vllm"]
category: vllm
draft: false
lang: ""
---

# 1. 初始化阶段（LLMEngine.__init__）

```
用户：LLM(path, ...)
```

| 步骤 | 内容                                   |
| ---- | -------------------------------------- |
| 1    | 解析 Config                            |
| 2    | 启动 Tensor Parallel 子进程（tp>1 时） |
| 3    | 创建 ModelRunner（主进程 rank 0）      |
| 4    | 加载 Tokenizer                         |
| 5    | 创建 Scheduler                         |

---

# 2. 模型构建阶段（ModelRunner.__init__）

```
ModelRunner(config, 0, events)
```

| 步骤 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 1    | `dist.init_process_group`                                    |
| 2    | `Qwen3ForCausalLM(hf_config)`：construct model structure + allocate space for parameters |
| 3    | `load_model(model, path)`：从 safetensors 加载权重           |
| 4    | `warmup_model()`：跑一次 prefill，触发 JIT                   |
| 5    | `allocate_kv_cache()`：分配 KV cache blocks，挂到 Attention  |
| 6    | `capture_cudagraph()`：为 decode 捕获 CUDAGraph（可选）      |

---









## 3. 请求入队阶段

```
用户：add_request(prompt) 或 generate(prompts)
```

| 步骤 | 内容                                   |
| ---- | -------------------------------------- |
| 1    | Tokenizer 编码 prompt → token_ids      |
| 2    | 构建 Sequence（含 sampling_params）    |
| 3    | Scheduler.add(seq) → 进入 waiting 队列 |

---

## 4. 推理循环阶段（step()）

```
while not is_finished():
    step()
```

每个 step 内部：

| 子阶段       | 职责                                                         |
| ------------ | ------------------------------------------------------------ |
| **调度**     | `scheduler.schedule()` → 选出 seqs，决定 prefill 或 decode   |
| **数据准备** | `prepare_prefill` / `prepare_decode` → input_ids, positions, slot_mapping 等 |
| **模型前向** | `run_model` → embed → layers → lm_head → logits              |
| **采样**     | `sampler(logits)` → token_ids                                |
| **后处理**   | `scheduler.postprocess` → append_token，更新状态，回收 blocks |

---



==Prefill 存在的意义就是：利用 prompt 已知的特点，用 GPU 并行一次性处理完，避免逐 token 串行的巨大开销。==





## 5. 输出阶段

```
generate() 返回后
```

| 步骤 | 内容                                          |
| ---- | --------------------------------------------- |
| 1    | 按 seq_id 收集 outputs                        |
| 2    | Tokenizer.decode(token_ids) → 文本            |
| 3    | 返回 `[{"text": ..., "token_ids": ...}, ...]` |

---

## 阶段关系示意

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. 初始化    Config + ModelRunner + Scheduler + Tokenizer          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  2. 模型构建  建图 → 加载权重 → warmup → 分配 KV cache → CUDAGraph   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  3. 请求入队  prompt → tokenize → Sequence → Scheduler.add()        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  4. 推理循环  schedule → prepare → run_model → sampler → postprocess │
│             （prefill 和 decode 交替）                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│  5. 输出     收集完成 seq → decode → 返回 texts                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 简要对照表

| 阶段       | 入口                       | 主要动作                                        |
| ---------- | -------------------------- | ----------------------------------------------- |
| 1 初始化   | `LLM(...)`                 | Config、进程、Tokenizer、Scheduler              |
| 2 模型构建 | `ModelRunner.__init__`     | 模型结构、加载权重、KV cache、CUDAGraph         |
| 3 请求入队 | `add_request` / `generate` | Tokenize → Sequence → waiting                   |
| 4 推理循环 | `step()`                   | schedule → prepare → run → sample → postprocess |
| 5 输出     | `generate` 返回            | 收集、decode、返回文本                          |
