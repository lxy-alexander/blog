---
title: "vllm serve CLI"
published: 2026-06-08
description: "vllm serve CLI"
image: ""
tags: ["vllm","vllm serve CLI"]
category: vllm
draft: false
lang: ""
createdAt: "2026-06-08T20:07:05.385.059990919Z"
---

#  `vllm serve` 常用参数

`vllm serve` 调参可以按这个顺序记：先定访问地址，再定安全 key，再定模型名，然后看上下文、显存、并发，最后才调多卡和性能后端。

## 1. 服务地址：`--host` 和 `--port`

这两个参数决定别人怎么访问你的 vLLM 服务。`--host` 是监听地址，`--port` 是端口，默认端口是 `8000`。

### 场景 A：只给本机访问

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000
```

含义：

1）`127.0.0.1`：只有当前机器能访问。

2）`8000`：服务地址是 `http://127.0.0.1:8000`。

面试口吻：本地测试用 `127.0.0.1`，安全但只能自己访问。

### 场景 B：局域网或服务器外部访问

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8080
```

含义：

1）`0.0.0.0`：监听所有网卡。

2）`8080`：服务地址可能是 `http://服务器IP:8080`。

面试口吻：对外提供服务用 `0.0.0.0`，但一定要配合认证或防火墙。

## 2. 访问认证：`--api-key`

`--api-key` 表示客户端请求时必须带上指定 key，vLLM 支持传入一个或多个 key。

### 启动服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key sk-vllm-123456
```

含义：

1）服务端只接受带 `sk-vllm-123456` 的请求。

2）没有 key 或 key 错误，请求会被拒绝。

### 客户端调用

```bash
curl http://localhost:8000/v1/models \
    -H "Authorization: Bearer sk-vllm-123456"
```

面试口吻：`--api-key` 不影响模型效果，只控制谁能调用服务。

## 3. API 模型名：`--served-model-name`

`--served-model-name` 决定 API 请求里写什么模型名；==如果不设置，默认使用 `--model` 的值。==

### 启动服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key sk-vllm-123456 \
    --served-model-name qwen-local
```

### 请求时使用自定义模型名

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-vllm-123456" \
    -d '{
        "model": "qwen-local",
        "messages": [
            {"role": "user", "content": "你好，简单介绍一下 vLLM"}
        ]
    }'
```

含义：

1）实际加载的是 `Qwen/Qwen2.5-1.5B-Instruct`。

2）客户端看到和填写的是 `qwen-local`。

面试口吻：`--model` 是后端真实模型，`--served-model-name` 是前端暴露名字。

## 4. 上下文长度：`--max-model-len`

It indicates the maximum number of tokens that can be processed in one request, including both input and output, limit the single request's context's length.

`--max-model-len` ==表示一次请求最多能处理多少 token，包括输入和输出==；它支持 `1k`、`1K`、`25.6k` 这种写法，比如 `1k = 1000`，`1K = 1024`。

### 设置 4096 token 上下文

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

含义：

1）输入 prompt 是 3000 token。

2）最多还能生成大约 1096 token。

3）因为 `3000 + 1096 = 4096`。

面试口吻：`max-model-len` 是输入加输出的总预算，不是只限制输入。

### 设置 8192 token 上下文

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

含义：

1）输入 6000 token。

2）输出最多大约 2192 token。

3）长上下文更强，但 KV cache（键值缓存）更吃显存。

面试口吻：上下文越长，能处理的内容越多，但显存压力越大。==The more content it can process, the greater pressure on the GPU memory.==

## 5. GPU 显存比例：`--gpu-memory-utilization`

It indicates the maximum proportion of GPU memory that the vLLM instance can use

`--gpu-memory-utilization` 表示 vLLM 这个实例最多使用多少比例的 GPU 显存，取值范围是 `0` 到 `1`，默认是 `0.92`。例如 `0.5` 表示使用 50% GPU 显存；如果两个 vLLM 实例跑在同一张 GPU 上，可以各自设置为 `0.5`。

### 单个服务使用 90% 显存

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90
```

假设一张 GPU 有 24GB 显存：

1）`0.90` 表示 vLLM 目标使用约 `24GB * 0.90 = 21.6GB`。

2）剩下约 `2.4GB` 给系统、CUDA、其他进程留空间。

面试口吻：这个参数不是越大越好，太满容易 OOM（显存溢出）。

### 同一张 GPU 跑两个服务

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.45
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.45
```

假设 GPU 是 24GB：

1）第一个服务约用 `24GB * 0.45 = 10.8GB`。

2）第二个服务约用 `24GB * 0.45 = 10.8GB`。

3）合计约 `21.6GB`，还剩约 `2.4GB` 缓冲。

面试口吻：多实例部署时，每个实例都要单独限制显存比例。

## 6. KV Cache 固定显存：`--kv-cache-memory-bytes`

`--kv-cache-memory-bytes` 可以直接指定每张 GPU 给 KV cache（键值缓存）多少显存；==一旦设置它，会忽略 `--gpu-memory-utilization` 对 KV cache 的自动推断。==

### 固定给 KV cache 8GB

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --kv-cache-memory-bytes 8G
```

含义：

1）KV cache 固定使用约 8GB。

2）适合你想精确控制显存分配的场景。

3）如果你只是普通部署，优先用 `--gpu-memory-utilization` 更简单。

面试口吻：`gpu-memory-utilization` 是粗粒度显存控制，`kv-cache-memory-bytes` 是精确控制 KV cache。

## 7. 并发请求数：`--max-num-seqs`

Indicates the maximum number of sequences that can be processed in one scheduling iteration.

`--max-num-seqs` 表示一次调度迭代最多处理多少个 sequence（序列），可以简单理解为同时处理多少条请求。

### 限制最多 32 条序列

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 32
```

含义：

1）一轮调度最多处理 32 条请求序列。

2）数值越大，并发能力可能越强。

3）但显存和调度压力也会增加。

面试口吻：`max-num-seqs` 控制并发请求数量，适合调服务吞吐。

## 8. 单轮 token 预算：`--max-num-batched-tokens`

显存压力 = 已有所有请求的 KV cache
        \+ 本轮新 token 产生的 KV cache
        \+ 本轮 attention / MLP / kernel 临时 buffer
        \+ 模型权重

`--max-num-batched-tokens` 表示单次调度迭代最多处理多少 token；它也支持 `1k`、`1K`、`25.6k` 这种人类可读格式。

### 设置单轮最多 8192 token

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-batched-tokens 8192
```

举例：

1）如果有 8 个请求，每个请求当前需要处理 1000 token，总共是 8000 token，可以放进一轮。

2）如果有 16 个请求，每个请求当前需要处理 1000 token，总共是 16000 token，就不能一轮全部处理。

面试口吻：`max-num-batched-tokens` 控制一轮能吃多少 token，是吞吐和显存之间的平衡点。

## 9. 多 GPU：`--tensor-parallel-size`

`--tensor-parallel-size`，简写 `-tp`，表示 tensor parallel（张量并行）数量，默认是 `1`。

### 单机 2 张 GPU

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2
```

含义：

1）模型会切到 2 张 GPU 上。

2）适合单卡显存放不下模型，或者希望提高吞吐的场景。

3）如果机器只有 1 张 GPU，就不能设置成 2。

面试口吻：模型太大单卡放不下时，优先考虑 `tensor-parallel-size`。

### 单机 4 张 GPU

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4
```

含义：

1）14B 模型被切到 4 张 GPU。

2）每张卡承担一部分权重和计算。

3）通信开销会增加，所以不是卡越多就线性变快。

面试口吻：TP 可以解决显存问题，但会引入 GPU 间通信成本。

## 10. CPU offload：`--cpu-offload-gb`

`--cpu-offload-gb` 表示每张 GPU 可以把多少 GiB 权重卸载到 CPU；比如 24GB GPU 设置 `10`，可以近似理解成“虚拟 34GB GPU”，但需要较快的 CPU-GPU 互联。

### 24GB 显卡强行跑更大模型

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --cpu-offload-gb 8 \
    --gpu-memory-utilization 0.90
```

假设：

1）GPU 物理显存是 24GB。

2）`--cpu-offload-gb 8` 表示额外把约 8GB 权重放到 CPU 侧。

3）可以近似理解成 `24GB + 8GB = 32GB` 的可用空间。

注意：速度可能下降，因为每次 forward（前向计算）可能涉及 CPU 到 GPU 的数据访问。

面试口吻：CPU offload 是用速度换显存，不是免费扩容。

## 11. 生成配置：`--generation-config`

`--generation-config auto` 默认会从模型路径加载 generation config（生成配置）；如果设置成 `vllm`，就不用模型自带生成配置，而使用 vLLM 默认生成参数。

### 使用 vLLM 默认生成配置

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --generation-config vllm
```

含义：

1）不使用 Hugging Face 模型仓库里的默认生成配置。

2）避免模型自带配置悄悄影响 temperature、top_p、max_new_tokens 等生成行为。

面试口吻：线上效果不一致时，要检查是不是 generation config 在影响默认采样参数。

## 12. Attention 后端：`--attention-backend`

`--attention-backend` 用来指定 attention（注意力计算）后端；如果是 `auto` 或不设置，vLLM 会自动选择。

### 手动指定 FlashAttention

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --attention-backend FLASH_ATTN
```

含义：

1）强制使用 `FLASH_ATTN`。

2）适合你在 benchmark（性能测试）时对比不同后端。

3）普通部署通常先不手动指定。

面试口吻：attention backend 一般交给 vLLM 自动选，只有调性能时才手动改。

## 13. 日志：`--enable-log-requests` 和 `--max-log-len`

`--enable-log-requests` 会记录请求信息；INFO 级别记录请求 ID 和参数，DEBUG 级别可能记录 prompt 输入，比如文本或 token IDs。

### 开启请求日志

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-log-requests \
    --max-log-len 200
```

含义：

1）记录请求基本信息，方便排查问题。

2）`--max-log-len 200` 控制日志里最多记录 200 长度内容。

3）生产环境要注意隐私，因为 prompt 可能包含敏感信息。

面试口吻：日志方便排错，但生产环境不能随便打印用户完整 prompt。

## 14. 综合示例：小模型单卡服务

适合：1 张 24GB GPU，Qwen 1.5B，给内网同事测试。

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key sk-test-123 \
    --served-model-name qwen-1.5b \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192
```

解释：

1）`--host 0.0.0.0`：允许其他机器访问。

2）`--port 8000`：API 地址是 `http://服务器IP:8000`。

3）`--api-key sk-test-123`：调用时必须带 key。

4）`--served-model-name qwen-1.5b`：客户端 model 字段写 `qwen-1.5b`。

5）`--max-model-len 8192`：单次请求输入加输出最多 8192 token。

6）`--gpu-memory-utilization 0.90`：最多使用约 90% GPU 显存。

7）`--max-num-seqs 32`：一轮最多处理 32 条序列。

8）`--max-num-batched-tokens 8192`：一轮最多处理 8192 token。

面试口吻：这是一个比较标准的单卡内网测试配置，兼顾安全、上下文和吞吐。

## 15. 综合示例：7B 模型双卡服务

适合：2 张 GPU，模型较大，单卡显存可能不够。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key sk-prod-456 \
    --served-model-name qwen-7b \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384
```

解释：

1）`--tensor-parallel-size 2`：把模型切到 2 张 GPU 上。

2）`--gpu-memory-utilization 0.88`：每张 GPU 使用约 88% 显存，留一点空间防止 OOM。

3）`--max-num-seqs 64`：并发能力比 32 更强。

4）`--max-num-batched-tokens 16384`：单轮 token 预算更大，吞吐更高，但更吃显存。

面试口吻：7B 以上模型经常需要多卡，核心参数就是 TP、上下文长度和显存比例。

## 16. 综合示例：显存紧张配置

适合：启动时报 OOM，或者 GPU 显存不够。

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096
```

解释：

1）`--max-model-len 4096`：降低上下文长度，减少 KV cache 占用。

2）`--gpu-memory-utilization 0.80`：减少 vLLM 目标显存占用。

3）`--max-num-seqs 16`：减少并发序列数。

4）`--max-num-batched-tokens 4096`：减少单轮 token 处理量。

面试口吻：显存不够时，先降上下文，再降并发和 batched tokens，最后考虑 CPU offload 或量化。

