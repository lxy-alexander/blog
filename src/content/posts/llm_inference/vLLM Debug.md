---
title: "vLLM Debug"
published: 2026-07-24
description: "vLLM Debug"
image: ""
tags: ["llm_inference","vLLM Debug"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-07-24T22:29:24.703.450257070Z"
---

# VS Code 调试 vLLM

一句话记忆：终端用 `debugpy` 启动 vLLM，VS Code 再连接 `5678` 端口。

## 1. 启动 vLLM

```bash
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
NCCL_DEBUG=INFO \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    --dtype bfloat16
    
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
NCCL_DEBUG=INFO \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
.venv/bin/python -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --kv-cache-dtype fp8 \
    --dtype bfloat16
```

说明：

```text
--listen 127.0.0.1:5678
# 在本机 5678 端口启动调试服务

--wait-for-client
# 等待 VS Code 连接后再继续运行
```

## 2. VS Code 配置

`.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach debugpy 5678",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "justMyCode": false
        }
    ]
}
```

其中：

```text
request: attach
# 连接已经启动的 Python 进程

justMyCode: false
# 允许进入 vLLM 等第三方库源码
```

## 3. 调试顺序

1）在 vLLM 源码中打断点。

2）终端运行 debugpy 启动命令。

3）确认端口已经监听：

```bash
ss -lntp | grep 5678

# 输出示例：
# LISTEN 0 128 127.0.0.1:5678 ...
```

4）VS Code 选择 `Attach debugpy 5678`。

5）按 `F5` 连接。

## 4. 常用环境变量

```bash
CUDA_LAUNCH_BLOCKING=1
# 让 CUDA 同步执行，便于定位真实报错位置

TORCH_SHOW_CPP_STACKTRACES=1
# 显示 PyTorch C++ 调用栈

NCCL_DEBUG=INFO
# 显示多 GPU 通信日志

VLLM_LOGGING_LEVEL=DEBUG
# 显示 vLLM 调试日志

VLLM_ENABLE_V1_MULTIPROCESSING=0
# 减少 vLLM 多进程，方便断点调试
```

## 5. 常见问题

### 1) `ECONNREFUSED 127.0.0.1:5678`

表示没有程序监听 `5678`：

```bash
ss -lntp | grep 5678
ps -ef | grep '[d]ebugpy'
```

### 2) 终端显示 `MULTILINE`

```text
Ctrl + J
# 执行整条多行命令

Ctrl + C
# 取消当前输入
```

### 3) 断点不命中

可能代码运行在子进程（工作进程）中，尤其是：

```bash
--tensor-parallel-size 2
```

可以打印进程号确认：

```python
import os

print(os.getpid())

# 输出示例：
# 90731
```

## 6. VS Code 调试快捷键

```text
F5
# 继续运行

F10
# 单步跳过

F11
# 单步进入

Shift + F11
# 单步跳出
```
