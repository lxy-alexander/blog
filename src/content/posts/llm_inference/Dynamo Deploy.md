---
title: "Dynamo Deploy"
published: 2026-08-24
description: "Dynamo Deploy"
image: ""
tags: ["llm_inference","Dynamo Deploy"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-24T17:36:12.980.320156671Z"
---



## Docker运行

如果你说的是 NVIDIA Dynamo，那么你这台 1× RTX A4000（16GB）完全可以先跑一个单 GPU 的 Dynamo + vLLM 示例。A4000 属于 Ampere，当前 Dynamo 1.4.0 明确支持 Ampere。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/reference/compatibility?utm_source=chatgpt.com))

1）先确认驱动

```bash
nvidia-smi
```

Dynamo 1.4.0 当前要求 CUDA 12 路径的 NVIDIA Driver 至少 `575.51.03`，CUDA 13 路径至少 `580.00.03`。([NVIDIA Docs](https://docs.nvidia.com/dynamo/zh-CN/cli/installation/install-dynamo?utm_source=chatgpt.com))

2）直接启动官方 vLLM 容器

```bash
docker run \
  --gpus all \              # 把宿主机上的所有 NVIDIA GPU 暴露给容器
  --network host \          # 容器直接使用宿主机网络，例如容器监听 8000 端口时，宿主机可直接访问 localhost:8000
  --rm \                    # 容器退出后自动删除，避免留下 stopped container
  -it \                     # -i 保持标准输入，-t 分配终端，方便进入容器交互操作
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0 \
                            # NVIDIA NGC 上的 Dynamo vLLM Runtime 镜像，版本为 1.4.0
    /bin/bash
```

这是 NVIDIA 当前 1.4.0 官方 runtime。([NVIDIA Docs](https://docs.nvidia.com/dynamo/zh-CN/cli/installation/install-dynamo?utm_source=chatgpt.com))

3）容器里面启动 Dynamo frontend

为了你这种单机单卡环境，直接用 `file discovery`，不需要 etcd / NATS：

```bash
python3 -m dynamo.frontend \
  --discovery-backend file \
  > /tmp/frontend.log 2>&1 &
```

```python
接收 HTTP / OpenAI API 请求
↓
找到后端 worker
↓
把请求转发过去
↓
把结果返回给客户端
```

```python
[1] 10
dynamo@spheron-a4000-pcie-1x-1787606950772-s3cleq:/workspace$ cat /tmp/frontend.log
2026-08-24T21:46:36.215519Z  INFO main._raise_fd_limit: Raised RLIMIT_NOFILE soft limit 1024 -> 8192 (hard=524288)
2026-08-24T21:46:36.217766Z  INFO router_args.log_rejection_thresholds: busy-worker rejection disabled: no rejection threshold is configured
2026-08-24T21:46:36.217809Z  INFO main.async_main: Request migration disabled (limit: 0)
2026-08-24T21:46:36.218766Z  INFO dynamo_runtime::distributed: Initializing KV store discovery backend: File(/tmp/dynamo_store_kv)
2026-08-24T21:46:36.219042Z  INFO dynamo_runtime::pipeline::network::manager: Initializing NetworkManager with TCP request plane mode=tcp host=10.0.0.167 port=OS-assigned
2026-08-24T21:46:36.221928Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP(S) service protocol="HTTP" address="0.0.0.0:8000"

这说明命令已经成功放到后台运行了。
1）[1] 表示这是当前 shell 里的第 1 个后台任务。
2）10 表示这个后台进程的 PID 是 10。
```

官方也推荐这种方式作为单机最轻量的测试方式。([NVIDIA Docs](https://docs.nvidia.com/dynamo/zh-CN/cli/installation/install-dynamo?utm_source=chatgpt.com))

4）启动一个 vLLM worker

先不要上 7B，先用官方默认的 `Qwen3-0.6B` 把整个链路跑通：

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file
```

官方给出的这个模型大约只需要 2GB 显存，所以 A4000 跑这个很轻松。([NVIDIA Docs](https://docs.nvidia.com/dynamo/zh-CN/cli/installation/install-dynamo?utm_source=chatgpt.com))

```text
加载 Qwen3-0.6B
↓
占用 A4000 显存
↓
真正执行推理
↓
生成 token
```

5）另开一个终端测试

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "你好，介绍一下你自己"
      }
    ],
    "max_tokens": 128
  }'
```

```python
curl
  ↓
Frontend
  ↓
vLLM Worker
  ↓
Qwen3-0.6B
  ↓
A4000
  ↓
结果返回
```

```python
为什么两边都有： --discovery-backend file

因为它们要用同一种方式互相发现。

Frontend：我去哪里找 worker？
答案：file discovery

Worker：我向哪里注册自己？
答案：file discovery

如果 frontend 用 file，worker 也应该用 file。
```



这个 `/v1/chat/completions` 是 OpenAI-compatible API。([NVIDIA Docs](https://docs.nvidia.com/dynamo/cli/getting-started/quickstart))

如果你想直接看流式：

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "用中文解释 Dynamo 是做什么的"
      }
    ],
    "max_tokens": 256,
    "stream": true
  }' z
```

对于你的 1×A4000，我建议现阶段先这么配：

```21text
 08asdfghjk df
```

7B FP16/BF16 已经处于 A4000 16GB 的边缘，官方估算 7B 大约要 `14–16GB`，还得给 KV Cache 和运行时留显存，所以不建议把 7B FP16 当第一个例子。([NVIDIA Docs](https://docs.nvidia.com/dynamo/zh-CN/cli/installation/install-dynamo?utm_source=chatgpt.com))

如果你就是想测试稍微真实一点的模型，可以第二步改成例如：

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-4B \
  --discovery-backend file \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

A4000 上我会优先从 `Qwen3-4B` 这个级别开始，而不是直接 8B。

另外，你现在只有一张 GPU，所以 Dynamo 最核心的 `prefill/decode disaggregation` 暂时体现不出优势。官方 vLLM 示例中，普通 aggregated serving 只需要 1 GPU，而 disaggregated serving 至少需要 2 GPU，带 KV routing 的 disaggregated 示例则需要更多 GPU。([NVIDIA Docs](https://docs.nvidia.com/dynamo/v-0-9-1/components/backends/v-llm?utm_source=chatgpt.com))

所以你的机器目前最值得实验的是：

```text
第一阶段
1×A4000
Dynamo frontend
      +
1×vLLM worker
      +
OpenAI API

第二阶段
测试并发
测试 batching
测试不同模型
测试 KV cache / prefix reuse

以后有 2×GPU
再测试 Prefill GPU
        |
        Dynamo
        |
Decode GPU
```

如果你愿意，我下一步可以直接给你一份“1×A4000 + Ubuntu + Docker + Dynamo 1.4 + Qwen3-4B”的完整可复制运行脚本，包括 `docker run`、模型缓存目录、显存参数、curl benchmark 和 Python OpenAI Client。







## 源码编译运行

可以。你现在已经完成了虚拟环境和 `maturin` 安装。下面给你从“干净 Ubuntu”到“源码 Dynamo + vLLM + Qwen3-0.6B 跑通”的完整流程。当前官方源码构建指南就是这条路线。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

1）先检查机器环境

你的 A4000 属于 Ampere，Dynamo 支持。当前本地环境要求 Ubuntu 22.04/24.04、Python 3.10 或 3.12；CUDA 12.9+ 或 13.0+。([NVIDIA Docs](https://docs.nvidia.com/dynamo/cli/installation/install-dynamo?utm_source=chatgpt.com))

```bash
nvidia-smi
python3 --version
```

如果装了 CUDA Toolkit：

```bash
nvcc --version
```

2）安装系统编译依赖

```bash
sudo apt update

sudo apt install -y \
  git \
  curl \
  build-essential \
  libhwloc-dev \
  libudev-dev \
  pkg-config \
  libclang-dev \
  protobuf-compiler \
  python3-dev \
  cmake
```

这是官方源码构建列出的 Ubuntu 依赖。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

其中大概可以理解为：

```text
build-essential   C/C++ 编译环境
cmake             native 项目构建工具
libclang-dev      clang 开发库
protobuf-compiler protobuf 编译器
python3-dev       Python native extension 编译
libhwloc-dev      CPU/GPU 拓扑相关
libudev-dev       Linux 设备相关
```

3）安装 Rust

Dynamo 的 runtime 很大一部分是 Rust，所以必须安装：

```bash
curl --proto '=https' \
  --tlsv1.2 \
  -sSf https://sh.rustup.rs | sh
```

然后：

```bash
source $HOME/.cargo/env
```

检查：

```bash
rustc --version
cargo --version
```

官方构建同样要求 Rust。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

4）拉 Dynamo 最新源码

如果你就是想用 `main`，不用 checkout：

```bash
cd ~
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
```

检查当前分支：

```bash
git branch --show-current
```

应该是：

```text
main
```

记录当前源码版本：

```bash
git log -1 --oneline
```

建议记住这个 commit。以后如果编译出问题，就知道到底是哪一版源码。

`main` 是开发分支；源码构建适合测试最新开发功能和修改 Dynamo。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

5）安装 uv

如果没有：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

检查：

```bash
uv --version
```

你现在已经完成这一步。

6）创建 Python 虚拟环境

在：

```text
~/dynamo
```

执行：

```bash
uv venv .venv
```

激活：

```bash
source .venv/bin/activate
```

看到：

```text
(.venv) ubuntu@xxx:~/dynamo$
```

说明成功。

建议再确认：

```bash
which python3
```

应该指向类似：

```text
/home/ubuntu/dynamo/.venv/bin/python3
```

7）安装编译工具

你已经完成：

```bash
uv pip install pip 'maturin[patchelf]'
```

你得到：

```text
maturin 1.15.0
patchelf 0.19.1.0
pip 26.2.1
```

这一步已经成功。

其中：==`maturin` 是专门给“Rust + Python”项目用的构建工具。==

```text
maturin
    ↓
编译 Rust
    ↓
生成 Python native binding
    ↓
安装到当前 .venv
    ↓
Python 可以 import
```

官方当前也是使用 `maturin[patchelf]`。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

8）开始真正编译 Dynamo

你现在下一步就是这个。

进入：==这个目录就是 Dynamo 的“Rust → Python bindings”构建目录。==

```bash
cd ~/dynamo/lib/bindings/python
```

执行：==把 Dynamo 这一部分的 Rust 代码编译成 Python 能直接导入的本地扩展，并安装进你当前的 `.venv` 开发环境里。== develop 表示“开发模式构建并安装”。--uv 按 `uv` 环境来安装构建结果

```bash
maturin develop --uv
```

这是整个源码构建最关键的一步。

它相当于：

```text
Dynamo Rust 源码
      ↓
Cargo
      ↓
rustc
      ↓
生成 native library
      ↓
maturin
      ↓
生成 Python binding
      ↓
安装进 .venv
```

官方源码构建也是这一条。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

编译过程中会看到大量类似：

```text
Compiling ...
Compiling ...
Compiling ...
Finished ...
```

第一次编译比较多是正常的。

9）编译完返回项目根目录

不要手动猜路径，直接：

```bash
cd "$(git rev-parse --show-toplevel)"
```

确认：

```bash
pwd
```

应该是：

```text
/home/ubuntu/dynamo
```

10）安装 GPU Memory Service

执行：

```bash
uv pip install -e lib/gpu_memory_service
```

```python
把 ~/dynamo/lib/gpu_memory_service
注册到当前 .venv
↓
Python 可以直接 import
↓
以后你修改这个目录里的 Python 源码
↓
通常不用重新 pip install
```

官方源码构建流程要求这一步。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

11）安装 Dynamo + vLLM

因为你的目的不是只跑 frontend，而是继续使用：

```bash
python3 -m dynamo.vllm
```

所以直接安装 vLLM extra：

```bash
uv pip install -e '.[vllm]'
```

这一点很重要。

只执行：

```bash
uv pip install -e .
```

主要安装：

```text
Dynamo runtime
+
Dynamo frontend
```

而：

```bash
uv pip install -e '.[vllm]'
```

则会再安装 vLLM backend 所需依赖。官方构建文档明确说明 backend extra 用 `[vllm]` 或 `[sglang]`。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

这一步可能下载很多 CUDA/PyTorch/vLLM 相关包。

12）验证 Dynamo 编译安装成功

先：

```bash
python3 -m dynamo.frontend --help
```

如果出现帮助：

```text
usage: ...
options:
...
```

说明：

```text
Python
+
Dynamo Rust binding
+
Frontend
```

基本正常。

这是官方建议的验证方法。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

再检查 vLLM：

```bash
python3 -m dynamo.vllm --help
```

如果也能显示帮助：

```text
Dynamo
+
vLLM backend
```

基本都装好了。

13）先启动 Frontend

你是一台机器一张 A4000，继续使用 `file discovery`：

```bash
python3 -m dynamo.frontend \
  --discovery-backend file \
  > /tmp/frontend.log 2>&1 &
```

你会看到类似：

```text
[1] 12345
```

检查日志：

```bash
cat /tmp/frontend.log
```

或者：

```bash
tail -f /tmp/frontend.log
```

检查 frontend：

```bash
curl http://localhost:8000/health
```

在 worker 还没启动前，可能只有 frontend 自己健康，但还没有 generate backend。

14）启动 vLLM Worker

保持 `.venv` 激活。

执行：

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file
```

这时候：

```text
Qwen3-0.6B
↓
vLLM
↓
CUDA
↓
A4000
```

模型第一次没有缓存的话会下载。

成功时重点找：

```text
VllmWorker for Qwen/Qwen3-0.6B has been initialized
```

以及：

```text
Registered endpoint 'dynamo.backend.generate'
```

15）确认 A4000 真正在工作

另外开一个 SSH 终端：

```bash
nvidia-smi
```

应该看到 Python/vLLM 占用 GPU 显存。

然后：

```bash
curl http://localhost:8000/health
```

如果和你 Docker 那次一样看到：

```json
{
  "status": "healthy",
  "endpoints": [
    "dyn://dynamo.backend.generate"
  ]
}
```

并且存在：

```text
device_type: cuda
```

就说明：

```text
Frontend
↓
发现 Worker
↓
Worker 使用 CUDA
↓
A4000 正常
```

16）最后测试非流式请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己"
      }
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

如果返回完整 JSON，整个源码环境就跑通了。

17）测试流式

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "用中文解释 NVIDIA Dynamo 是做什么的"
      }
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

应该看到：

```text
data: {..."delta":{"content":"..."}} 

data: {..."delta":{"content":"..."}} 

data: {..."delta":{"content":"..."}} 
```

说明流式也正常。

18）你的完整运行架构

最后实际变成：

```text
Ubuntu
│
├── ~/dynamo
│      │
│      ├── Dynamo Python 源码
│      │
│      ├── Dynamo Rust 源码
│      │
│      └── .venv
│
├── Dynamo Frontend
│        │
│        │ file discovery
│        ↓
├── Dynamo vLLM Worker
│        │
│        ↓
├── Qwen3-0.6B
│        │
│        ↓
└── RTX A4000
```

请求流程：

```text
curl
↓
localhost:8000
↓
Dynamo Frontend
↓
dynamo.backend.generate
↓
Dynamo vLLM Worker
↓
vLLM
↓
Qwen3-0.6B
↓
CUDA
↓
A4000
```

19）以后修改 Rust 源码怎么办

如果你修改的是 Dynamo Rust binding 相关源码，需要重新：

```bash
cd ~/dynamo/lib/bindings/python
maturin develop --uv
```

然后重启服务。

如果只是想检查 Rust 能不能编译：

```bash
cd ~/dynamo
cargo build
```

整个 Rust workspace 都会编译。Dynamo 仓库当前也给出了 `cargo build` 作为 Rust-only 构建方式。([GitHub](https://github.com/ai-dynamo/dynamo/blob/main/AGENTS.md?utm_source=chatgpt.com))

20）一个值得提前记住的问题

如果你本地是 CUDA 13，之后启动 vLLM 时出现 FlashInfer JIT / CUDA headers 版本不匹配，当前官方源码构建文档给出的 workaround 是：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

然后再：

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --discovery-backend file
```

官方文档目前专门记录了这个 CUDA 13 源码安装问题。([NVIDIA Docs](https://docs.nvidia.com/dynamo/dev/getting-started/building-from-source?utm_source=chatgpt.com))

你现在已经完成到第 7 步，所以此刻不用从头重做。直接继续：

```bash
cd ~/dynamo/lib/bindings/python

maturin develop --uv
```

这就是你当前真正进入“编译 Dynamo 源码”的下一步。

