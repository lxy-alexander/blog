---
title: "Moe Perf"
published: 2026-08-25
description: "Moe Perf"
image: ""
tags: ["llm_inference","vllm","Moe Perf"]
category: llm_inference / vllm
draft: false
lang: ""
createdAt: "2026-08-25T18:19:14.838.606006151Z"
---

# performance degradation 0.26 vs 0.27 for Qwen3.5 122B A10 on 4xrtx 3090

## 启动 vLLM 服务（不使用nsys）

```python
CUDA_VISIBLE_DEVICES=2 vllm serve \
  Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --tensor-parallel-size 1 \
  --quantization moe_wna16 \
  --kv-cache-dtype fp8 \
  --max-model-len 80K \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16K \
  --gpu-memory-utilization 0.95
```

并发测试时将 `--max-concurrency 1` 改成 `8`。

```python
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 256 \
  --num-prompts 32 \
  --max-concurrency 1 \
  --request-rate inf \
  --ignore-eos
```

-   `--request-rate inf`：默认就是 `inf`，可以删除。

-   `--ignore-eos`：建议保留，确保每个请求生成完整的 256 tokens，否则模型可能提前遇到 EOS，无法准确复现 `tg256`。

## nsys补获vLLM

### 启动 nsys + vLLM 服务

`vllm bench serve --profile` 会自动触发 nsys 开始和停止采集，只捕获正式 benchmark，不包含模型加载过程。

```python
CUDA_VISIBLE_DEVICES=2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --output output_file \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8 \
    --max-model-len 80K \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16K \
    --gpu-memory-utilization 0.95 \
    --profiler-config.profiler cuda
```

-   `CUDA_VISIBLE_DEVICES=2`

    只使用第 0 张 GPU。

-   `VLLM_WORKER_MULTIPROC_METHOD=spawn`

    让 vLLM 使用 `spawn` 创建工作进程，避免 `fork` 与 CUDA、nsys 产生兼容问题。

-   `nsys profile`

    使用 NVIDIA Nsight Systems 分析后面的 `vllm serve` 进程。

-   `--output qwen35_h100_c1`

    指定输出文件名，通常生成 `qwen35_h100_c1.nsys-rep`。

-   `--force-overwrite=true`

    如果同名结果文件已经存在，直接覆盖。

-   `--trace=cuda,nvtx,osrt,cublas`

    指定采集内容：

    -   `cuda`：CUDA API、GPU kernel 和显存复制
    -   `nvtx`：PyTorch/vLLM 的代码区间标记
    -   `osrt`：线程同步等操作系统运行时调用
    -   `cublas`：cuBLAS 矩阵运算

-   `--trace-fork-before-exec=true`

    跟踪 vLLM 创建的子进程，避免遗漏 GPU worker 中的 CUDA 活动。

-   `--cuda-graph-trace=node`

    将 CUDA Graph 展开到节点级，可以看到其中的具体 kernel，但采集开销和结果文件会更大。低开销模式可以使用 `graph`。

-   `--capture-range=cudaProfilerApi`

    启动服务时暂不采集，等待程序调用 `cudaProfilerStart()` 后再开始。

-   `--capture-range-end=repeat`

    调用 `cudaProfilerStop()` 后结束当前采集区间，并等待下一次采集。适合在同一个服务上运行多次 benchmark。只采集一次可改为 `stop`。

-   `--profiler-config.profiler cuda`

    启用 vLLM 的 CUDA Profiler 控制，让 `/start_profile` 和 `/stop_profile` 调用 CUDA Profiler API。

-   benchmark 端的 `--profile`

    自动在正式 benchmark 前后调用 `/start_profile` 和 `/stop_profile`，因此不会采集模型加载过程。

-   `\`

    Bash 续行符，表示下一行仍属于同一条命令。



### 执行具体的指令

-   `pp2048 c1`：输入 2048 tokens，并发 1

```python
CUDA_VISIBLE_DEVICES=2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --output ../profiles/qwen35_h100_pp2048_c1_latest \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8 \
    --max-model-len 80K \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16K \
    --gpu-memory-utilization 0.95 \
    --profiler-config.profiler cuda
```

```
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 1 \
  --num-prompts 32 \
  --max-concurrency 1 \
  --ignore-eos \
  --profile
```

-   `pp2048 c8`：输入 2048 tokens，并发 8

```python
CUDA_VISIBLE_DEVICES=2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --output ../profiles/qwen35_h100_pp2048_c8_latest \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8 \
    --max-model-len 80K \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16K \
    --gpu-memory-utilization 0.95 \
    --profiler-config.profiler cuda
```

```
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 1 \
  --num-prompts 32 \
  --max-concurrency 8 \
  --ignore-eos \
  --profile
```

-   `tg256 c1`：生成 256 tokens，并发 1

```python
CUDA_VISIBLE_DEVICES=2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --output ../profiles/qwen35_h100_tg256_c1_latest \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8 \
    --max-model-len 80K \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16K \
    --gpu-memory-utilization 0.95 \
    --profiler-config.profiler cuda
```

```
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 256 \
  --num-prompts 32 \
  --max-concurrency 1 \
  --ignore-eos \
  --profile
```

-   `tg256 c8`：生成 256 tokens，并发 8

```python
CUDA_VISIBLE_DEVICES=2 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
nsys profile \
  --output ../profiles/qwen35_h100_tg256_c8_latest \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  vllm serve Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tensor-parallel-size 1 \
    --quantization moe_wna16 \
    --kv-cache-dtype fp8 \
    --max-model-len 80K \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16K \
    --gpu-memory-utilization 0.95 \
    --profiler-config.profiler cuda
```

```
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 256 \
  --num-prompts 32 \
  --max-concurrency 8 \
  --ignore-eos \
  --profile
```

由于 nsys 使用了 `--capture-range-end=repeat`，同一个服务进程可以连续采集这四次 benchmark。全部完成后，在服务终端按 `Ctrl+C` 生成 `.nsys-rep` 文件。





## v0.26, v0.27, v0.27-latest版本对比实验

可以，三个版本都使用 `git worktree`，能够保持源码、环境和可执行文件完全隔离。

### 1. 创建三个 worktree

在当前仓库执行：

```
cd /data/home/xli49/lxy/vllm

git fetch upstream main --tags

git worktree add --detach ../vllm-v026 v0.26.0
git worktree add --detach ../vllm-v027 v0.27.0
git worktree add --detach ../vllm-latest upstream/main
```

查看结果：

```
git worktree list
```

### 2. 安装 vLLM 0.26.0

```
cd /data/home/xli49/lxy/vllm-v026

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 \
  uv pip install -e . --torch-backend=auto

.venv/bin/vllm --version
git rev-parse HEAD
```

对应程序：

```
/data/home/xli49/lxy/vllm-v026/.venv/bin/vllm
```

### 3. 安装 vLLM 0.27.0

```
cd /data/home/xli49/lxy/vllm-v027

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 \
  uv pip install -e . --torch-backend=auto

.venv/bin/vllm --version
git rev-parse HEAD
```

对应程序：

```
/data/home/xli49/lxy/vllm-v027/.venv/bin/vllm
```

如果要匹配附件中的 `0.27.1`，创建 worktree 时改成：

```
git worktree add --detach ../vllm-v027 v0.27.1
```

### 4. 安装最新 main

```
cd /data/home/xli49/lxy/vllm-latest

uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 \
  uv pip install -e . --torch-backend=auto

.venv/bin/vllm --version
git rev-parse HEAD
```

对应程序：

```
/data/home/xli49/lxy/vllm-latest/.venv/bin/vllm
```

### 5. 对应关系

-   v0.26 服务端：

```
/data/home/xli49/lxy/vllm-v026/.venv/bin/vllm serve ...
```

-   v0.27 服务端：

```
/data/home/xli49/lxy/vllm-v027/.venv/bin/vllm serve ...
```

-   latest 服务端及固定 benchmark 客户端：

```
/data/home/xli49/lxy/vllm-latest/.venv/bin/vllm serve ...
/data/home/xli49/lxy/vllm-latest/.venv/bin/vllm bench serve ...
```

三个 worktree 都有自己的 `.venv`，不会相互覆盖依赖。



### v0.26 源码错误地搭配了 nightly

````python
报错原因：v0.26 源码错误地搭配了 nightly `_moe_C`，两者接口不一致。

重新安装匹配当前 commit 的 precompiled wheel：

```bash
cd /data/home/xli49/lxy/vllm-v026

uv pip uninstall --python .venv/bin/python vllm

git clean -fdX -- vllm/ vllm.egg-info/

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT="$(git rev-parse HEAD)" \
uv pip install \
  --python .venv/bin/python \
  -e . \
  --torch-backend=auto
```

使用独立编译缓存：

```bash
export VLLM_CACHE_ROOT=/data/home/xli49/.cache/vllm-v026
```

然后重新启动服务。如果提示该 commit 没有 precompiled wheel，就只能安装官方 `vllm==0.26.0` wheel 或本地编译。
````



### 安装官方wheel

````python
建议先卸载 editable 版本并清理旧 `.so`，再安装官方 wheel：

```bash
cd /data/home/xli49/lxy/vllm-v026

uv pip uninstall \
  --python .venv/bin/python \
  vllm

git clean -fdX -- vllm/ vllm.egg-info/


uv venv --python 3.12
source .venv/bin/activate

uv pip install \
  --python .venv/bin/python \
  "vllm==0.26.0" \
  --torch-backend=auto
```

验证：

```bash
.venv/bin/vllm --version

uv pip show vllm \
  --python .venv/bin/python
```

`uv pip show` 中不应再出现 `Editable project location`。启动时建议离开源码目录：

```bash
cd /data/home/xli49/lxy

/data/home/xli49/lxy/vllm-v026/.venv/bin/vllm serve ...
```
````





## 测试结果

### v0.26

#### V0.26-pp2048-c1

```python
INFO 08-25 23:32:26 [utils.py:90] Sampling input_len from [2048, 2048] and output_len from [1, 1]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:09<00:00,  3.31it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  9.68      
Total input tokens:                      65536     
Total generated tokens:                  32        
Request throughput (req/s):              3.31      
Output token throughput (tok/s):         3.31      
Peak output token throughput (tok/s):    4.00      
Peak concurrent requests:                5.00      
Total token throughput (tok/s):          6774.21   
---------------Time to First Token----------------
Mean TTFT (ms):                          302.32    
Median TTFT (ms):                        300.19    
P99 TTFT (ms):                           351.28    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
Stopping profiler...
```



#### V0.26-pp2048-c8

```python
INFO 08-25 20:44:38 [utils.py:90] Sampling input_len from [2048, 2048] and output_len from [1, 1]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8
100%|████████████████████████████████████████████████████████| 32/32 [00:11<00:00,  2.82it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  11.35     
Total input tokens:                      65536     
Total generated tokens:                  32        
Request throughput (req/s):              2.82      
Output token throughput (tok/s):         2.82      
Peak output token throughput (tok/s):    8.00      
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          5779.45   
---------------Time to First Token----------------
Mean TTFT (ms):                          2787.49   
Median TTFT (ms):                        1804.56   
P99 TTFT (ms):                           5932.92   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
Stopping profiler...
```

#### V0.26-tg256-c1

```python
INFO 08-25 21:07:04 [utils.py:90] Sampling input_len from [1, 1] and output_len from [256, 256]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [01:18<00:00,  2.47s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  78.97     
Total input tokens:                      32        
Total generated tokens:                  8192      
Request throughput (req/s):              0.41      
Output token throughput (tok/s):         103.73    
Peak output token throughput (tok/s):    105.00    
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          104.14    
---------------Time to First Token----------------
Mean TTFT (ms):                          17.28     
Median TTFT (ms):                        15.48     
P99 TTFT (ms):                           51.47     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.61      
Median TPOT (ms):                        9.61      
P99 TPOT (ms):                           9.63      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.61      
Median ITL (ms):                         9.61      
P99 ITL (ms):                            9.75      
==================================================
Stopping profiler...
```



#### v0.26-tg256-c8

```python
INFO 08-25 21:17:29 [utils.py:90] Sampling input_len from [1, 1] and output_len from [256, 256]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:24<00:00,  1.31it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  24.43     
Total input tokens:                      32        
Total generated tokens:                  8192      
Request throughput (req/s):              1.31      
Output token throughput (tok/s):         335.36    
Peak output token throughput (tok/s):    352.00    
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          336.67    
---------------Time to First Token----------------
Mean TTFT (ms):                          60.94     
Median TTFT (ms):                        54.97     
P99 TTFT (ms):                           94.85     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          23.69     
Median TPOT (ms):                        23.47     
P99 TPOT (ms):                           24.68     
---------------Inter-token Latency----------------
Mean ITL (ms):                           23.69     
Median ITL (ms):                         23.51     
P99 ITL (ms):                            25.95     
==================================================
Stopping profiler...
```



### v0.27

#### V0.27-pp2048-c1

```python
INFO 08-25 23:41:08 [utils.py:90] Sampling input_len from [2048, 2048] and output_len from [1, 1]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:36<00:00,  1.13s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  36.21     
Total input tokens:                      65536     
Total generated tokens:                  32        
Request throughput (req/s):              0.88      
Output token throughput (tok/s):         0.88      
Peak output token throughput (tok/s):    1.00      
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          1810.71   
---------------Time to First Token----------------
Mean TTFT (ms):                          1131.42   
Median TTFT (ms):                        1119.23   
P99 TTFT (ms):                           1383.83   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
Stopping profiler...
[ble: elapsed 78.391s (CPU 54.0%)] vllm bench serve \
```



#### latest-pp2048-c8

```python
INFO 08-25 23:47:45 [utils.py:90] Sampling input_len from [2048, 2048] and output_len from [1, 1]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:28<00:00,  1.11it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  28.72     
Total input tokens:                      65536     
Total generated tokens:                  32        
Request throughput (req/s):              1.11      
Output token throughput (tok/s):         1.11      
Peak output token throughput (tok/s):    7.00      
Peak concurrent requests:                15.00     
Total token throughput (tok/s):          2283.17   
---------------Time to First Token----------------
Mean TTFT (ms):                          7079.73   
Median TTFT (ms):                        6520.61   
P99 TTFT (ms):                           9133.82   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
Stopping profiler...

```

#### latest-tg256-c1

```python
INFO 08-25 23:53:28 [utils.py:90] Sampling input_len from [1, 1] and output_len from [256, 256]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [03:38<00:00,  6.82s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  218.25    
Total input tokens:                      32        
Total generated tokens:                  8192      
Request throughput (req/s):              0.15      
Output token throughput (tok/s):         37.53     
Peak output token throughput (tok/s):    38.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          37.68     
---------------Time to First Token----------------
Mean TTFT (ms):                          34.42     
Median TTFT (ms):                        32.72     
P99 TTFT (ms):                           69.36     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.61     
Median TPOT (ms):                        26.61     
P99 TPOT (ms):                           26.64     
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.61     
Median ITL (ms):                         26.61     
P99 ITL (ms):                            26.77     
==================================================
Stopping profiler...
[ble: elapsed 306.292s (CPU 13.6%)] vllm bench serve \
```

#### v0.26-tg256-c8

```python
INFO 08-26 00:03:20 [utils.py:90] Sampling input_len from [1, 1] and output_len from [256, 256]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [02:21<00:00,  4.42s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  141.41    
Total input tokens:                      32        
Total generated tokens:                  8192      
Request throughput (req/s):              0.23      
Output token throughput (tok/s):         57.93     
Peak output token throughput (tok/s):    72.00     
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          58.16     
---------------Time to First Token----------------
Mean TTFT (ms):                          190.74    
Median TTFT (ms):                        188.94    
P99 TTFT (ms):                           298.47    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          137.85    
Median TPOT (ms):                        135.35    
P99 TPOT (ms):                           149.04    
---------------Inter-token Latency----------------
Mean ITL (ms):                           137.85    
Median ITL (ms):                         136.09    
P99 ITL (ms):                            155.11    
==================================================
Stopping profiler...

```







## PP2048性能下降分析

### 使用triton3.6.0对最新代码进行测试

```python
cd /data/home/xli49/lxy/vllm

uv venv --python 3.12 .venv-triton36

# 1. 安装完整 vLLM
VLLM_USE_PRECOMPILED=1 \
uv pip install \
  --python .venv-triton36/bin/python \
  -e . \
  --torch-backend=auto

# 2. 最后覆盖 Triton
uv pip install \
  --python .venv-triton36/bin/python \
  --no-deps \
  triton==3.6.0
```

#### latest-pp2048-c1

```python
INFO 08-26 03:44:08 [utils.py:90] Sampling input_len from [2048, 2048] and output_len from [1, 1]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:10<00:00,  3.07it/s]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  10.41     
Total input tokens:                      65536     
Total generated tokens:                  32        
Request throughput (req/s):              3.07      
Output token throughput (tok/s):         3.07      
Peak output token throughput (tok/s):    4.00      
Peak concurrent requests:                5.00      
Total token throughput (tok/s):          6296.31   
---------------Time to First Token----------------
Mean TTFT (ms):                          325.27    
Median TTFT (ms):                        306.73    
P99 TTFT (ms):                           726.09    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
Stopping profiler...
```

### latest-tg256-c1

```python
INFO 08-26 03:54:35 [utils.py:90] Sampling input_len from [1, 1] and output_len from [256, 256]
WARNING: vllm bench serve no longer sets temperature==0 (greedy) in requests by default. The default will be determined on the server side and can be model/API specific. For the old behavior, include --temperature=0.
Starting initial single prompt test run...
Skipping endpoint ready check.
Starting main benchmark run...
Starting profiler...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [01:32<00:00,  2.90s/it]
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     32        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  92.83     
Total input tokens:                      32        
Total generated tokens:                  8192      
Request throughput (req/s):              0.34      
Output token throughput (tok/s):         88.25     
Peak output token throughput (tok/s):    89.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          88.59     
---------------Time to First Token----------------
Mean TTFT (ms):                          18.77     
Median TTFT (ms):                        17.10     
P99 TTFT (ms):                           52.01     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.30     
Median TPOT (ms):                        11.30     
P99 TPOT (ms):                           11.33     
---------------Inter-token Latency----------------
Mean ITL (ms):                           11.30     
Median ITL (ms):                         11.30     
P99 ITL (ms):                            11.44     
==================================================
Stopping profiler...

```











````
`max(activation_out_dim, K)` 的原意是：activation 临时结果和最终输出生命周期不重叠，可以复用同一块显存，所以分配两者中较大的容量。

但这里比较“最后一维”是不正确的，因为两个张量的其他维度不同。

### 两个需要共享显存的张量

Activation 输出：

```text
cache2.shape = [M, topk, activation_out_dim]
```

最终 MoE 输出：

```text
output.shape = [M, K]
```

你的数值：

```text
cache2 = [16384, 8, 256]
output = [16384, 3072]
```

它们的元素数量分别是：

```text
cache2:
M × topk × activation_out_dim
= 16384 × 8 × 256

output:
M × K
= 16384 × 3072
```

要共享同一块 storage，正确比较应该是：

```python
max(
    M * topk * activation_out_dim,
    M * K,
)
```

提取共同的 `M`：

```python
M * max(
    topk * activation_out_dim,
    K,
)
```

代入数值：

```text
topk × activation_out_dim = 8 × 256 = 2048
K                         = 3072

max(2048, 3072) = 3072
```

所以共享空间需要：

```text
M × 3072 × 2 bytes
= 96 MiB
```

### 当前代码做了什么

当前代码是：

```python
workspace1 = (M, topk, max(activation_out_dim, K))
```

计算的其实是：

```python
M * topk * max(activation_out_dim, K)
```

代入数值：

```text
M × 8 × max(256, 3072)
= M × 8 × 3072
```

这相当于把最终输出错误地当成：

```text
[M, topk, K]
```

而最终输出真实形状只有：

```text
[M, K]
```

因此无故多乘了一个 `topk=8`。

### 这个写法的历史来源

早期模块化 MoE 在 top-k reduce 移入 `TritonExperts` 之前，W2 输出确实是：

```text
[M, topk, K]
```

因为每个 token 对应每个被选中的 expert 都有一份输出，之后才在外面执行：

```text
top-k 权重乘法 + reduce
```

当时共享 cache1/cache3：

```text
cache1 = [M, topk, N]
cache3 = [M, topk, K]
```

它们前三个维度相同，因此写：

```python
(M, topk, max(N, K))
```

是合理的。

在 PR [#20725](https://github.com/vllm-project/vllm/pull/20725) 中，top-k weight/reduce 被移入 `TritonExperts`：

```python
intermediate_cache3 = [M, topk, K]
ops.moe_sum(intermediate_cache3, output)
output = [M, K]
```

与此同时 workspace 的职责发生变化：

```text
workspace2：
    cache1 [M, topk,N]
    cache3 [M, topk,K]

workspace1/common：
    activation [M,topk,activation_out_dim]
    output     [M,K]
```

但是旧的“按最后一维取 max，并保留 topk”模式被延续下来：

```python
(M, topk, max(activation_out_dim, K))
```

这是一种保守但过大的声明。

### 当前架构下为什么直接去掉 `K`

现在工作区管理器已经单独拿到两个完整形状：

```python
workspace13_shape = (
    M,
    topk,
    activation_out_dim,
)

fused_out_shape = (
    M,
    K,
)
```

然后按完整元素数量比较：

```python
max_shape_size = max(
    prod(workspace13_shape),
    prod(fused_out_shape),
)
```

见 [modular_kernel.py](/data/home/xli49/lxy/vllm/vllm/model_executor/layers/fused_moe/modular_kernel.py:1169)。

因此 `workspace_shapes()` 只需准确描述 activation：

```python
workspace1 = (M, topk, activation_out_dim)
```

后续自然计算：

```python
max(
    M * topk * activation_out_dim,
    M * K,
)
```

所以总结为：

```text
为什么比较 activation_out_dim 和 K：
为了让 activation 和最终输出复用 storage。

为什么现在不正确：
activation 带 topk 维，最终输出不带 topk 维，
不能只比较最后一维后共同乘 topk。

为什么会多 672 MiB：
代码把 [M,K] 错误地按 [M,topk,K] 预留了空间。
```
````

