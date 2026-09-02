---
title: "voxtral Bug"
published: 2026-08-30
description: "voxtral Bug"
image: ""
tags: ["llm_inference","vllm","voxtral Bug"]
category: llm_inference / vllm
draft: false
lang: ""
createdAt: "2026-08-30T19:25:20.440.426150676Z"
---

下面是重新整理后的完整对比测试流程。只测试普通 greedy，不测试 Beam Search，也不设置 `max_completion_tokens`。

测试三种音频：

1.  英文原音频
2.  与原音频完全等长的全静音
3.  原音频 + 与原音频等长的尾部静音

Hugging Face 官方实现会根据音频长度自动限制生成长度；测试目标是比较 vLLM 是否也会在音频 embedding 边界停止。[Hugging Face VoxtralRealtime 文档](https://huggingface.co/docs/transformers/model_doc/voxtral_realtime)

------

## 1. 准备目录

```
cd /data/home/xli49/lxy/vllm

export TEST_ROOT=/data/home/xli49/lxy/voxtral-audio-tests
export RESULT_ROOT="$TEST_ROOT/results"

mkdir -p \
  "$TEST_ROOT/original" \
  "$TEST_ROOT/derived" \
  "$RESULT_ROOT"
```

确认变量：

```
echo "$TEST_ROOT"
echo "$RESULT_ROOT"
```

------

## 2. 下载新的测试音频

使用 vLLM 官方公开测试资源，不再使用已经返回 404 的临时链接。

```
curl -fL \
  --retry 3 \
  --retry-delay 2 \
  -o "$TEST_ROOT/original/en-source.ogg" \
  "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
```

验证文件：

```
ls -lh "$TEST_ROOT/original/en-source.ogg"
sha256sum "$TEST_ROOT/original/en-source.ogg"
```

------

## 3. 生成三份 WAV 测试数据

### 3.1 原音频

统一转换成 16 kHz、单声道、16-bit PCM WAV：

```
ffmpeg -y \
  -i "$TEST_ROOT/original/en-source.ogg" \
  -map 0:a:0 \
  -ar 16000 \
  -ac 1 \
  -c:a pcm_s16le \
  "$TEST_ROOT/derived/en-original.wav"
```

### 3.2 与原音频等长的静音

`volume=0` 保留原音频的采样点数量，因此静音文件与原音频等长：

```
ffmpeg -y \
  -i "$TEST_ROOT/derived/en-original.wav" \
  -af "volume=0" \
  -ar 16000 \
  -ac 1 \
  -c:a pcm_s16le \
  "$TEST_ROOT/derived/en-silence-same-duration.wav"
```

### 3.3 原音频加等长尾部静音

```
ffmpeg -y \
  -i "$TEST_ROOT/derived/en-original.wav" \
  -i "$TEST_ROOT/derived/en-silence-same-duration.wav" \
  -filter_complex \
  "[0:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a0];\
[1:a]aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono[a1];\
[a0][a1]concat=n=2:v=0:a=1[out]" \
  -map "[out]" \
  -c:a pcm_s16le \
  "$TEST_ROOT/derived/en-original-plus-silence.wav"
```

------

## 4. 验证测试文件

### 4.1 检查格式和时长

```
for file in \
  "$TEST_ROOT/derived/en-original.wav" \
  "$TEST_ROOT/derived/en-silence-same-duration.wav" \
  "$TEST_ROOT/derived/en-original-plus-silence.wav"
do
  echo "===== $file ====="
  ffprobe -v error \
    -show_entries stream=codec_name,sample_rate,channels \
    -show_entries format=duration \
    -of default=noprint_wrappers=1 \
    "$file"
done
```

预期：

```
en-original.wav                 = D 秒
en-silence-same-duration.wav    = D 秒
en-original-plus-silence.wav    = 2D 秒
```

允许有非常小的浮点显示误差。

### 4.2 确认静音文件确实静音

```
ffmpeg \
  -hide_banner \
  -i "$TEST_ROOT/derived/en-silence-same-duration.wav" \
  -af "astats=metadata=1:reset=0" \
  -f null - 2>&1 |
grep -E "Peak level dB|RMS level dB"
```

真正的数字零静音通常显示：

```
Peak level dB: -inf
RMS level dB: -inf
```

------

## 5. 记录测试环境

```
cd /data/home/xli49/lxy/vllm

git branch --show-current
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv
.venv/bin/python -c \
  "import torch, transformers, vllm; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('vllm=', vllm.__version__)"
```

测试修复前后时，都保存这一段输出，避免混淆分支、GPU和安装版本。

------

## 6. Hugging Face 基线测试

Hugging Face 测试不设置：

-   `max_length`
-   `max_new_tokens`
-   `max_completion_tokens`

让模型自己根据音频长度决定生成边界。

```
cd /data/home/xli49/lxy/vllm

export AUDIO=/data/home/xli49/lxy/voxtral-audio-tests/derived/en-original.wav

CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import os
import time

import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
AUDIO = os.environ["AUDIO"]

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

audio, sample_rate = sf.read(AUDIO, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

inputs = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt",
)

prompt_tokens = inputs.input_ids.shape[1]
inputs = inputs.to(model.device, dtype=model.dtype)

torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=1,
    )

torch.cuda.synchronize()

generated_ids = outputs[0, prompt_tokens:].cpu().tolist()

print("audio_seconds:", len(audio) / sample_rate)
print("prompt_tokens:", prompt_tokens)
print("generated_tokens:", len(generated_ids))
print("elapsed_seconds:", time.perf_counter() - start)
print(
    "text:",
    processor.decode(generated_ids, skip_special_tokens=True),
)
PY
```

### Hugging Face 预期结果

| 输入          | 预期                                                         |
| ------------- | ------------------------------------------------------------ |
| 原音频        | 返回正常英文转写                                             |
| 等长静音      | 生成多个 blank token，但在音频长度边界停止，清理后的文本为空 |
| 原音频 + 静音 | 返回原音频文本，允许产生较多 blank，但仍在扩展后的音频边界停止 |

关键不是“是否产生 blank token”，而是“是否在音频边界停止”。

------

## 7. 启动 vLLM 服务

在服务终端中运行：

```
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --port 8002 \
  2>&1 | tee voxtral-vllm-before-fix.log
  
  
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --port 8002 \
  2>&1 | tee voxtral-vllm-after-fix.log
```

------

## 8. 测试 vLLM 三种音频

重新设置变量：

```
export TEST_ROOT=/data/home/xli49/lxy/vllm/voxtral-audio-tests
```

### 8.1 原音频

```
TEST_ROOT=/data/home/xli49/lxy/vllm/voxtral-audio-tests
curl \
  -w '\nHTTP=%{http_code} TIME=%{time_total}s\n' \
  http://127.0.0.1:8002/v1/audio/transcriptions \
  -F "file=@$TEST_ROOT/derived/en-original.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "language=en" \
  -F "temperature=0"
```

```python
{"text":" First words I spoke in the original phonograph. A little piece of practical poetry. Mary had a little lamb, it sleeps with quite a flow, and everywhere that Mary went, the lamb was sure to go.","usage":{"type":"duration","seconds":16}}
HTTP=200 TIME=1.368523s
```

```python
{"text":" First words I spoke in the original phonograph. A little piece of practical poetry. Mary had a little lamb, it sleeps with quite a flow, and everywhere that Mary went, the lamb was sure to go.","usage":{"type":"duration","seconds":16}}
HTTP=200 TIME=55.362134s
```

```python
tests/models/multimodal/generation/test_voxtral_realtime.py::test_offline_transcription_reaches_audio_embedding_boundary PASSED  [ 33%]
tests/models/multimodal/generation/test_voxtral_realtime.py::test_voxtral_realtime_forward PASSED                                [ 66%]
tests/models/multimodal/generation/test_voxtral_realtime.py::test_voxtral_realtime_generator PASSED                              [100%]

```



`````
我复现了相同的日志淹没和请求明显变慢， 但没有遇到服务停止响应：

vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --port 8002

TEST_ROOT=/data/home/xli49/lxy/vllm/voxtral-audio-tests
curl \
  -w '\nHTTP=%{http_code} TIME=%{time_total}s\n' \
  http://127.0.0.1:8002/v1/audio/transcriptions \
  -F "file=@$TEST_ROOT/derived/en-original.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "language=en" \
  -F "temperature=0"

修复前
```python
{"text":" First words I spoke in the original phonograph. A little piece of practical poetry. Mary had a little lamb, it sleeps with quite a flow, and everywhere that Mary went, the lamb was sure to go.","usage":{"type":"duration","seconds":16}}
HTTP=200 TIME=55.362134s
```

Voxtral Realtime 的解码位置需要与音频 embedding 按时间对齐。每个位置可能生成文字 token、blank token（ID 32）或 EOS。部分音频上模型没有及时生成 EOS，解码便超过音频 embedding 边界。此后 vLLM 只能补零 embedding，并可能继续生成大量 blank token，产生上述 warning。

修复后
```python
{"text":" First words I spoke in the original phonograph. A little piece of practical poetry. Mary had a little lamb, it sleeps with quite a flow, and everywhere that Mary went, the lamb was sure to go.","usage":{"type":"duration","seconds":16}}
HTTP=200 TIME=1.368523s
```


````

vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --port 8002

TEST_ROOT=/data/home/xli49/lxy/vllm/voxtral-audio-tests
curl \
  -w '\nHTTP=%{http_code} TIME=%{time_total}s\n' \
  http://127.0.0.1:8002/v1/audio/transcriptions \
  -F "file=@$TEST_ROOT/derived/en-original.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "language=en" \
  -F "temperature=0"




同一段 16 秒音频：

```text
修改前：HTTP 200，55.36 秒
修改后：HTTP 200，1.37 秒
```

Voxtral Realtime 的解码位置需要与音频 embedding 按时间对齐。每个位置可能生成文字 token、blank token（ID 32）或 EOS。部分音频上模型没有及时生成 EOS，解码便超过音频 embedding 边界。此后 vLLM 只能补零 embedding，并可能继续生成大量 blank token，产生上述 warning。

warning 本身会带来日志 I/O 开销，但主要性能损失来自无效解码：GPU 不断计算 blank token，直到生成 EOS 或达到通用 `max_model_len`。因此它可能表现为长时间无响应，但我目前不能确认是无限解码。

解决方案是在离线 `/v1/audio/transcriptions` 中，按照当前音频实际提供的 embedding 范围限制生成长度：

```python
max_tokens = audio_end - len(prompt_token_ids)
```

模型仍可通过 EOS 提前停止；如果没有生成 EOS，也会在最后一个有效音频 embedding 处安全结束，避免进入补零 embedding 和 blank-token 循环。该限制仅用于 Voxtral Realtime 的离线 transcription，不影响 `/v1/realtime` 流式路径。
````

`````





### 8.2 与原音频等长的静音

```
curl \
  --silent \
  --show-error \
  --output "$RESULT_ROOT/silence-same-duration.json" \
  --write-out \
  'HTTP=%{http_code} TIME=%{time_total}s SIZE=%{size_download}\n' \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@$TEST_ROOT/derived/en-silence-same-duration.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "language=en" \
  -F "temperature=0"

echo "curl_exit=$?"

if test -s "$RESULT_ROOT/silence-same-duration.json"; then
  cat "$RESULT_ROOT/silence-same-duration.json"
else
  echo "No response body"
fi
```

### 8.3 原音频加等长静音

```
curl \
  --silent \
  --show-error \
  --output "$RESULT_ROOT/original-plus-silence.json" \
  --write-out \
  'HTTP=%{http_code} TIME=%{time_total}s SIZE=%{size_download}\n' \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@$TEST_ROOT/derived/en-original-plus-silence.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "language=en" \
  -F "temperature=0"

echo "curl_exit=$?"

if test -s "$RESULT_ROOT/original-plus-silence.json"; then
  cat "$RESULT_ROOT/original-plus-silence.json"
else
  echo "No response body"
fi
```

这里每条命令只发送一次请求。`timeout 45s` 是客户端保护，不是连续发送请求。

------

## 9. 查看服务日志

```
grep -E \
  "empty multimodal embeddings|chunk 0 completed|cancelled|Running:" \
  /tmp/voxtral-server.log |
tail -n 200
```

如果已经加入详细 token 日志，再查看：

```
grep -E \
  "generated_tokens=|blank_tokens=|blank_completion_span=|last_token_ids=" \
  /tmp/voxtral-server.log |
tail -n 100
```

单独统计越界 warning 数量：

```
(vllm) (vllm-toolchain) [xli49@ghpc010 vllm]$ grep -c \
  "Realtime model received empty multimodal embeddings" \
  voxtral-vllm-before-fix.log
7944


grep -c \
  "Realtime model received empty multimodal embeddings" \
  voxtral-vllm-after-fix.log

(vllm) (vllm-toolchain) [xli49@ghpc010 vllm]$ grep -c \
  "Realtime model received empty multimodal embeddings" \
  voxtral-vllm-after-fix.log
2  # 关键是这个 dummy request 没有音频：这两条日志来自 V2 Model Runner 的启动内核预热，不来自真实音频请求。

(APIServer pid=2992012) INFO:     Started server process [2992012]
(APIServer pid=2992012) INFO:     Waiting for application startup.
(APIServer pid=2992012) INFO:     Application startup complete.
(APIServer pid=2992012) INFO:     127.0.0.1:51772 - "POST /v1/audio/transcriptions HTTP/1.1" 200 OK
(APIServer pid=2992012) INFO 08-30 23:36:28 [loggers.py:310] Engine 000: Avg prompt throughput: 3.9 tokens/s, Avg generation throughput: 21.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=2992012) INFO 08-30 23:36:38 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
```

------





## 10. 如何判断结果

### 未修复版本可能出现

原音频：

```
HTTP=200
```

因为有语音时，模型可能正常生成 EOS。

等长静音：

```
curl_exit=124
No response body
```

服务日志持续出现：

```
Realtime model received empty multimodal embeddings...
```

这表示模型已经越过音频 embedding 边界，但 vLLM 仍允许继续生成 blank token。

如果不使用客户端超时，请求可能一直生成到：

```
max_model_len - prompt_tokens
```

例如：

```
16000 - 39 = 15961
```

这不是数学意义上的无限生成，但在较慢 GPU 上可能持续数分钟，表现得像“挂起”。

### 修复版本应当出现

三种输入都应在与音频长度相符的时间内结束：

-   原音频返回转写；
-   静音返回空文本；
-   原音频加静音返回原语音的文本；
-   不应生成到 15961 tokens；
-   不应在音频 embedding 用完后继续大量打印 empty embedding warning；
-   `/health` 始终可响应。

### 结果记录表

| 测试                      | HF     | vLLM 修复前 | vLLM 修复后 |
| ------------------------- | ------ | ----------- | ----------- |
| 原音频返回时间            | 记录   | 记录        | 记录        |
| 原音频输出文本            | 记录   | 记录        | 记录        |
| 等长静音返回时间          | 记录   | 记录        | 记录        |
| 等长静音 generated tokens | 记录   | 记录        | 记录        |
| 原音频+静音返回时间       | 记录   | 记录        | 记录        |
| empty embedding warnings  | 不适用 | 记录        | 记录        |
| 请求后 `/health`          | 不适用 | 记录        | 记录        |

修复的核心验收条件是：不传 `max_completion_tokens` 时，vLLM 也必须按照 Voxtral Realtime 的音频 token 数量限制 transcription 的生成长度，而不是使用整个 `max_model_len`。





```python

```

