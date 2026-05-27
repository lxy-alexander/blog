---
title: "sglang Installation"
published: 2026-05-24
description: "sglang Installation"
image: ""
tags: ["sglang","sglang Installation"]
category: sglang
draft: false
lang: ""
createdAt: "2026-05-24T13:19:09.294.680813135Z"
---



apptainer pull sglang_latest.sif docker://docker.io/lmsysorg/sglang:latest

```
SGL_PROJECT_PATH="/data/home/xli49/sglang"


export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
cd "$SGL_PROJECT_PATH"

apptainer shell --nv \
  --bind "${SGL_PROJECT_PATH}:${SGL_PROJECT_PATH}" \
  --bind "${HOME}/.cache/huggingface:${HOME}/.cache/huggingface" \
  --pwd "${SGL_PROJECT_PATH}" \
  --env HF_TOKEN="" \
  ../container/sglang_latest.sif
```

