---
title: "Custom Command"
published: 2026-02-05
description: "Custom Command"
image: ""
tags: ["tools","Custom Command"]
category: tools
draft: false
lang: ""
---

# 1）先写好脚本 `sr`

```bash
#!/bin/bash

# ===== 参数检查 =====
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  sr <gpu_type> [node_id]"
    echo ""
    echo "Examples:"
    echo "  sr h100"
    echo "  sr h100 007"
    echo "  sr a100"
    exit 1
fi

GPU_TYPE="$1"
PARTITION="gpucompute-$GPU_TYPE"
NODE_ARG=""

# ===== 可选节点 =====
if [ -n "$2" ]; then
    NODE_ARG="--nodelist=ghpc$2"
fi

# ===== 启动 =====
srun \
  --gpus-per-node=1 \
  --cpus-per-gpu=4 \
  $NODE_ARG \
  --partition=$PARTITION \
  --time=12:00:00 \
  --pty /bin/bash
```

------

# 2）真正的 `sr` 命令

把脚本放到 `~/bin`

```bash
mkdir -p ~/bin
mv sr ~/bin/
chmod +x ~/bin/sr
```

确保 `~/bin` 在 PATH 里：

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

# 3）使用

```
sr h100
sr h100 007
sr a100
```

