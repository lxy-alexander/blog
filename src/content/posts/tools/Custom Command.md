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


# **I. `sr` — Custom SLURM Interactive Job Launcher**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">sr</code> is a small wrapper script around <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">srun</code> that lets you launch an interactive GPU session on an HPC cluster with a short, memorable command — optionally targeting a specific node.
</div>

---

## 1. Write the Script

Create a file named <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sr</code> with the following content:

```bash
#!/bin/bash

# ===== Argument check =====
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

# ===== Optional node targeting =====
if [ -n "$2" ]; then
    NODE_ARG="--nodelist=ghpc$2"
fi

# ===== Launch interactive session =====
srun \
  --gpus-per-node=1 \
  --cpus-per-gpu=4 \
  $NODE_ARG \
  --partition=$PARTITION \
  --time=12:00:00 \
  --pty /bin/bash
```

---

## 2. Install the Script

Move the script to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/bin</code> and make it executable:

```bash
mkdir -p ~/bin
mv sr ~/bin/
chmod +x ~/bin/sr
```

Ensure <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/bin</code> is on your `PATH`:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. Usage

```bash
sr h100          # Request any H100 node
sr h100 007      # Request H100 node ghpc007 specifically
sr a100          # Request any A100 node
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Drop <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sr</code> into <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/bin</code>, add <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/bin</code> to your <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">PATH</code>, and replace verbose <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">srun</code> commands with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sr h100</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sr h100 007</code> to spin up an interactive GPU session instantly.</div>