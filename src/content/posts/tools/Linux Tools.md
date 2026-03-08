---
title: "Linux Tools"
published: 2026-02-07
description: "Linux Tools"
image: ""
tags: ["tools","Linux Tools"]
category: tools
draft: false
lang: ""
---

# **I. HPC Setup Without `sudo` — CUDA, CMake & `.bashrc` Configuration**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> This guide covers three tasks on HPC clusters where you have no root access: installing the <strong>CUDA Toolkit</strong> to your home directory without <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">sudo</code>; installing <strong>CMake</strong> locally from a pre-built binary; and structuring your <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">~/.bashrc</code> with a modular <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">~/.bashrc.d/</code> system for managing CUDA versions, libtorch, cuDNN, and custom paths.
</div>

---

## 1. Installing CUDA Without `sudo`

CUDA only truly requires two things: the <span style="color:#E8600A;font-weight:700">nvcc compiler</span> and user-space libraries like <span style="color:#E8600A;font-weight:700">libcudart</span>. Both can be installed entirely inside your home directory:

```
$HOME/cuda
```

### 1) Check Your Linux Distribution and Architecture

```bash
cat /etc/os-release
```

```
NAME="Rocky Linux"
VERSION="9.7 (Blue Onyx)"
ID="rocky"
...
```

```bash
uname -m
```

```
x86_64
```

### 2) Download the Runfile from NVIDIA

Go to: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Select your OS/architecture and choose the <span style="color:#E8600A;font-weight:700">runfile (local)</span> installer type.

![CUDA download page](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260207201609158)

The downloaded file will be named something like:

```
cuda_12.9.1_575.57.08_linux.run
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> To install an older CUDA version, visit the archive at <a href="https://developer.nvidia.com/cuda-toolkit-archive">https://developer.nvidia.com/cuda-toolkit-archive</a>. Make sure the CUDA version you choose is <strong>less than or equal to</strong> the CUDA Driver version reported by <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nvidia-smi</code>.</div>

### 3) Install the Toolkit Only (No Driver)

Make the runfile executable:

```bash
chmod +x cuda_*.run
```

Run the installer with driver installation disabled:

```bash
./cuda_12.9*.run \
  --silent \
  --toolkit \
  --toolkitpath=$HOME/cuda-12.9 \
  --no-drm \
  --no-man-page
```

| Flag | Purpose |
| --- | --- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--silent</code> | Non-interactive installation |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--toolkit</code> | Install CUDA Toolkit only |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--toolkitpath</code> | Target installation directory (user home) |
| No driver flag | Avoids any root requirement |

### 4) Configure Environment Variables (Modular `.bashrc.d`)

Create the directory:

```bash
mkdir -p ~/.bashrc.d
```

Create a CUDA config file:

```bash
nano ~/.bashrc.d/cuda.sh
```

Write the following:

```bash
# ===== Default CUDA =====
export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ===== CUDA version switcher =====
use_cuda () {
    local ver=$1

    if [ ! -d "$HOME/cuda-$ver" ]; then
        echo "CUDA $ver not found in \$HOME"
        return 1
    fi

    export CUDA_HOME=$HOME/cuda-$ver
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    echo "Switched to CUDA $ver"
    nvcc --version | head -n 1
}
```

Ensure `~/.bashrc` loads all files in `~/.bashrc.d/` (add if not present):

```bash
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        [ -f "$rc" ] && . "$rc"
    done
fi
```

Reload the environment:

```bash
source ~/.bashrc
```

### 5) Verify the Installation

Check the compiler:

```bash
nvcc -V
```

If a CUDA version string is printed, the toolkit is installed correctly.

Check GPU availability:

```bash
nvidia-smi
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">This is a critical and often overlooked distinction.</span> If <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nvidia-smi</code> runs successfully, the server already has a GPU driver installed and you can use the GPU. If it fails, the GPU driver is missing — without <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sudo</code> you cannot install the driver yourself, meaning CUDA can only be used for compilation, not for GPU execution.</div>

---

## 2. Installing CMake Locally (No Root)

### 1) Download the Official Pre-built Installer

The official binary installer requires no source compilation, no `gcc` or `make`, installs quickly, and is compatible with most Linux environments.

```bash
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-x86_64.sh
```

### 2) Install to Your User Directory

```bash
bash cmake-3.29.6-linux-x86_64.sh --skip-license --prefix=$HOME/.local
```

| Flag | Purpose |
| --- | --- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--skip-license</code> | Skip the interactive license confirmation |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--prefix=$HOME/.local</code> | Install into the user-level software directory (Linux convention) |

### 3) Add to `PATH` and Verify

```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cmake --version
```

---

## 3. `.bash_profile` — Auto-load `.bashrc` on Login

```bash
# .bash_profile

# Load aliases and functions from .bashrc
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
```

This ensures `.bashrc` is sourced automatically on every SSH login session.

---

## 4. `.bashrc` Section-by-Section Walkthrough

### 1) System Initialization

Loads the system-level bash configuration (modules, colors, completions, etc.):

```bash
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi
```

### 2) User `PATH` Initialization

Adds user program directories to `PATH` without creating duplicates:

```bash
# Only add if not already present (prevents duplicate PATH entries)
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    # Prepend user-level bin dirs so locally installed tools take priority
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi

# Export so child processes (Python, bash, etc.) inherit this PATH
export PATH
```

### 3) The `.bashrc.d` Modular Config System

Splits shell configuration into separate, focused files. <span style="color:#2980B9">Recommended when managing</span> multiple CUDA versions, multiple Conda environments, multi-project research, or many custom aliases.

```bash
# Load all user config modules from ~/.bashrc.d/
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        # Only source regular files (not directories or other types)
        if [ -f "$rc" ]; then
            # Source the file — equivalent to: source "$rc"
            # Makes aliases, functions, and exports take effect immediately
            . "$rc"
        fi
    done
fi
```

### 4) PATH and Library Configuration

#### CUDA 12.6 Environment

```bash
# ===== CUDA 12.6 =====
export CUDA_HOME=$HOME/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Sets the default CUDA version to **12.6**.

#### libtorch Headers and Library Paths

```bash
# ===== libtorch — PyTorch's official C++ API and runtime =====
export CPATH=$HOME/libtorch/include:$HOME/libtorch/include/torch/csrc/api/include:$CPATH
export LIBRARY_PATH=$HOME/libtorch/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH
```

Enables calling PyTorch from C++.

| Use Case | Keep? |
| --- | --- |
| Writing C++/CUDA code with libtorch | ✔ Required |
| Python-only PyTorch usage | ❌ Can be removed |

#### cuDNN Paths

Only needed for C++ builds, custom CUDA kernels, or TensorRT:

```bash
# ===== cuDNN =====
export CPATH=$HOME/cudnn/include:$CPATH
export LIBRARY_PATH=$HOME/cudnn/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/cudnn/lib:$LD_LIBRARY_PATH
```

#### CUTLASS

```bash
# ===== CUTLASS =====
export CUTLASS=$HOME/cutlass
```

#### Custom Command Paths

```bash
# ===== PATH: tells the shell where to find executables =====
export PATH=$HOME/.local/bin:$PATH
export PATH="$HOME/bin:$PATH"
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> On a no-root HPC cluster: install CUDA with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--toolkitpath=$HOME/cuda-X.Y --no-drm</code>, install CMake with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">--prefix=$HOME/.local</code>, and keep your shell config clean by splitting everything into <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/.bashrc.d/</code> modules — remember that CUDA without a GPU driver can compile but not run.</div>