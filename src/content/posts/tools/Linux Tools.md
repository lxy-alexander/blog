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



# 无 sudo 安装 CUDA

不使用用 apt/yum/sudo 只把 **CUDA Toolkit 安装到你自己的 home 目录**

CUDA 实际只需要：

-   nvcc 编译器
-   libcudart 等用户态库

这些都可以安装在：

```
$HOME/cuda
```

------

## 1）官网下载安装包

### 查看自己的linux版本

```
cat /etc/os-release
```

```
[xli49@ghpc005 ~]$ cat /etc/os-release
NAME="Rocky Linux"
VERSION="9.7 (Blue Onyx)"
ID="rocky"
ID_LIKE="rhel centos fedora"
VERSION_ID="9.7"
PLATFORM_ID="platform:el9"
PRETTY_NAME="Rocky Linux 9.7 (Blue Onyx)"
ANSI_COLOR="0;32"
LOGO="fedora-logo-icon"
CPE_NAME="cpe:/o:rocky:rocky:9::baseos"
HOME_URL="https://rockylinux.org/"
VENDOR_NAME="RESF"
VENDOR_URL="https://resf.org/"
BUG_REPORT_URL="https://bugs.rockylinux.org/"
SUPPORT_END="2032-05-31"
ROCKY_SUPPORT_PRODUCT="Rocky-Linux-9"
ROCKY_SUPPORT_PRODUCT_VERSION="9.7"
REDHAT_SUPPORT_PRODUCT="Rocky Linux"
REDHAT_SUPPORT_PRODUCT_VERSION="9.7"
```

````
uname -m
````

```
[xli49@ghpc005 ~]$ uname -m
x86_64
```

### NVIDIA 官网下载 **runfile (local)** 版本

https://developer.nvidia.com/cuda-downloads

![image-20260207201609158](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260207201609158)



下载后会得到类似：

```
cuda_12.9.1_575.57.08_linux.run
```



### 下载其他版本（版本号$\le$CUDA Driver ）

https://developer.nvidia.com/cuda-toolkit-archive





------

## 2）无 sudo 安装 CUDA Toolkit

### 赋权

```bash
chmod +x cuda_*.run
```

### 只安装 toolkit（不要 driver）

必须关闭 driver 安装：

```bash
./cuda_12.9*.run \
  --silent \
  --toolkit \
  --toolkitpath=$HOME/cuda-12.9 \
  --no-drm \
  --no-man-page
```

参数说明：

| 参数            | 作用                |
| --------------- | ------------------- |
| `--silent`      | 静默安装            |
| `--toolkit`     | 只安装 CUDA Toolkit |
| `--toolkitpath` | 安装到用户目录      |
| 不安装 driver   | 避免需要 root       |

------



## 3）配置环境变量（`.bashrc.d` 模块化）

### 创建目录

```bash
mkdir -p ~/.bashrc.d
```



### 创建 CUDA 配置文件

```bash
nano ~/.bashrc.d/cuda.sh
```

写入：

```bash
# ===== 默认 CUDA =====
export CUDA_HOME=$HOME/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ===== CUDA 切换函数 =====
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



### 确保 `.bashrc` 会加载 `.bashrc.d`

在 `~/.bashrc` 中加入（若没有）：

```bash
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        [ -f "$rc" ] && . "$rc"
    done
fi
```



### 重新加载环境

```bash
source ~/.bashrc
```

------



## 4）验证是否成功

### 检查 nvcc

```bash
nvcc -V
```

如果能看到 CUDA 版本信息，说明安装成功。



### 检查 GPU 是否可用

```bash
nvidia-smi
```

-   能运行：说明服务器已经安装驱动，可以使用 GPU
-   不能运行：在没有 sudo 的情况下无法安装驱动，CUDA 只能编译，不能使用 GPU

这是一个非常关键、经常被忽略的点。

------









# 在无 root 权限的 HPC 集群中本地安装 CMake 并永久生效

## 下载官方预编译安装脚本

这里下载的是 **官方提供的二进制安装包**，
 无需源码编译，因此：

-   不需要 gcc / make
-   安装速度快
-   兼容大多数 Linux 环境

```shell
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-x86_64.sh
```



## 在用户目录安装 CMake

关键参数说明：

-   `--skip-license`：跳过交互式许可确认
-   `--prefix=$HOME/.local`：
     将 CMake 安装到用户级目录： `.local` 是 Linux 约定的**用户级软件安装位置**。

```
bash cmake-3.29.6-linux-x86_64.sh --skip-license --prefix=$HOME/.local
```



## 将本地 CMake 加入 PATH

```
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cmake --version
```











## .bash_profile

登录时自动执行 `.bashrc`

```
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
```







# 逐段解释你当前 `.bashrc`

## 1）系统初始化

读取系统级 bash 配置（module、颜色、补全等）

```bash
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi
```

------



## 2） 用户 PATH 初始化

把用户程序目录加入 PATH，且避免重复添加。

```bash
# 如果当前 PATH 里还没有包含：
#   $HOME/.local/bin 和 $HOME/bin 这两个用户程序目录
# 就执行下面的添加操作（防止重复加入 PATH）
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    # 把用户自己的可执行文件目录放到 PATH 最前面，
    # 这样通过 pip --user / 本地安装的程序可以优先生效
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi

# 将修改后的 PATH 导出为环境变量，
# 让子进程（Python、bash、程序等）也能继承这个 PATH
export PATH
```

------





## 3）`.bashrc.d` 机制

允许把配置拆成多个小文件。`.bashrc` 是必须的主配置文件， `.bashrc.d/` 是为了让配置更清晰而设计的“可选模块目录”。

通常在你开始：

-   管理多个 CUDA
-   多个 Conda 环境
-   多项目科研
-   自定义很多 alias



```bash
# ===== 用户自定义别名和函数的模块加载机制 =====

# 如果用户家目录下存在 ~/.bashrc.d 这个目录
if [ -d ~/.bashrc.d ]; then

    # 遍历该目录中的每一个文件
    for rc in ~/.bashrc.d/*; do

        # 如果当前遍历到的是“普通文件”（而不是目录或其他类型）
        if [ -f "$rc" ]; then

            # 在当前 shell 中执行这个文件的内容
            # 等价于：source "$rc"
            # 这样文件里的 alias / function / export 会直接生效
            . "$rc"
        fi
    done
fi
```

------





## 4） PATH配置

### CUDA 12.6 环境

```bash
# ===== cuda-12.6 =====
export CUDA_HOME=$HOME/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

作用：

>   指定默认 CUDA = **12.6**

------



###  libtorch 头文件和库路径（按需）

```bash
# ===== Libtorch 是 PyTorch 官方提供的 C++ API + 运行时库 =====
export CPATH=$HOME/libtorch/include:$HOME/libtorch/include/torch/csrc/api/include:$CPATH
export LIBRARY_PATH=$HOME/libtorch/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/libtorch/lib:$LD_LIBRARY_PATH
```

作用：

>   支持 **C++ 调用 PyTorch**。

### 是否需要？

| 用途                   | 是否保留 |
| ---------------------- | -------- |
| 写 C++/CUDA + libtorch | ✔ 必须   |
| 只用 Python PyTorch    | ❌ 可以删 |

------



### cuDNN 路径

只在：C++ / 自编 CUDA / TensorRT 才需要。

```
# ===== cuDNN =====
export CPATH=$HOME/cudnn/include:$CPATH
export LIBRARY_PATH=$HOME/cudnn/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/cudnn/lib:$LD_LIBRARY_PATH
```

------



### CUTLASS

```bash
# ===== CUTLASS =====
export CUTLASS=$HOME/cutlass
```

---





### 自定义命令

```bash
# ===== PATH 决定：终端去哪里找可执行文件 ===== 
export PATH=$HOME/.local/bin:$PATH
export PATH="$HOME/bin:$PATH"
```

------



