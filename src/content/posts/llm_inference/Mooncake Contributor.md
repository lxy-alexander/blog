---
title: "Mooncake Contributor"
published: 2026-08-13
description: "Mooncake Contributor"
image: ""
tags: ["llm_inference","Mooncake Contributor"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-13T21:40:15.071.303282038Z"
---

## 编译流程

````python
下面是按你这台机器整理的完整流程（无 sudo、RHEL 9、conda 工具链 + uv Python + `~/cuda-12.9`），以及一路踩过的坑。

---

# Mooncake 本机源码编译流程

## 分层

```
系统            不动
conda           mooncake-toolchain：cmake、gcc 13、ninja、C++ 库
uv .venv        Python 3.12 和 Python 包
~/cuda-12.9     CUDA（不是 /usr/local/cuda）
```

激活顺序固定：

```bash
conda activate mooncake-toolchain
cd /data/home/xli49/lxy/Mooncake
source .venv/bin/activate
```

`.venv` 在 PATH 前面 → `python` 走 uv；`cmake`/`gcc` 走 conda。不要在 `.venv` 激活后再切别的 conda 环境。永远用 `-n mooncake-toolchain`，不要在 `(base)` 里直接 `conda install`。

---

## 1. 建 conda 环境（GCC 必须 13，不要 16）

```bash
conda deactivate

conda create -n mooncake-toolchain -c conda-forge -y \
  cmake ninja pkg-config \
  gcc_linux-64=13 gxx_linux-64=13
```

C++ 依赖（对应 `dependencies.sh` 里的 devel 包）：

```bash
conda install -n mooncake-toolchain -c conda-forge -y \
  rdma-core \
  glog gflags jsoncpp \
  libunwind libnuma \
  libboost libboost-headers libboost-devel \
  protobuf yaml-cpp \
  hiredis liburing jemalloc \
  msgpack-c msgpack-cxx xxhash \
  patchelf gtest
```

说明：

- `libboost` 只有运行库，头文件要 `libboost-headers`
- `msgpack-c` 是 C 接口；Mooncake 要 `msgpack.hpp`，必须 `msgpack-cxx`
- `protobuf`/`hiredis` 可能顺带装上 conda 的 Python，**不要用那个 Python 装包**

工具链环境里没有名叫 `gcc` 的文件，只有 `x86_64-conda-linux-gnu-gcc`。已在该 env 的 `bin/` 做了 `gcc`/`g++` 链接；bashrc 里也去了写死的 base `LD_LIBRARY_PATH`。新终端里：

```bash
conda activate mooncake-toolchain
which gcc
readlink -f "$(which gcc)"
# 应是 .../envs/mooncake-toolchain/bin/x86_64-conda-linux-gnu-gcc
# 版本 13.x，不能是 16.x
```

---

## 2. uv 虚拟环境

```bash
conda activate mooncake-toolchain
cd /data/home/xli49/lxy/Mooncake
uv venv --python 3.12
source .venv/bin/activate

uv pip install pip build setuptools wheel numpy aiohttp requests msgpack
python -m pip --version    # 必须有输出；.venv 默认没有 pip
```

根目录 `requirements.txt` 只是 lint，不是运行时依赖。根目录没有 `pyproject.toml`，不要 `uv sync`，不要 `uv pip install -e .`。

---

## 3. CUDA（每次新开终端都要）

没有 `/usr/local/cuda`，nvlink 脚本却写死了这个路径：

```bash
export CUDA_HOME=/data/home/xli49/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

`CUDA_HOME` 有了还不够，必须设 `CPLUS_INCLUDE_PATH` / `LIBRARY_PATH`。

---

## 4. yalantinglibs

没有 sudo，装到 conda 前缀，不要 `/usr/local`：

```bash
cd /data/home/xli49/lxy/Mooncake
git submodule update --init --recursive

cmake -S extern/yalantinglibs -B extern/yalantinglibs/build \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build extern/yalantinglibs/build -j"$(nproc)"
cmake --install extern/yalantinglibs/build
```

---

## 5. 编 Mooncake

换过 gcc、或 CMake 曾经指向过 `base`，必须删 `build/`：

```bash
cd /data/home/xli49/lxy/Mooncake
rm -rf build

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DUSE_CUDA=ON

cmake --build build -j"$(nproc)"
```

日志里应看到：

- 编译器：`.../envs/mooncake-toolchain/bin/x86_64-conda-linux-gnu-c++`
- 头文件：`-isystem .../envs/mooncake-toolchain/include`

若还是 `/data/home/xli49/miniconda3/bin/...`（没有 `envs/mooncake-toolchain`），说明在用 base：`conda deactivate` 两次再重新激活。

---

## 6. 装进 Python（不要跑完整 build_wheel.sh）

`build_wheel.sh` 最后会跑 **auditwheel**（给 PyPI 打 manylinux 包）。本机 conda GCC 13 过不了，开发也不需要它。

```bash
cp build/mooncake-integration/engine.cpython-*.so mooncake-wheel/mooncake/engine.so
cp build/mooncake-integration/store.cpython-*.so mooncake-wheel/mooncake/store.so
cp build/mooncake-common/libasio.so mooncake-wheel/mooncake/libasio.so
cp build/mooncake-store/src/mooncake_master mooncake-wheel/mooncake/
cp build/mooncake-store/src/mooncake_client mooncake-wheel/mooncake/

uv pip install -e mooncake-wheel --no-build-isolation --force-reinstall
python -c "from mooncake.store import MooncakeDistributedStore; print('ok')"
```

---

## 7. 第一次运行（TCP put/get）

终端 A：

```bash
./build/mooncake-store/src/mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port=8080
```

终端 B（同样激活环境）：

```bash
python3 << 'EOF'
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup(
    local_hostname="localhost",
    metadata_server="P2PHANDSHAKE",
    global_segment_size=512 * 1024 * 1024,
    local_buffer_size=128 * 1024 * 1024,
    protocol="tcp",
    rdma_devices="",
    master_server_addr="127.0.0.1:50051",
)
print("put:", store.put("hello_key", b"Hello, Mooncake Store!"))
print(store.get("hello_key").decode())
store.close()
EOF
```

期望：`put: 0` 和 `Hello, Mooncake Store!`。

只测 C++ 是否编过：

```bash
./build/mooncake-transfer-engine/tests/common_test
```

---

## 8. 以后改代码怎么编

```bash
git pull
git submodule update --init --recursive
cmake --build build -j"$(nproc)"
# 若改了 pybind / master，再拷 .so 并 uv pip install -e ...
```

CMake 选项变了、换了 gcc 大版本、yalantinglibs 更新了：先重装 ylt（第 4 步），再 `rm -rf build` 走第 5 步。

---

## 踩坑记录

| 现象 | 原因 | 处理 |
|---|---|---|
| `conda install` 报 `.../miniconda3/envs` 不是环境 | 没带 `-n`，前缀指错 | `conda install -n mooncake-toolchain ...` |
| `./dependencies.sh` 要 root | 脚本强制 root + yum | 不要跑；用 conda 装库 |
| `infiniband/verbs.h` 找不到 | 系统没有 devel；或 CMake 在用 **base** | 装 `rdma-core`；确认 `$CONDA_PREFIX` 是 toolchain；删 `build/` 重配 |
| `which gcc` 是 `~/miniconda3/bin/gcc` | toolchain 里原来没有名为 `gcc` 的文件 | env `bin/` 已加链接；新开终端再 activate |
| `cuda.h` 找不到（nvlink allocator） | 脚本写死 `/usr/local/cuda` | 设 `CPLUS_INCLUDE_PATH`/`LIBRARY_PATH` 指向 `~/cuda-12.9`；只设 `CUDA_HOME` 不够 |
| `boost/functional/hash.hpp` 找不到 | 只装了 `libboost` | 再装 `libboost-headers` |
| `msgpack.hpp` 找不到 | 只装了 `msgpack-c` | 再装 `msgpack-cxx` |
| `LocalFileSnapshotObjectStore` 变成抽象类 | **GCC 16** 和 `tl::expected` 不兼容 | `gcc_linux-64=13`，删 `build/` 重配 |
| `python3.12: No module named pip` | uv venv 默认没有 pip；`build_wheel.sh` 误用 `python -m pip` | `uv pip install pip build setuptools wheel` |
| `auditwheel cannot repair ... x86_64_v2 / too-recent symbols` | 脚本按 **PyPI manylinux** 修包；本机 conda GCC 13 过不了 | 开发跳过；不要为此换成 GCC 11。拷 `.so` + `uv pip install -e` |
| 换 GCC 11 能否过 auditwheel | 不能。conda-forge 库本身常是 v2 | 真要过只能 Ubuntu 22.04 Docker / 官方 CI |

官方文档：gcc **9.4+**；CI 主力 Ubuntu 22.04 **GCC 11**，也测 24.04 **GCC 13**。本机用 **13** 即可，不要 16。

---

## 每次开终端的最小集合

```bash
conda activate mooncake-toolchain
cd /data/home/xli49/lxy/Mooncake
source .venv/bin/activate

export CUDA_HOME=/data/home/xli49/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```
````





## 执行程序

两个终端，都先进入环境。

```bash
conda activate mooncake-toolchain
cd /data/home/xli49/lxy/Mooncake
source .venv/bin/activate
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
```

**终端 A — 启动 master：**

```bash
./build/mooncake-store/src/mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port=8080
```

不要关这个窗口。

**终端 B — 第一个程序：**

```bash
python3 << 'EOF'
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup(
    local_hostname="localhost",
    metadata_server="P2PHANDSHAKE",
    global_segment_size=512 * 1024 * 1024,
    local_buffer_size=128 * 1024 * 1024,
    protocol="tcp",
    rdma_devices="",
    master_server_addr="127.0.0.1:50051",
)

print("put:", store.put("hello_key", b"Hello, Mooncake Store!"))
print(store.get("hello_key").decode())
store.close()
EOF
```

成功应打印：

```
put: 0
Hello, Mooncake Store!
```

若 `from mooncake.store` 失败，先装一次：

```bash
uv pip install -e mooncake-wheel --no-build-isolation
```





## auditwheel

`auditwheel` 是 Linux 上给 **Python wheel** 做检查和“修理”的工具，目的是让这个包能在别人的机器上用 `pip install` 装上。

Python 的 C/C++ 扩展 `.so` 会链到系统库（`libstdc++`、`libglog` 等）。你在 A 机器编出来的 wheel，拿到 B 机器可能缺库或 glibc 太旧。`auditwheel repair` 会：

1. 看 wheel 里的 ELF 依赖了哪些 `.so`
2. 把能打包的第三方库塞进 wheel
3. 打上 **manylinux** 标签（例如 `manylinux_2_34_x86_64`），表示“在 glibc ≥ 2.34 的普通 x86_64 Linux 上应该能跑”

所以它是 **发到 PyPI、给陌生人装** 用的，不是本机编译必须的一步。

你这边失败是因为：conda GCC 13 编出来的程序用了 `x86_64_v2` 指令，以及比 manylinux 基线更新的符号。`auditwheel` 认为这包不够“通用”，拒绝修理。官方 Mooncake 是在 Ubuntu 22.04 + GCC 11 的 CI 里打这个包的。

本机已经能 `import mooncake` 的话，不需要它。



## PyPI

**PyPI**（Python Package Index）是 Python 的官方包仓库，相当于 `pip` / `uv` 的默认下载源。

你在网上装包时：

```bash
pip install mooncake-transfer-engine
```

就是从 **pypi.org** 把别人已经打好的 wheel 拉下来。那些包要过 `auditwheel`、打上 manylinux 标签，才能在各种 Linux 上比较稳地安装。

你现在是 **自己从源码编、装进本机 `.venv`**，不经过 PyPI。只有把包发布给别人 `pip install` 时，才需要 PyPI 和 auditwheel。
