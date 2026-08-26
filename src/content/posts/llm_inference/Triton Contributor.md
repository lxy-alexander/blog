---
title: "Triton Contributor"
published: 2026-08-26
description: "Triton Contributor"
image: ""
tags: ["llm_inference","Triton Contributor"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-26T04:57:31.023.758501596Z"
---

和当前仓库要求的对照：

| 你装的                                | 仓库要求                                                 | 结论                                         |
| :------------------------------------ | :------------------------------------------------------- | :------------------------------------------- |
| `python=3.12`                         | `>=3.10,<3.15`，wheel 也覆盖 3.10–3.14                   | 符合，而且官方 abi3 wheel 就是在 3.12 上编的 |
| `cmake`                               | `CMake >= 3.20`，`python/requirements.txt` 还写了 `<4.0` | 版本范围要卡住，别装到 4.x                   |
| `ninja`                               | `setup.py` 强制要有 ninja                                | 符合                                         |
| `gcc_linux-64=12` / `gxx_linux-64=12` | C++17；官方 manylinux wheel 用的是 GCC 12.2.1            | 符合，但不是必须                             |

需要改的一点：给 cmake 加上限。conda-forge 现在很容易给到 CMake 4，而 Triton 明确不要 4.x

给 cmake 加上限。conda-forge 现在很容易给到 CMake 4，而 Triton 明确不要 4.x：

```
conda create -n triton-toolchain -y \
    python=3.12 \
    "cmake>=3.20,<4" ninja \
    gcc_linux-64=12 gxx_linux-64=12 \
    -c conda-forge
```

这只是编译器环境。激活之后还要按仓库步骤装 Python 依赖并编译：

```
conda activate triton-toolchain
pip install -r python/requirements.txt
pip install -e .
```



下面是修好后的版本：去掉了无效的 `CMAKE_*` 环境变量和 deactivate 里改 `PATH` 的逻辑，激活钩子改成最后执行，避免被 conda 自带的编译器脚本盖掉。

```bash
CONDA_ENV_NAME=triton-toolchain bash <<'SETUP'
set -euo pipefail

CONDA_BASE=$(conda info --base)
CONDA_ENV_PREFIX="$CONDA_BASE/envs/$CONDA_ENV_NAME"

if [ ! -d "$CONDA_ENV_PREFIX" ]; then
    echo "Conda environment not found: $CONDA_ENV_NAME" >&2
    echo "Expected path: $CONDA_ENV_PREFIX" >&2
    exit 1
fi

require_tool() {
    tool_path="$CONDA_ENV_PREFIX/bin/$1"
    if [ ! -x "$tool_path" ]; then
        echo "Missing toolchain binary: $tool_path" >&2
        exit 1
    fi
}

for compiler in \
    x86_64-conda-linux-gnu-gcc \
    x86_64-conda-linux-gnu-g++ \
    x86_64-conda-linux-gnu-cc \
    x86_64-conda-linux-gnu-c++
do
    require_tool "$compiler"
done

for tool in \
    x86_64-conda-linux-gnu-cpp \
    x86_64-conda-linux-gnu-ld \
    x86_64-conda-linux-gnu-ar \
    x86_64-conda-linux-gnu-as \
    x86_64-conda-linux-gnu-nm \
    x86_64-conda-linux-gnu-ranlib \
    x86_64-conda-linux-gnu-strip \
    x86_64-conda-linux-gnu-objcopy \
    x86_64-conda-linux-gnu-objdump \
    x86_64-conda-linux-gnu-readelf
do
    require_tool "$tool"
done

ln -sfn x86_64-conda-linux-gnu-gcc "$CONDA_ENV_PREFIX/bin/gcc"
ln -sfn x86_64-conda-linux-gnu-g++ "$CONDA_ENV_PREFIX/bin/g++"
ln -sfn x86_64-conda-linux-gnu-cc  "$CONDA_ENV_PREFIX/bin/cc"
ln -sfn x86_64-conda-linux-gnu-c++ "$CONDA_ENV_PREFIX/bin/c++"

mkdir -p "$CONDA_ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_ENV_PREFIX/etc/conda/activate.d/zz-triton-toolchain.sh" <<'ACTIVATE'
#!/usr/bin/env sh

_triton_backup_var() {
    var_name="$1"
    backup_var_name="TRITON_BACKUP_${var_name}"
    eval "is_set=\${$var_name+x}"
    if [ -n "${is_set:-}" ]; then
        eval "export $backup_var_name=\"\${$var_name}\""
    else
        eval "unset $backup_var_name"
    fi
}

for var_name in \
    CC \
    CXX \
    CPP \
    LD \
    AR \
    AS \
    NM \
    RANLIB \
    STRIP \
    OBJCOPY \
    OBJDUMP \
    READELF \
    CUDAHOSTCXX \
    CUDA_HOST_COMPILER
do
    _triton_backup_var "$var_name"
done
unset _triton_backup_var var_name backup_var_name is_set

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CPP="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cpp"
export LD="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ld"
export AR="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ar"
export AS="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-as"
export NM="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-nm"
export RANLIB="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-ranlib"
export STRIP="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-strip"
export OBJCOPY="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-objcopy"
export OBJDUMP="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-objdump"
export READELF="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-readelf"

# nvcc uses CUDAHOSTCXX; some CMake CUDA paths also look at CUDA_HOST_COMPILER.
export CUDAHOSTCXX="$CXX"
export CUDA_HOST_COMPILER="$CXX"
ACTIVATE

cat > "$CONDA_ENV_PREFIX/etc/conda/deactivate.d/zz-triton-toolchain.sh" <<'DEACTIVATE'
#!/usr/bin/env sh

_triton_restore_var() {
    var_name="$1"
    backup_var_name="TRITON_BACKUP_${var_name}"
    eval "backup_is_set=\${$backup_var_name+x}"
    if [ -n "${backup_is_set:-}" ]; then
        eval "export $var_name=\"\${$backup_var_name}\""
    else
        eval "unset $var_name"
    fi
    eval "unset $backup_var_name"
}

for var_name in \
    CC \
    CXX \
    CPP \
    LD \
    AR \
    AS \
    NM \
    RANLIB \
    STRIP \
    OBJCOPY \
    OBJDUMP \
    READELF \
    CUDAHOSTCXX \
    CUDA_HOST_COMPILER
do
    _triton_restore_var "$var_name"
done
unset _triton_restore_var var_name backup_var_name backup_is_set
DEACTIVATE

echo "Configured conda toolchain for: $CONDA_ENV_NAME"
echo
echo "Verify with:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  hash -r"
echo "  which gcc"
echo "  which g++"
echo '  echo $CC'
echo '  echo $CXX'
SETUP
```

主要改动：

- 钩子改名为 `zz-triton-toolchain.sh`，激活时最后跑，避免被 conda-forge 自带的 gcc 脚本盖掉
- 删掉 `CMAKE_C_COMPILER` 这类环境变量，CMake 不认它们，只认 `CC` / `CXX`
- 删掉 deactivate 里改 `PATH` 的 `sed`，conda 自己会处理
- 导出前检查 `ld`/`ar` 等 binutils 是否存在
- 补上 `CUDA_HOST_COMPILER`，给 nvcc/CMake CUDA 用

跑完后重新激活再验证：

```bash
conda activate triton-toolchain
hash -r
which gcc
echo $CC
echo $CXX
gcc --version
```

`which gcc` 和 `$CC` 都应该落在 `.../envs/triton-toolchain/bin/` 下。



Triton 仓库没有官方 `uv` 文档，用法就是把 README 里的 `pip` 换成 `uv pip`。你现在已经有 `triton-toolchain`，**Python / GCC / CMake 继续用 conda，包用 uv 往这个环境里装。**

```bash
conda activate triton-toolchain
hash -r
cd /data/home/xli49/lxy/triton

# 用 conda 的 Python 3.12 建 venv，不要用系统 python
uv venv .venv --python "$CONDA_PREFIX/bin/python" --prompt triton

source .venv/bin/activate
hash -r

uv pip install -r python/requirements.txt
uv pip install -e . --no-build-isolation
# 让 uv 装进当前 conda 的 Python，而不是另建 venv
export UV_PYTHON="$CONDA_PREFIX/bin/python"

# 编译依赖
uv pip install -r python/requirements.txt

# 源码可编辑安装；必须关 build isolation，才会用上刚刚装的 cmake/ninja/nanobind
uv pip install -e . --no-build-isolation
```

如果 uv 提示 “not a virtual environment”，加上 `--system`：

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" --system -r python/requirements.txt
uv pip install --python "$CONDA_PREFIX/bin/python" --system -e . --no-build-isolation
```

装完后确认：

```bash
python -c "import triton; print(triton.__file__); print(triton.__version__)"
which gcc
echo "$CC"
```

`triton.__file__` 应在 `/data/home/xli49/lxy/triton/python/triton/` 下，`gcc` / `$CC` 应指向 conda 环境。

---

**不要用** `uv sync` / `uv add`。这个仓库的 `pyproject.toml` 只有 `build-system`，没有项目依赖表，`uv sync` 对不上官方安装路径。

第一次编译会下载 LLVM，时间会比较长。内存紧张时：

```bash
MAX_JOBS=8 uv pip install --python "$CONDA_PREFIX/bin/python" --system -e . --no-build-isolation
```
