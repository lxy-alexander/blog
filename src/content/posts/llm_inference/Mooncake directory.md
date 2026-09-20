---
title: "Mooncake directory"
published: 2026-09-19
description: "Mooncake directory"
image: ""
tags: ["llm_inference","Mooncake directory"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-19T16:51:20.504.204035062Z"
---

这个仓库不是一个单纯的 Python 项目，而是一个以 C/C++ 为核心、同时提供 Python/Go/Rust 接口和二进制发布包的 monorepo。理解它最简单的方法是分成四层：

```
应用与框架
SGLang / vLLM / RL / 用户 Python 程序
                  │
                  ▼
Python 接口层
python/ + mooncake-wheel/ + mooncake-integration/
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
Mooncake Store          EP / PG / Reshard
分布式 KV 缓存          弹性并行与数据重分片
        │
        ▼
Transfer Engine
RDMA/TCP/NVLink/EFA 等数据传输
        │
        ▼
mooncake-common
日志、RPC、配置、公共工具、元数据后端
```

## 一、最核心的目录

```
Mooncake/
├── mooncake-common/
├── mooncake-transfer-engine/
├── mooncake-store/
├── mooncake-integration/
├── python/
├── mooncake-wheel/
├── mooncake-ep/
├── mooncake-pg/
├── mooncake-reshard/
├── mooncake-conductor/
├── mooncake-p2p-store/
└── mooncake-rl/
```

### `mooncake-common/`：公共基础设施

这是底层公共库，被 Transfer Engine、Store 等模块复用。

```
mooncake-common/
├── include/       对外或跨模块使用的头文件
├── src/           C++ 实现
├── tests/         单元测试
├── etcd/          etcd 元数据服务适配，包含 Go 封装
└── k8s-lease/     Kubernetes Lease 高可用/选主适配
```

可以把它理解为 Mooncake 的“基础工具箱”，包含：

-   日志、错误处理
-   RPC/异步通信公共能力
-   配置及辅助函数
-   etcd、Kubernetes Lease 等元数据后端
-   `libasio.so` 等运行时公共库

顶层构建会最先引入它，见 [CMakeLists.txt (line 119)](/data/home/xli49/lxy/Mooncake/CMakeLists.txt:119)。

------

### `mooncake-transfer-engine/`：数据传输核心

这是 Mooncake 最底层、最核心的功能模块，简称 TE。

它解决的问题是：

>   将 CPU 内存、GPU 显存、NPU 内存、SSD 等位置的数据，通过 RDMA、TCP、EFA、NVLink 等协议高速搬到另一台机器或设备。

```
mooncake-transfer-engine/
├── include/                 C++ 公共接口
│   ├── transport/           各种传输协议接口
│   ├── shared_segment/      共享内存段抽象
│   └── gpu_vendor/          GPU 厂商相关抽象
├── src/
│   ├── transport/           RDMA、TCP、EFA、Ascend 等实现
│   ├── shared_segment/
│   └── common/
├── example/                 C++ 示例和 benchmark 程序
├── tests/                   单元及容错测试
├── benchmark/               性能测试
├── rust/                    Rust 接口
├── nvlink-allocator/        CUDA/NVLink 内存分配支持
├── ubshmem-allocator/       Ascend NPU 共享内存支持
└── tent/                    新的传输/监控相关子系统
```

典型调用逻辑是：

```
registerLocalMemory()
        │
        ▼
注册 CPU/GPU 内存区域
        │
        ▼
submitTransfer()
        │
        ▼
选择 RDMA / TCP / EFA / NVLink 等传输路径
```

Store、vLLM Connector、SGLang Connector 都依赖这个模块。

------

### `mooncake-store/`：分布式对象/KV Cache 存储

Store 构建在 Transfer Engine 之上。

Transfer Engine 只负责“搬数据”，Store 则负责：

-   一个对象叫什么
-   对象存在哪里
-   有几个副本
-   数据是否需要淘汰
-   放在 DRAM、SSD 还是远端节点
-   哪个节点负责管理
-   如何进行主从、高可用和故障恢复

```
mooncake-store/
├── include/                 C++ 头文件
├── src/                     Store 核心实现
│   ├── placement/           对象放置策略
│   ├── storage/             存储层
│   ├── segment/             内存段管理
│   ├── local_ssd/           本地 SSD 层
│   ├── spdk/                SPDK 支持
│   ├── ha/                  高可用
│   ├── kv_event/            KV 事件
│   ├── serialize/           序列化
│   └── cachelib_memory_allocator/
├── conf/                    配置样例
├── tests/                   单元、集成、E2E 测试
├── benchmarks/              Store 性能测试
├── tools/                   管理工具
├── go/                      Go SDK
└── rust/                    Rust SDK
```

Store 里有几个重要角色：

```
Client
  │ put/get/remove
  ▼
Master
  │ 查询元数据、决定对象放置位置
  ▼
存储节点内存/SSD
  │
  ▼
Transfer Engine 实际搬运数据
```

------

### `mooncake-integration/`：C++ 到 Python 的桥梁

这个目录容易和 `mooncake-wheel` 混淆。

它主要使用 pybind11，把 C++ 能力变成 Python 模块：

```
mooncake-integration/
├── transfer_engine/
│   ├── transfer_engine_py.cpp
│   └── shared_segment_py.cpp
├── store/
│   ├── store_py.cpp
│   ├── buffer_pool.cpp
│   ├── engram_store_py.cpp
│   └── async_store.py
├── allocator.py
├── allocator_ascend_npu.py
├── shared_segment.py
└── CMakeLists.txt
```

构建后主要产生：

```
engine.so    # Transfer Engine 的 Python 扩展
store.so     # Mooncake Store 的 Python 扩展
```

因此 Python 调用：

```
from mooncake import engine
from mooncake import store
```

底层实际上进入的是这里构建出的 C++ 扩展。

------

## 二、三个容易混淆的 Python 目录

仓库目前处于 Python 打包结构迁移期，所以同时存在：

```
python/
mooncake-wheel/
mooncake-integration/
```

它们的职责不一样。

### `python/`：新的 Python 源码和统一构建入口

```
python/
├── mooncake/
│   ├── __init__.py
│   ├── ep.py
│   ├── mooncake_ep_buffer.py
│   └── mooncake_elastic_buffer.py
├── tests/
│   ├── ep/
│   └── packaging/
└── CMakeLists.txt
```

仓库根目录的 [pyproject.toml (line 1)](/data/home/xli49/lxy/Mooncake/pyproject.toml:1) 使用 `scikit-build-core`：

```
build-backend = "scikit_build_core.build"
wheel.packages = ["python/mooncake"]
```

这是新的统一 Python 打包架构。它允许在一次构建中同时处理：

-   Python 源码
-   CMake 项目
-   pybind11 扩展
-   C/C++ 动态库
-   可执行程序

但目前并非所有 Python 文件都已经迁入 `python/mooncake`。

------

### `mooncake-wheel/`：旧 Python 源码 + 传统 Wheel 组装区

```
mooncake-wheel/
├── mooncake/                       Python 包内容
│   ├── cli.py
│   ├── cli_client.py
│   ├── mooncake_config.py
│   ├── mooncake_store_service.py
│   ├── structured_object_store.py
│   ├── http_metadata_server.py
│   ├── engine.so                   构建时复制进来
│   ├── store.so                    构建时复制进来
│   ├── mooncake_master             构建时复制进来
│   └── transfer_engine_bench       构建时复制进来
├── tests/                          Python API 和打包测试
├── pyproject.toml                  旧 setuptools 配置
├── setup.py
├── build/                          临时构建目录
└── dist/                           最终 .whl 输出
```

它有双重身份：

1.  尚未迁移的 Python 模块源码目录。
2.  传统发布脚本的 Wheel 组装区。

传统脚本 [build_wheel.sh (line 23)](/data/home/xli49/lxy/Mooncake/scripts/build_wheel.sh:23) 会把各处产物汇总到这里：

```
build/mooncake-integration/engine*.so
                    │
                    ▼
mooncake-wheel/mooncake/engine.so

build/mooncake-integration/store*.so
                    │
                    ▼
mooncake-wheel/mooncake/store.so

build/mooncake-store/src/mooncake_master
                    │
                    ▼
mooncake-wheel/mooncake/mooncake_master
```

然后生成：

```
mooncake-wheel/dist/mooncake_transfer_engine-*.whl
```

所以：

>   `mooncake-wheel` 不是 Mooncake 的一个运行时子系统，而是 Python 源码和二进制产物的发布/组装区域。

------

### 新旧打包路径为什么同时存在

当前有两条构建路径：

```
新路径：
根 pyproject.toml
    → scikit-build-core
    → CMake 安装各组件
    → wheel

传统发布路径：
CMake 构建到 build/
    → scripts/build_wheel.sh 复制产物
    → mooncake-wheel/
    → setuptools
    → dist/*.whl
```

新的构建路径也暂时会从旧目录读取尚未迁移的 `.py` 文件，见 [python/CMakeLists.txt (line 16)](/data/home/xli49/lxy/Mooncake/python/CMakeLists.txt:16)。

最终 Wheel 大致是这些目录的合成结果：

```
python/mooncake/
mooncake-wheel/mooncake/
mooncake-reshard/python/mooncake/reshard/
mooncake-integration 生成的 engine.so、store.so
Store/TE/EP/PG 生成的动态库和可执行文件
                         │
                         ▼
            一个可 pip install 的 mooncake 包
```

------

## 三、并行计算相关模块

### `mooncake-ep/`：Expert Parallel

EP 是 MoE 模型的专家并行通信实现。

```
mooncake-ep/
├── include/elastic/       弹性 EP 接口
├── src/                   C++/CUDA 等原生实现
└── benchmarks/            EP 性能测试
```

主要负责：

-   token dispatch：把 token 发送给负责相应 expert 的 GPU
-   combine：汇总 expert 输出
-   感知失效或不活跃的 rank
-   支持弹性专家并行

构建后会产生类似 `_ep*.so` 的 Python 扩展。

------

### `mooncake-pg/`：Process Group

这是面向 `torch.distributed` 的通信后端。

```
mooncake-pg/
├── include/
│   └── control_plane/
├── src/
│   └── control_plane/
├── torch/                 PyTorch ProcessGroup 扩展
├── tests/
└── benchmark/
```

它提供：

-   collective 通信
-   rank 状态检测
-   故障报告
-   rank 恢复
-   PyTorch ProcessGroup 接口

EP 更偏向 MoE 专家通信；PG 更偏向通用分布式进程组和容错。

------

### `mooncake-reshard/`：模型权重和 KV Cache 重分片

这是一个以 Python 为主的逻辑规划层。

```
mooncake-reshard/
├── python/mooncake/reshard/
├── tests/
├── benchmarks/
└── typecheck/
```

它负责描述：

-   TP/PP/EP/DP 拓扑
-   tensor 如何分片
-   从旧布局到新布局需要复制哪些区域
-   模型权重快照
-   KV Cache 布局变换
-   逻辑计划如何绑定到真实内存地址

需要注意：

>   Reshard 主要负责“算出怎么搬”，真正的数据传输仍由 Transfer Engine 执行。

详细设计写在 [mooncake-reshard/README.md (line 1)](/data/home/xli49/lxy/Mooncake/mooncake-reshard/README.md:1)。

------

## 四、其他功能模块

### `mooncake-conductor/`

Store 的控制面/索引相关组件，目前是可选构建模块。

```
mooncake-conductor/
├── include/conductor/
├── src/
│   ├── prefixindex/       前缀索引
│   ├── kvevent/           KV 事件处理
│   ├── zmq/               ZMQ 消息处理
│   └── common/
└── tests/
```

它更偏向：

-   接收 KV Cache 事件
-   建立 prefix 索引
-   帮助定位可复用的缓存对象

默认没有启用，需要 `WITH_CONDUCTOR=ON`。

### `mooncake-p2p-store/`

较早期、较轻量的 Go P2P Store 示例/实现：

```
mooncake-p2p-store/
└── src/
    ├── p2pstore/          Go 实现
    └── example/           使用样例
```

它和现在完整的 `mooncake-store` 不应混为一体。

### `mooncake-rl/`

面向强化学习场景的示例，目前内容较少，主要展示训练和推理之间的数据/权重传输。

------

## 五、工程辅助目录

| 目录                          | 作用                                          |
| ----------------------------- | --------------------------------------------- |
| `docs/`                       | Sphinx 文档源码、设计说明、部署指南和生成结果 |
| `benchmarks/`                 | 跨模块或完整应用场景的性能测试                |
| `scripts/`                    | 编译、CI、安装、测试、Ascend 支持及管理脚本   |
| `cmake/`                      | CMake 依赖查找和构建辅助逻辑                  |
| `extern/`                     | 第三方依赖，如 pybind11、yalantinglibs        |
| `docker/`                     | CUDA、ROCm、MUSA 等环境的 Dockerfile          |
| `monitoring/`                 | Prometheus 和 Grafana 配置                    |
| `FAST25-release/`             | FAST 2025 论文相关 trace 和发布材料           |
| `image/`                      | README 和文档使用的图片                       |
| `.github/`                    | GitHub Actions、PR 模板、CODEOWNERS           |
| `.devcontainer/`              | 开发容器配置                                  |
| `.claude/`、`.claude-plugin/` | AI 开发助手的工作流和技能配置                 |

------

## 六、哪些是源码，哪些是构建产物

当前工作区还有一些不属于核心源码的目录：

```
build/
build-repro-3517/
build-tent-bugs/
bin/
lib/
.venv/
.cache/
.ruff_cache/
master.log
mooncake-wheel/build/
mooncake-wheel/dist/
mooncake-wheel/*.egg-info/
```

通常含义如下：

-   `build/`：正常 CMake 编译输出
-   `build-repro-*`、`build-*-bugs`：为复现问题创建的额外构建目录
-   `bin/`、`lib/`：安装或构建生成的程序和库
-   `.venv/`：Python 虚拟环境
-   `.cache/`、`.ruff_cache/`：工具缓存
-   `master.log`：Store Master 运行日志
-   `mooncake-wheel/build/`：setuptools 中间产物
-   `mooncake-wheel/dist/`：最终 Wheel
-   `*.egg-info/`：Python 包元数据

修改代码时，一般不要直接改 `build/`、`dist/` 中的文件，因为重新构建后会被覆盖。

## 七、最实用的阅读路线

如果你的目标是理解整个项目，建议按这个顺序：

1.  [README.md (line 78)](/data/home/xli49/lxy/Mooncake/README.md:78)：先理解 TE、Store、EP/PG 的总体关系。
2.  [顶层 CMakeLists.txt (line 19)](/data/home/xli49/lxy/Mooncake/CMakeLists.txt:19)：了解哪些模块被组合构建。
3.  `mooncake-transfer-engine/include/`：了解数据传输 API。
4.  `mooncake-transfer-engine/src/`：了解具体 transport。
5.  `mooncake-store/include/` 和 `src/`：理解 Store 如何使用 TE。
6.  `mooncake-integration/`：理解 C++ API 如何导出给 Python。
7.  [根 pyproject.toml (line 89)](/data/home/xli49/lxy/Mooncake/pyproject.toml:89) 和 [python/CMakeLists.txt (line 1)](/data/home/xli49/lxy/Mooncake/python/CMakeLists.txt:1)：理解最终 Python 包如何形成。
8.  最后再看 `mooncake-wheel/`，把它当成“旧源码兼容区和传统发布组装区”。
