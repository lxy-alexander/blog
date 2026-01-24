---
title: vLLM Arch
published: 2026-01-23
description: "vLLM Arch"
image: ""
tags: ["Blogging","vLLM", "vLLM Arch"]
category: Guides
draft: false
---

# vLLM Directory

## 根目录文件

```
vllm/
├── CMakeLists.txt          # CMake 构建配置
├── setup.py                # Python 包安装脚本
├── README.md               # 项目说明文档
├── LICENSE                 # 许可证
├── MANIFEST.in             # 打包清单
├── CODE_OF_CONDUCT.md      # 行为准则
├── CONTRIBUTING.md         # 贡献指南
├── SECURITY.md             # 安全政策
├── RELEASE.md              # 发布说明
├── DCO                     # 开发者证书
├── use_existing_torch.py   # PyTorch 使用工具
└── codecov.yml             # 代码覆盖率配置
```

## 主要目录结构

### 1. **vllm/** - 核心源代码目录

```c++
vllm/ 
├── assets/                 # 多模态资源处理（音频、图像、视频） 
├── attention/              # 注意力机制实现
│   ├── backends/           # 不同后端的注意力实现
│   ├── layers/             # 注意力层
│   ├── ops/                # 注意力操作
│   └── utils/              # 工具函数
├── benchmarks/             # 性能基准测试
├── compilation/            # 编译相关（CUDA graphs, fusion等）
├── config/                 # 配置管理
├── device_allocator/       # 设备内存分配器
├── distributed/            # 分布式相关
│   ├── device_communicators/  # 设备通信
│   ├── ec_transfer/        # EC 传输
│   ├── eplb/               # EP 负载均衡
│   └── kv_transfer/        # KV 缓存传输
├── engine/                 # LLM 引擎核心
│   ├── llm_engine.py       # 主引擎
│   ├── async_llm_engine.py # 异步引擎
│   ├── arg_utils.py        # 参数工具
│   └── protocol.py         # 协议定义
├── entrypoints/            # 入口点
│   ├── anthropic/          # Anthropic API
│   ├── cli/                # 命令行接口
│   ├── openai/             # OpenAI 兼容 API
│   ├── pooling/            # 池化操作
│   ├── sagemaker/          # AWS SageMaker
│   └── serve/              # 服务相关
├── inputs/                 # 输入处理
├── logging_utils/          # 日志工具
├── lora/                   # LoRA 适配器支持
│   ├── layers/             # LoRA 层
│   ├── ops/                # LoRA 操作（Triton, Torch, IPEX, XLA）
│   └── punica_wrapper/     # Punica 包装器
├── model_executor/         # 模型执行器
│   ├── layers/             # 模型层（FLA, MoE, Mamba等）
│   ├── model_loader/       # 模型加载器
│   ├── models/             # 支持的模型实现
│   └── warmup/             # 预热相关
├── multimodal/             # 多模态支持
├── platforms/              # 平台支持（CUDA, ROCM, CPU, TPU, XPU）
├── plugins/                # 插件系统
│   ├── io_processors/      # IO 处理器
│   └── lora_resolvers/     # LoRA 解析器
├── profiler/               # 性能分析工具
├── ray/                    # Ray 集成
├── reasoning/              # 推理相关
├── tokenizers/             # 分词器
├── tool_parsers/           # 工具解析器
├── transformers_utils/      # Transformers 工具
│   ├── chat_templates/     # 聊天模板
│   ├── configs/            # 配置解析
│   └── processors/         # 处理器
├── triton_utils/           # Triton 工具
├── utils/                  # 通用工具
├── usage/                  # 使用示例
└── v1/                     # v1 API（新版本）
    ├── attention/          # v1 注意力
    ├── core/               # v1 核心
    ├── engine/             # v1 引擎
    ├── executor/           # v1 执行器
    ├── kv_offload/         # KV 卸载
    ├── metrics/            # 指标
    ├── sample/             # 采样
    ├── spec_decode/        # 推测解码
    ├── structured_output/  # 结构化输出
    └── worker/             # 工作器
```



### 2. **csrc/** - C++/CUDA 源代码

```
csrc/
├── attention/              # 注意力 CUDA 内核
│   ├── mla/                # Multi-head Latent Attention
│   └── [各种注意力实现]
├── core/                   # 核心功能
├── cpu/                    # CPU 实现
│   ├── micro_gemm/         # 微矩阵乘法
│   └── sgl-kernels/        # SGL 内核
├── cutlass_extensions/     # CUTLASS 扩展
│   └── epilogue/           # 尾声操作
├── mamba/                  # Mamba SSM 实现
├── moe/                    # Mixture of Experts
│   ├── marlin_moe_wna16/   # Marlin MoE
│   └── permute_unpermute_kernels/  # 置换内核
├── quantization/           # 量化实现
│   ├── awq/                # AWQ 量化
│   ├── gptq/               # GPTQ 量化
│   ├── marlin/             # Marlin 量化
│   ├── w8a8/               # W8A8 量化
│   └── [其他量化方法]
├── quickreduce/            # 快速归约
├── rocm/                   # ROCm 支持
└── sparse/                 # 稀疏矩阵
```

### 3. **tests/** - 测试目录

```
tests/
├── basic_correctness/      # 基本正确性测试
├── benchmarks/             # 基准测试
├── compile/                # 编译测试
├── config/                 # 配置测试
├── distributed/            # 分布式测试
├── engine/                 # 引擎测试
├── entrypoints/            # 入口点测试
│   ├── llm/                # LLM 测试
│   ├── openai/             # OpenAI API 测试
│   ├── pooling/            # 池化测试
│   └── sagemaker/          # SageMaker 测试
├── evals/                  # 评估测试
├── kernels/                # 内核测试
│   ├── attention/          # 注意力内核
│   ├── moe/                # MoE 内核
│   └── quantization/       # 量化内核
├── lora/                   # LoRA 测试
├── models/                 # 模型测试
│   ├── language/           # 语言模型
│   ├── multimodal/         # 多模态模型
│   └── quantization/       # 量化模型
├── quantization/           # 量化测试
├── reasoning/              # 推理测试
├── tool_use/               # 工具使用测试
├── transformers_utils/     # Transformers 工具测试
├── utils_/                 # 工具测试
└── v1/                     # v1 API 测试
```

### 4. **examples/** - 示例代码

```
examples/
├── offline_inference/      # 离线推理示例
│   ├── basic/              # 基础示例
│   ├── logits_processor/   # Logits 处理器
│   └── [其他示例]
├── online_serving/         # 在线服务示例
├── pooling/                # 池化示例
└── others/                 # 其他示例
```

### 5. **docs/** - 文档目录

```
docs/
├── api/                    # API 文档
├── assets/                 # 文档资源（图片等）
├── benchmarking/           # 基准测试文档
├── cli/                    # CLI 文档
├── configuration/          # 配置文档
├── contributing/           # 贡献文档
├── deployment/             # 部署文档
├── design/                 # 设计文档
├── features/               # 功能文档
├── getting_started/        # 入门指南
├── models/                 # 模型文档
├── serving/                # 服务文档
└── usage/                  # 使用文档
```

### 6. **tools/** - 工具脚本

```
tools/
├── ep_kernels/             # EP 内核工具
├── pre_commit/             # 预提交钩子
├── profiler/               # 性能分析工具
└── vllm-tpu/               # TPU 相关工具
```

### 7. **benchmarks/** - 基准测试

```
benchmarks/
└── [各种基准测试脚本]
```

### 8. **docker/** - Docker 配置

```
docker/
├── Dockerfile              # 主 Dockerfile
├── Dockerfile.cpu          # CPU 版本
├── Dockerfile.rocm         # ROCm 版本
├── Dockerfile.tpu          # TPU 版本
└── [其他平台 Dockerfile]
```

### 9. **requirements/** - 依赖管理

```
requirements/
├── build.txt               # 构建依赖
├── common.txt              # 通用依赖
├── cuda.txt                # CUDA 依赖
├── rocm.txt                # ROCm 依赖
├── cpu.txt                 # CPU 依赖
├── dev.txt                 # 开发依赖
├── docs.txt                # 文档依赖
└── [其他依赖文件]
```

### 10. **其他重要目录**

```
.buildkite/                 # Buildkite CI/CD 配置
.github/                    # GitHub 配置（workflows, issue templates）
cmake/                      # CMake 模块
```



# Core Modules

## 引擎系统

- **vllm/engine/**: 主要的 LLM 引擎实现
- **vllm/v1/engine/**: v1 版本的引擎（新架构）

## 模型执行

- **vllm/model_executor/**: 模型加载和执行
- **vllm/model_executor/models/**: 各种模型架构的实现

## 注意力机制

- **vllm/attention/**: Python 层面的注意力实现
- **csrc/attention/**: CUDA 内核级别的注意力实现

## 分布式支持

- **vllm/distributed/**: 分布式训练和推理支持

## 量化支持

- **vllm/model_executor/layers/quantization/**: Python 量化层
- **csrc/quantization/**: CUDA 量化内核

## LoRA 支持

- **vllm/lora/**: LoRA 适配器的完整实现

## 多模态支持

- **vllm/multimodal/**: 多模态模型支持
- **vllm/assets/**: 多模态资源处理

## 平台支持

- **CUDA**: NVIDIA GPU
- **ROCM**: AMD GPU
- **CPU**: CPU 推理
- **TPU**: Google TPU
- **XPU**: Intel GPU



