---
title: "LLM Theory"
published: 2026-07-17
description: "LLM Theory"
image: ""
tags: ["llm","LLM Theory"]
category: llm
draft: false
lang: ""
createdAt: "2026-07-17T17:26:27.966.077468413Z"
---

# 一、概率与信息论基础

1）概率分布、条件概率、联合概率、边缘概率
2）最大似然估计 MLE
3）贝叶斯公式
4）熵、交叉熵、条件熵
5）KL 散度、JS 散度
6）互信息
7）负对数似然 NLL
8）Perplexity
9）Softmax、Log-Softmax
10）温度参数与概率分布变化
11）Log-Sum-Exp 数值稳定技巧
12）采样、期望、方差

------

# 二、语言模型基础

1）什么是语言模型
2）自回归语言模型
3）自编码语言模型
4）因果语言模型 Causal LM
5）掩码语言模型 Masked LM
6）序列概率的链式分解
7）Next Token Prediction
8）Teacher Forcing
9）Exposure Bias
10）生成式模型与判别式模型的区别

自回归语言模型：

# [ P(x_1,\dots,x_n)

\prod_{t=1}^{n}P(x_t|x_{<t})
]

训练目标：

# [ \mathcal L

-\sum_{t=1}^{n}
\log P(x_t|x_{<t})
]

------

# 三、文本表示与 Tokenization

1）字符级、词级、子词级 Tokenization
2）BPE
3）Byte-level BPE
4）WordPiece
5）Unigram Language Model
6）SentencePiece
7）词表大小对模型的影响
8）特殊 Token
9）BOS、EOS、PAD、UNK
10）中文 Tokenization 特点
11）Byte fallback
12）不同 tokenizer 下 PPL 不可直接比较
13）词表扩展
14）Embedding 矩阵
15）输入 Embedding 与输出权重共享

常见面试问题：

1）为什么不用纯字符或纯词级 tokenizer
2）词表越大是否越好
3）中文一个汉字是否一定对应一个 token
4）Tokenizer 如何影响训练速度和上下文长度

------

# 四、Transformer 整体结构

1）Transformer 的整体架构
2）Encoder-only 模型
3）Decoder-only 模型
4）Encoder-Decoder 模型
5）Transformer Block 的组成
6）Multi-Head Attention
7）Feed Forward Network
8）Residual Connection
9）Layer Normalization
10）Dropout
11）Embedding 层
12）输出投影层
13）因果 Mask
14）Padding Mask

典型结构：

[
x
\rightarrow
\text{Attention}
\rightarrow
\text{Residual}
\rightarrow
\text{FFN}
\rightarrow
\text{Residual}
]

LLM 通常以 Decoder-only Transformer 为主。

------

# 五、Self-Attention

这是 LLM 理论最核心的部分。

1）Query、Key、Value 的含义
2）Scaled Dot-Product Attention
3）为什么要除以 (\sqrt{d_k})
4）Attention Score
5）Softmax 权重
6）Self-Attention
7）Cross-Attention
8）Multi-Head Attention
9）Head 的作用
10）因果 Attention Mask
11）Attention 的时间和空间复杂度
12）Attention 是否等价于相似度计算
13）Attention 与卷积、RNN 的区别

公式：

[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
]

# [ \operatorname{Attention}(Q,K,V)

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
]

复杂度通常为：

[
O(n^2d)
]

其中 (n) 是序列长度。

------

# 六、Multi-Head Attention

1）为什么使用多个 Attention Head
2）每个 Head 是否学习不同关系
3）Head 数量与 Head Dimension
4）多个 Head 如何拼接
5）Grouped Query Attention
6）Multi-Query Attention
7）Multi-Head Latent Attention
8）Attention Head 冗余
9）Head 剪枝

公式：

# [ \operatorname{head}_i

\operatorname{Attention}
(QW_i^Q,KW_i^K,VW_i^V)
]

# [ \operatorname{MHA}

\operatorname{Concat}
(\operatorname{head}_1,\dots,\operatorname{head}_h)W_O
]

重点区分：

1）MHA：每个 Query Head 有独立的 K、V Head
2）MQA：多个 Query Head 共享一组 K、V
3）GQA：若干 Query Head 共享一组 K、V

GQA 和 MQA 可以显著减少 KV Cache。

------

# 七、位置编码

1）为什么 Transformer 需要位置编码
2）绝对位置编码
3）正弦余弦位置编码
4）可学习位置 Embedding
5）相对位置编码
6）RoPE
7）ALiBi
8）旋转位置编码的原理
9）位置外推
10）长上下文扩展
11）RoPE Scaling
12）位置插值 Position Interpolation
13）NTK-aware Scaling

经典正弦位置编码：

# [ PE(pos,2i)

\sin
\left(
\frac{pos}{10000^{2i/d}}
\right)
]

# [ PE(pos,2i+1)

\cos
\left(
\frac{pos}{10000^{2i/d}}
\right)
]

RoPE 的核心思想是通过旋转 Query 和 Key，将相对位置信息引入内积。

------

# 八、Feed Forward Network

1）FFN 的作用
2）为什么 FFN 逐 Token 独立计算
3）升维再降维
4）ReLU
5）GELU
6）SiLU / Swish
7）GLU
8）SwiGLU
9）GeGLU
10）FFN 参数量为什么很大
11）FFN 与知识存储的关系

标准 FFN：

# [ \operatorname{FFN}(x)

W_2\sigma(W_1x+b_1)+b_2
]

SwiGLU：

# [ \operatorname{SwiGLU}(x)

(W_1x)\odot
\operatorname{SiLU}(W_2x)
]

然后再经过输出投影。

------

# 九、归一化与残差连接

1）LayerNorm
2）RMSNorm
3）BatchNorm 为什么不常用于 LLM
4）Pre-Norm
5）Post-Norm
6）Residual Connection
7）梯度消失与梯度爆炸
8）深层网络稳定训练
9）Norm 的位置对训练的影响

LayerNorm：

# [ \operatorname{LN}(x)

\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
]

RMSNorm 不减均值：

# [ \operatorname{RMSNorm}(x)

\gamma
\frac{x}
{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}
]

现代 LLM 中常见的是 Pre-Norm 和 RMSNorm。

------

# 十、模型参数量与计算量

1）参数量如何估算
2）Embedding 参数量
3）Attention 参数量
4）FFN 参数量
5）层数、隐藏维度、Head 数之间的关系
6）训练 FLOPs
7）推理 FLOPs
8）计算密集型与内存带宽密集型
9）模型宽度与深度
10）Scaling Law
11）Chinchilla Scaling Law
12）参数量、数据量、计算量如何平衡

一个 Transformer 层的大致参数量：

[
4d^2+2d d_{\text{ff}}
]

如果：

[
d_{\text{ff}}\approx4d
]

则一层参数量约为：

[
12d^2
]

这只是忽略 Bias 和 Norm 后的粗略估算。

------

# 十一、预训练

1）预训练目标
2）自监督学习
3）Next Token Prediction
4）数据清洗
5）数据去重
6）数据配比
7）数据质量
8）训练语料污染
9）Benchmark Contamination
10）课程学习 Curriculum Learning
11）数据重复对模型的影响
12）多语言训练
13）代码数据训练
14）上下文长度设置
15）训练稳定性
16）Checkpoint
17）断点恢复

------

# 十二、优化器与训练技巧

1）SGD
2）Momentum
3）Adam
4）AdamW
5）Weight Decay
6）Learning Rate
7）Warmup
8）Cosine Decay
9）Gradient Clipping
10）Gradient Accumulation
11）Mixed Precision
12）FP32、FP16、BF16
13）Loss Scaling
14）梯度检查点
15）参数初始化
16）Dropout
17）正则化
18）训练不稳定与 Loss Spike

Adam：

[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
]

# [ \theta_t

## \theta_{t-1}

\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
]

重点理解 Adam 和 AdamW 中 Weight Decay 的区别。

------

# 十三、分布式训练

1）数据并行 Data Parallel
2）模型并行 Model Parallel
3）张量并行 Tensor Parallel
4）流水线并行 Pipeline Parallel
5）序列并行 Sequence Parallel
6）上下文并行 Context Parallel
7）专家并行 Expert Parallel
8）ZeRO-1、ZeRO-2、ZeRO-3
9）FSDP
10）All-Reduce
11）All-Gather
12）Reduce-Scatter
13）通信开销
14）显存估算
15）参数、梯度、优化器状态的显存占用
16）3D Parallelism

混合精度下，训练显存不仅包含参数，还包括：

1）梯度
2）优化器状态
3）激活值
4）临时计算结果
5）通信缓冲区

------

# 十四、指令微调 SFT

1）预训练与微调的区别
2）Instruction Tuning
3）Supervised Fine-Tuning
4）Prompt 格式
5）Chat Template
6）System、User、Assistant Role
7）Loss Mask
8）是否只对 Assistant 部分计算 Loss
9）多轮对话训练
10）数据质量
11）能力遗忘
12）灾难性遗忘
13）Full Fine-Tuning
14）Parameter-Efficient Fine-Tuning

------

# 十五、参数高效微调

1）Adapter
2）Prompt Tuning
3）Prefix Tuning
4）P-Tuning
5）LoRA
6）QLoRA
7）低秩分解
8）LoRA Rank
9）LoRA Alpha
10）Target Modules
11）LoRA Merge
12）多个 Adapter 的组合
13）DoRA
14）IA3

LoRA：

[
W'=W+\Delta W
]

[
\Delta W=BA
]

其中：

[
A\in\mathbb R^{r\times d_{\text{in}}},
\quad
B\in\mathbb R^{d_{\text{out}}\times r}
]

并且：

[
r\ll \min(d_{\text{in}},d_{\text{out}})
]

------

# 十六、人类偏好对齐

1）Alignment
2）Reward Model
3）RLHF
4）PPO
5）DPO
6）IPO
7）KTO
8）ORPO
9）SimPO
10）GRPO
11）Preference Data
12）Chosen / Rejected Pair
13）KL Penalty
14）Reward Hacking
15）Over-optimization
16）长度偏差
17）偏好模型偏差

RLHF 基本流程：

1）预训练
2）SFT
3）训练 Reward Model
4）使用 PPO 优化策略模型

DPO 不需要显式训练 Reward Model，直接从偏好对中优化策略。

DPO 的核心形式与策略模型相对于参考模型的对数概率差有关。

------

# 十七、推理与解码

1）Greedy Search
2）Beam Search
3）Temperature Sampling
4）Top-k Sampling
5）Top-p Sampling
6）Typical Sampling
7）Min-p Sampling
8）Repetition Penalty
9）Frequency Penalty
10）Presence Penalty
11）Length Penalty
12）停止条件
13）EOS Token
14）随机种子
15）确定性生成
16）结构化输出
17）Constrained Decoding
18）Speculative Decoding

重点理解：

1）Greedy：每一步选概率最大 token
2）Beam Search：保留多个高概率序列
3）Temperature：控制分布尖锐程度
4）Top-k：只保留概率最高的 (k) 个 token
5）Top-p：保留累计概率达到 (p) 的最小集合

------

# 十八、KV Cache

1）为什么自回归生成需要 KV Cache
2）缓存哪些内容
3）为什么不缓存 Query
4）Prefill 阶段
5）Decode 阶段
6）KV Cache 显存占用
7）上下文长度对显存的影响
8）MHA、GQA、MQA 对 KV Cache 的影响
9）Paged Attention
10）Continuous Batching
11）Prefix Caching
12）KV Cache Quantization

KV Cache 大致占用与以下变量成正比：

[
\text{batch size}
\times
\text{sequence length}
\times
\text{layers}
\times
\text{KV heads}
\times
\text{head dimension}
]

------

# 十九、长上下文

1）长上下文的难点
2）二次 Attention 复杂度
3）位置外推
4）RoPE Scaling
5）稀疏 Attention
6）滑动窗口 Attention
7）局部 Attention
8）全局 Attention
9）Ring Attention
10）FlashAttention
11）Context Parallel
12）Lost in the Middle
13）上下文利用率
14）长文本训练
15）长文本评估
16）外部记忆

------

# 二十、FlashAttention 与高效 Attention

1）标准 Attention 的内存瓶颈
2）IO Complexity
3）Tiling
4）Online Softmax
5）FlashAttention
6）FlashAttention-2
7）FlashAttention-3
8）Memory Efficient Attention
9）稀疏 Attention
10）Linear Attention
11）Sliding Window Attention

FlashAttention 的核心不是改变 Attention 数学公式，而是改变计算顺序，减少显存读写。

------

# 二十一、MoE 混合专家模型

1）Mixture of Experts
2）Router
3）Top-k Routing
4）Expert FFN
5）Sparse Activation
6）共享专家
7）负载均衡
8）Auxiliary Loss
9）Router Z-Loss
10）Expert Collapse
11）专家并行
12）通信开销
13）总参数量与激活参数量
14）MoE 的优缺点

MoE 每个 Token 通常只激活少量专家，因此可以增加总参数量，而不同比例增加单 Token 计算量。

------

# 二十二、量化

1）什么是量化
2）PTQ
3）QAT
4）权重量化
5）激活量化
6）KV Cache 量化
7）对称量化
8）非对称量化
9）Per-Tensor
10）Per-Channel
11）Per-Group
12）INT8
13）INT4
14）FP8
15）NF4
16）GPTQ
17）AWQ
18）SmoothQuant
19）动态量化
20）静态量化
21）量化误差
22）Outlier Channel

基本形式：

[
q=\operatorname{round}\left(\frac{x}{s}\right)+z
]

反量化：

[
\hat x=s(q-z)
]

------

# 二十三、模型压缩

1）知识蒸馏
2）Teacher Model
3）Student Model
4）Soft Target
5）Logit Distillation
6）Feature Distillation
7）剪枝
8）结构化剪枝
9）非结构化剪枝
10）稀疏化
11）权重共享
12）低秩分解
13）模型合并
14）模型插值
15）Task Vector

------

# 二十四、RAG

1）为什么需要 RAG
2）Embedding Model
3）向量数据库
4）文档切分 Chunking
5）语义检索
6）稀疏检索 BM25
7）Dense Retrieval
8）Hybrid Search
9）Top-k Retrieval
10）Reranker
11）Query Rewrite
12）Multi-Query Retrieval
13）Context Compression
14）引用与证据
15）召回率与准确率
16）RAG 幻觉
17）检索污染
18）RAG 评估

RAG 基本流程：

[
\text{Query}
\rightarrow
\text{Retrieval}
\rightarrow
\text{Reranking}
\rightarrow
\text{Prompt}
\rightarrow
\text{Generation}
]

------

# 二十五、Embedding

1）词向量与句向量
2）语义空间
3）余弦相似度
4）点积相似度
5）欧氏距离
6）对比学习
7）正样本与负样本
8）InfoNCE Loss
9）Hard Negative
10）向量归一化
11）维度与存储成本
12）双塔模型
13）Cross-Encoder
14）Embedding 微调
15）Pooling 方法

余弦相似度：

# [ \cos(x,y)

\frac{x^\top y}
{|x||y|}
]

------

# 二十六、Prompt Engineering

1）Zero-shot
2）Few-shot
3）In-context Learning
4）Chain of Thought
5）Self-Consistency
6）ReAct
7）Role Prompt
8）System Prompt
9）Prompt Injection
10）上下文顺序
11）示例选择
12）结构化输出
13）Prompt 压缩
14）Prompt 缓存
15）Instruction Hierarchy

需要区分：

1）模型参数中的知识
2）上下文中临时提供的信息
3）外部工具返回的信息

------

# 二十七、工具调用与 Agent

1）Function Calling
2）Tool Use
3）Agent Loop
4）Planning
5）Memory
6）Observation
7）Action
8）ReAct
9）Multi-Agent
10）任务分解
11）工具选择
12）错误恢复
13）循环终止
14）权限控制
15）工具调用安全
16）Agent 评估

典型循环：

[
\text{Thought}
\rightarrow
\text{Action}
\rightarrow
\text{Observation}
\rightarrow
\text{Next Action}
]

------

# 二十八、幻觉与事实性

1）什么是幻觉
2）内在幻觉
3）外在幻觉
4）知识过时
5）错误归因
6）虚假引用
7）长上下文幻觉
8）解码策略对幻觉的影响
9）RAG 是否能完全消除幻觉
10）事实性评估
11）不确定性表达
12）校准 Calibration
13）拒答机制
14）自我验证
15）外部工具验证

------

# 二十九、模型评估

1）Intrinsic Evaluation
2）Extrinsic Evaluation
3）Perplexity
4）Accuracy
5）Exact Match
6）F1
7）BLEU
8）ROUGE
9）BERTScore
10）Pass@k
11）Human Evaluation
12）LLM-as-a-Judge
13）Pairwise Evaluation
14）Benchmark Contamination
15）数据泄漏
16）鲁棒性评估
17）安全性评估
18）事实性评估
19）长上下文评估
20）代码能力评估

代码生成中的 Pass@k：

# [ \operatorname{Pass@k}

1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
]

其中 (n) 是采样数量，(c) 是正确答案数量。

------

# 三十、可解释性

1）Attention 可视化
2）Attention 是否等于解释
3）探针 Probing
4）Activation Analysis
5）Feature Visualization
6）Mechanistic Interpretability
7）Circuit
8）Induction Head
9）Neuron Activation
10）Sparse Autoencoder
11）Logit Lens
12）Causal Intervention
13）Activation Patching
14）模型编辑

------

# 三十一、安全与对齐

1）Prompt Injection
2）Jailbreak
3）Data Poisoning
4）Model Extraction
5）Membership Inference
6）隐私泄露
7）训练数据记忆
8）有害内容
9）偏见与公平性
10）拒答
11）安全微调
12）Red Teaming
13）Guardrail
14）输入过滤
15）输出过滤
16）工具权限隔离
17）模型对齐与能力损失

------

# 三十二、常见模型架构差异

1）GPT：Decoder-only
2）BERT：Encoder-only
3）T5：Encoder-Decoder
4）LLaMA 类架构
5）Mistral 类架构
6）MoE 模型
7）多模态 LLM
8）视觉语言模型
9）语音语言模型
10）Diffusion Language Model
11）State Space Model
12）Mamba
13）Hybrid Architecture

重点比较：

| 架构            | 典型用途              | Attention Mask             |
| --------------- | --------------------- | -------------------------- |
| Encoder-only    | 理解、分类、Embedding | 双向                       |
| Decoder-only    | 文本生成              | 因果 Mask                  |
| Encoder-Decoder | 翻译、摘要            | Encoder 双向，Decoder 因果 |

------

# 三十三、多模态模型

1）视觉编码器
2）语言模型
3）视觉 Token
4）Projection Layer
5）Cross-Attention
6）Early Fusion
7）Late Fusion
8）图文对齐
9）对比学习
10）Image Captioning
11）Visual Question Answering
12）OCR
13）视频 Tokenization
14）音频编码
15）多模态指令微调
16）多模态幻觉

------

# 三十四、Scaling Law 与涌现能力

1）参数规模
2）训练数据规模
3）训练计算量
4）Compute-Optimal Training
5）Chinchilla Law
6）模型欠训练
7）数据不足
8）Loss 与规模的幂律关系
9）Emergent Abilities
10）涌现是真实现象还是评估尺度效应
11）小模型与大模型的能力差异
12）Inference-Time Scaling

------

# 三十五、推理能力

1）Chain of Thought
2）Hidden Reasoning
3）Self-Consistency
4）Tree of Thoughts
5）Best-of-N
6）Verifier
7）Process Reward Model
8）Outcome Reward Model
9）Test-Time Compute
10）Search
11）Reflection
12）Self-Correction
13）Reasoning Distillation
14）数学推理
15）代码推理
16）长链条推理的错误累积

------

# 三十六、面试优先级

准备时间有限时，建议按以下顺序。

1）Transformer 整体结构
2）Self-Attention 公式与复杂度
3）Multi-Head Attention
4）位置编码与 RoPE
5）FFN、SwiGLU、RMSNorm
6）交叉熵、KL、Perplexity
7）Tokenizer 与 BPE
8）预训练目标
9）SFT、RLHF、DPO
10）LoRA、QLoRA
11）推理解码方法
12）KV Cache、GQA、MQA
13）FlashAttention
14）量化
15）分布式训练
16）RAG
17）MoE
18）长上下文
19）模型评估
20）幻觉与安全

