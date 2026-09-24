---
title: "RL"
published: 2026-09-23
description: "RL"
image: ""
tags: ["llm_inference","RL"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-24T03:08:38.549.972627620Z"
---

1）中文

### 语言建模

自回归语言模型（Autoregressive Language Model）
定义：模型按照从左到右的方式生成文本，每个 token 的预测都依赖前面已经出现的 token。

$P_\theta(x)=\prod_{t=1}^{T}P_\theta(x_t|x_{

例子：
输入“今天天气很”，模型会根据前面的“今天天气很”预测下一个 token，例如“好”。

负对数似然损失（Negative Log-Likelihood, NLL）
定义：训练时让模型尽可能提高真实下一个 token 的概率。真实 token 的概率越低，损失越大。

$L(\theta)=-\mathbb{E}\left[\sum_t\log P_\theta(x_t|x_{

交叉熵损失（Cross-Entropy Loss）
定义：衡量模型预测的 token 概率分布和真实 token 之间的差异，是训练语言模型最常见的损失函数。

Transformer
定义：现代大语言模型最常用的神经网络架构。核心是自注意力机制，让模型判断上下文中哪些 token 最重要。

Self-Attention（自注意力）
定义：模型在处理一个 token 时，可以参考上下文中的其他 token，并给不同 token 分配不同的重要程度。

LM Head
定义：Transformer 最后的输出层，把模型内部的隐藏表示映射到整个词表中的 token 概率。

例子：
模型内部产生一个向量，LM Head 将其转换成：

“苹果”：0.4
“香蕉”：0.2
“汽车”：0.01

------

### KL 散度

KL Divergence（KL 散度）

定义：衡量两个概率分布之间有多大差异。

$D_{KL}(P||Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}$

在 RLHF 中，它经常用于限制新模型不要偏离原模型太远。

例子：
原模型倾向：

A：60%
B：40%

训练后变成：

A：5%
B：95%

KL 散度会比较大。

------

### NLP 基本概念

Prompt
定义：输入给模型的文本。

例子：
“解释一下什么是 RLHF。”

Completion
定义：模型根据 Prompt 生成的回答。

Chosen Completion
定义：多个回答中，人类更偏好的回答。

Rejected Completion
定义：多个回答中，人类认为较差的回答。

Preference Relation

$y_{chosen}\succ y_{rejected}$

表示人类更偏好 chosen response。

Policy

$\pi_\theta(y|x)$

定义：给定 Prompt $x$，模型生成不同回答 $y$ 的概率分布。

换句话说，在 RLHF 中，语言模型本身就是 policy。

------

### 强化学习基本概念

State（状态）
定义：智能体当前所处的环境信息。

在语言模型里，可以理解为目前已经生成的上下文。

Action（动作）
定义：智能体下一步采取的行为。

在语言模型里，一个 action 通常可以理解为生成下一个 token。

Reward（奖励）
定义：一个数值，用来评价某个行为或回答有多好。

例子：

好回答：+1
差回答：-1

Trajectory（轨迹）
定义：智能体从开始到结束经历的完整状态、动作和奖励序列。

在语言模型中，可以近似理解为一次完整的文本生成过程。

Policy（策略）
定义：在某个状态下选择不同动作的概率。

$\pi(a|s)$

在 LLM 中，就是“根据已有文本，决定下一个 token 的概率”。

Discount Factor（折扣因子，$\gamma$）
定义：决定未来奖励的重要程度。

$0\leq\gamma\leq1$

$\gamma$ 越小，越重视近期奖励；越接近 1，越重视长期奖励。

------

### Value、Q 和 Advantage

Value Function

$V(s)$

定义：从当前状态开始，未来预计能获得多少累计奖励。

Q-Function

$Q(s,a)$

定义：在状态 $s$ 下先采取动作 $a$，之后预计能获得多少累计奖励。

Advantage Function

$A(s,a)=Q(s,a)-V(s)$

定义：衡量某个动作比当前状态下的“平均选择”好多少。

例子：

当前状态平均预期奖励：

$V(s)=5$

采取动作 A：

$Q(s,A)=8$

那么：

$A(s,A)=3$

说明动作 A 比平均选择更好。

------

### 强化学习目标

强化学习的核心目标是：

$\max_\theta E[\text{累计奖励}]$

也就是调整模型参数，让模型长期获得尽可能高的 reward。

Finite Horizon Reward
定义：只计算有限 $T$ 个步骤内的累计奖励。

------

### On-policy 与 Off-policy

On-policy
定义：训练数据由当前正在训练的模型自己生成。

Off-policy
定义：训练数据由其他模型或者旧版本模型生成。

例子：

当前模型生成回答，再拿这些回答训练当前模型：on-policy。

GPT-4 生成数据，用来训练另一个小模型：off-policy。

------

### Reference Model

Reference Model（参考模型）

定义：RLHF 中保留的一个固定模型，用来约束训练后的模型不要偏离原模型太远。

通常会通过 KL penalty 比较：

$\pi_\theta$

和

$\pi_{ref}$

------

### Synthetic Data

合成数据（Synthetic Data）

定义：由 AI 模型生成，而不是人工直接创建的训练数据。

例子：
让一个强模型生成 100 万道数学题及答案，再用这些数据训练另一个模型。

------

### Distillation

蒸馏（Distillation）

定义：使用强模型的输出训练另一个模型。

强模型通常叫 Teacher，训练出来的模型叫 Student。

例子：

GPT 类强模型生成答案
→ 用这些答案训练一个 7B 小模型。

------

### Knowledge Distillation

知识蒸馏（Knowledge Distillation）

定义：学生模型不仅学习教师最终选出的 token，还学习教师对整个词表的概率分布。

例如教师模型预测：

苹果：0.6
香蕉：0.3
汽车：0.01

学生不仅学习“苹果是正确答案”，还学习这种概率结构。

------

### In-context Learning

上下文学习（In-Context Learning, ICL）

定义：不给模型更新参数，而是在 Prompt 中提供信息或示例，让模型临时学会任务。

例子：

输入：

猫 → cat
狗 → dog
苹果 →

模型可能回答：

apple

这就是 few-shot ICL。

------

### Chain of Thought

思维链（Chain of Thought, CoT）

定义：让模型通过中间推理步骤解决复杂问题，而不是直接给最终答案。

例子：

问题：23 × 17 是多少？

CoT：

23 × 10 = 230
23 × 7 = 161
230 + 161 = 391

答案：391。

------

2）English

### Language Modeling

Autoregressive Language Model
Definition: A model generates text sequentially, where each token is predicted based on the tokens that came before it.

$P_\theta(x)=\prod_{t=1}^{T}P_\theta(x_t|x_{

Example:
Given “The weather today is very”, the model predicts the next token based on the previous context.

Negative Log-Likelihood (NLL)
Definition: During training, the model is encouraged to assign high probability to the correct next token. A lower probability for the correct token produces a larger loss.

Cross-Entropy Loss
Definition: A loss function that measures the difference between the model's predicted token distribution and the true token.

Transformer
Definition: The dominant neural-network architecture used in modern large language models. Its central mechanism is self-attention.

Self-Attention
Definition: A mechanism that allows each token to attend to other tokens in the context and assign different levels of importance to them.

LM Head
Definition: The final layer that maps the model's internal hidden representation into probabilities over the vocabulary.

------

### KL Divergence

Kullback-Leibler Divergence

Definition: A measure of how different one probability distribution is from another.

$D_{KL}(P||Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}$

In RLHF, it is commonly used to prevent the trained model from moving too far away from a reference model.

------

### NLP Concepts

Prompt
Definition: The text provided as input to a language model.

Completion
Definition: The output generated by the model in response to the prompt.

Chosen Completion
Definition: The response preferred by a human or preference model.

Rejected Completion
Definition: The less preferred response.

Preference Relation

$y_{chosen}\succ y_{rejected}$

This means the chosen response is preferred over the rejected response.

Policy

$\pi_\theta(y|x)$

Definition: The probability distribution over possible outputs $y$ given a prompt $x$.

In RLHF, the language model itself can be viewed as the policy.

------

### Reinforcement Learning Concepts

State
Definition: The current information available to the agent.

For a language model, this can be viewed as the text generated so far.

Action
Definition: A decision made by the agent.

For an LLM, an action can be viewed as selecting the next token.

Reward
Definition: A scalar value indicating how good an action or output is.

Trajectory
Definition: A complete sequence of states, actions, and rewards experienced by an agent.

For an LLM, it can roughly correspond to one complete generation.

Policy
Definition: A probability distribution over actions given a state.

$\pi(a|s)$

For an LLM, this corresponds to the probability distribution over the next token.

Discount Factor

$\gamma$

Definition: A parameter controlling how much future rewards matter.

A smaller $\gamma$ emphasizes immediate rewards, while a value closer to 1 emphasizes long-term rewards.

------

### Value, Q, and Advantage

Value Function

$V(s)$

Definition: The expected cumulative future reward starting from state $s$.

Q-Function

$Q(s,a)$

Definition: The expected cumulative reward after taking action $a$ in state $s$.

Advantage Function

$A(s,a)=Q(s,a)-V(s)$

Definition: Measures how much better an action is compared with the average action in that state.

------

### Reinforcement Learning Objective

The main goal of reinforcement learning is:

$\max_\theta E[\text{cumulative reward}]$

The model parameters are optimized so that the policy receives higher expected reward.

Finite Horizon Reward
Definition: The expected cumulative reward over a limited number of steps $T$.

------

### On-policy and Off-policy

On-policy
Definition: Training data is generated by the current model being optimized.

Off-policy
Definition: Training data is generated by another model or an older version of the model.

Example:

Current model generates responses and learns from them → on-policy.

A stronger external model generates training responses → off-policy.

------

### Reference Model

Reference Model

Definition: A fixed model used during RLHF to prevent the optimized model from deviating too far from its original behavior.

The difference between

$\pi_\theta$

and

$\pi_{ref}$

is often controlled using KL divergence.

------

### Synthetic Data

Synthetic Data

Definition: Training data generated by another AI system rather than directly created by humans.

Example:
A strong model generates one million math problems and solutions, which are then used to train another model.

------

### Distillation

Distillation

Definition: Training a new model using outputs generated by a stronger model.

The stronger model is often called the teacher, while the trained model is called the student.

------

### Knowledge Distillation

Knowledge Distillation

Definition: The student learns the teacher model's full probability distribution over possible tokens rather than only learning the final selected answer.

For example, the teacher may predict:

Apple: 0.60
Banana: 0.30
Car: 0.01

The student learns this probability structure.

------

### In-Context Learning

In-Context Learning (ICL)

Definition: The model learns how to perform a task from information or examples included in the prompt without changing its parameters.

Example:

Cat → cat
Dog → dog
Apple →

The model predicts:

apple

------

### Chain of Thought

Chain of Thought (CoT)

Definition: A method in which a model uses intermediate reasoning steps to solve a problem rather than immediately producing the final answer.

Example:

23 × 17

23 × 10 = 230
23 × 7 = 161
230 + 161 = 391

Final answer: 391.
