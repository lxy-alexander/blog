---
title: "PPO|GRPO Core Concepts and Relationships"
published: 2026-09-29
description: "PPO|GRPO Core Concepts and Relationships"
image: ""
tags: ["llm_inference","PPO|GRPO Core Concepts and Relationships"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-29T18:21:38.930.020261418Z"
---

# PPO|GRPO Core Concepts and Relationships

>    \[ \text{Prompt} \rightarrow \text{Policy/Actor 生成回答} \rightarrow \text{Reward} \rightarrow \text{计算 Advantage} \rightarrow \text{PPO / GRPO 更新 Policy} \]

更准确的 LLM 在线 RL 流程是：

\[ \text{Prompt} \rightarrow \underbrace{\mu\text{ 生成回答}}_{\text{behavior/rollout policy}} \rightarrow \text{Reward} \rightarrow \text{Advantage/Return} \rightarrow \underbrace{\pi_\theta\text{ 计算新 logprob}}_{\text{training policy}} \rightarrow \text{Policy Loss} \rightarrow \text{更新参数} \]

其中 rollout policy \(\mu\) 和正在训练的 policy \(\pi_\theta\) 在同步训练时通常很接近；异步训练时可能是旧版本模型。

其中 PPO 和 GRPO 是“怎么训练”，GAE 是“怎么算 advantage”，Critic 是“帮助算 advantage 的模型”。

不完全是。你画的流程是一个很好的 **在线 Policy Gradient / LLM RL 通用骨架**：

\[ \text{采样轨迹} \rightarrow \text{计算奖励} \rightarrow \text{构造训练信号} \rightarrow \text{更新 Policy} \]

但不同算法在“Reward 怎么来、是否显式计算 Advantage、是否需要 Critic、数据是否由当前 Policy 生成”等方面差别很大。

更准确的 LLM 在线 RL 流程是：

\[ \text{Prompt} \rightarrow \underbrace{\mu\text{ 生成回答}}_{\text{behavior/rollout policy}} \rightarrow \text{Reward} \rightarrow \text{Advantage/Return} \rightarrow \underbrace{\pi_\theta\text{ 计算新 logprob}}_{\text{training policy}} \rightarrow \text{Policy Loss} \rightarrow \text{更新参数} \]

其中 rollout policy \(\mu\) 和正在训练的 policy \(\pi_\theta\) 在同步训练时通常很接近；异步训练时可能是旧版本模型。

## PPO 和 GRPO 的具体流程

### PPO：Actor-Critic

==**A verifier can be either a rule-based function or a learned model.**==
==**Verifier 可以是规则函数，也可以是训练出来的模型。==**

```
Prompt
  → Old Actor 生成回答
  → Reward Model / Verifier 产生 reward
  → Critic 预测每个 token 的 V(s)
  → GAE 计算 advantage 和 return
  → PPO clipped loss 更新 Actor
  → Value loss 更新 Critic
```

公式上：

\[ \delta_t=r_t+\gamma V(s_{t+1})-V(s_t) \]

\[ \hat A_t=\delta_t+\gamma\lambda\hat A_{t+1} \]

\[ \rho_t= \frac{\pi_\theta(a_t|s_t)} {\pi_{\text{old}}(a_t|s_t)} \]

\[ L_{\text{actor}} = -\min\left( \rho_t\hat A_t, \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t \right) \]

同时训练 Critic：

\[ L_V=(V_\phi(s_t)-\hat R_t)^2 \]

因此 PPO 不只是更新 Policy，还会更新 Critic。

### GRPO：通常没有 Critic

```
同一个 Prompt
  → Policy 生成 G 个回答
  → Verifier 分别计算 G 个 reward
  → 组内中心化/归一化得到 advantage
  → clipped policy loss 更新 Policy
```

\[ \hat A_i = \frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)} {\operatorname{std}(R_1,\ldots,R_G)+\epsilon} \]

GRPO 用组内其他回答代替 Critic 充当 baseline，所以不预测 \(V(s)\)。

## 其他在线 Policy Gradient 算法

它们大体符合你的流程，但 advantage 的构造方式不同：

| 算法                 | Reward 后如何得到训练信号         | Critic |
| -------------------- | --------------------------------- | ------ |
| REINFORCE            | 直接使用累计回报 \(G_t\)          | 不需要 |
| REINFORCE + baseline | \(A=G-b\)                         | 可选   |
| PPO/GAE              | \(A\) 由 \(V(s)\) 和 GAE 得到     | 需要   |
| GRPO                 | reward 减组均值，可能除组标准差   | 不需要 |
| Dr.GRPO              | reward 减组均值，通常不除组标准差 | 不需要 |
| RLOO                 | reward 减去其他回答的均值         | 不需要 |
| ReMax                | reward 减 greedy 回答的 reward    | 不需要 |
| A2C/A3C              | TD error / GAE                    | 需要   |

例如 RLOO：

\[ A_i = R_i-\frac{1}{G-1}\sum_{j\ne i}R_j \]

ReMax：

\[ A_i = R_{\text{sample},i}-R_{\text{greedy}} \]

它们都有“生成—奖励—更新”的结构，但不一定训练价值模型。

## 不是所有 Policy 训练都走这个流程

### SFT

```
Prompt + 标准答案
  → Cross-Entropy Loss
  → 更新 Policy
```

没有 rollout、reward 和 advantage：

\[ L_{\text{SFT}} = -\sum_t\log\pi_\theta(y_t^*|x,y_{<t}^*) \]

### DPO / IPO / KTO 等离线偏好优化

```
Prompt + preferred answer + rejected answer
  → 计算两个回答的 logprob
  → Preference loss
  → 更新 Policy
```

通常不需要在线生成，也没有显式 Reward Model 或 Advantage：

\[ L_{\text{DPO}} = -\log\sigma\left( \beta \left[ \log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)} \right] \right) \]

它可以被解释为隐式 reward 优化，但代码里通常不会出现 `reward → advantage` 这两个步骤。

### 蒸馏

```
Prompt
  → Teacher 提供答案或概率分布
  → KL / Cross-Entropy
  → 更新 Student Policy
```

也不一定有 reward 和 advantage。

### Off-policy RL

例如 SAC、DDPG、Q-learning：

```
环境交互
  → 数据放入 Replay Buffer
  → 从 Buffer 采样旧数据
  → 更新 Q/V 网络
  → 根据 Q 更新 Policy
```

数据不一定由当前 Policy 生成，而且核心训练信号可能是 \(Q(s,a)\)，而不是显式 advantage。

## 更合适的统一抽象

现代 LLM RL 框架通常可以拆成四个相对独立的组件：

```
1. Rollout：谁生成训练回答
2. Reward：谁给回答打分
3. Credit Assignment：reward 如何变成 token-level advantage
4. Policy Objective：如何利用 advantage 更新模型
```

例如：

| 算法      | Rollout              | Reward      | Credit assignment | Policy objective         |
| --------- | -------------------- | ----------- | ----------------- | ------------------------ |
| PPO       | old policy           | RM/Verifier | Critic + GAE      | PPO clip                 |
| GRPO      | policy，每题多个回答 | RM/Verifier | group baseline    | 通常 PPO-style clip      |
| RLOO      | policy，每题多个回答 | RM/Verifier | leave-one-out     | policy gradient / clip   |
| REINFORCE | policy               | RM/Verifier | return            | policy gradient          |
| DPO       | 离线偏好数据         | 隐式        | 无显式 advantage  | pairwise preference loss |
| SFT       | 标准答案             | 无          | 无                | cross-entropy            |

因此，你的流程适合概括 **PPO、GRPO、RLOO、REINFORCE 等在线策略梯度算法**，但不适合概括所有 Policy 训练方法。尤其要注意：

>   GRPO 和 PPO 的主要差异往往不在“最后是否使用 clipped policy loss”，而在于 advantage 是由 Critic + GAE 得到，还是由组内 reward 比较得到。



1）**Policy（策略）**：The strategy that determines what action to take in a given state. 决定在当前状态下采取什么动作的策略。

2）**Actor（执行者）**：The model that follows the policy and takes actions. 按照 Policy 实际执行动作的模型。

3）**Reward（奖励）**：A score given to an action or response. 对一个动作或回答给出的分数。

4）**Reward Model（奖励模型）**：A model that assigns a reward score to a response. 负责给回答打分的模型。

5）**Return（回报）**：==The actual accumulated future rewards== starting from the current step. 从当前这一步开始，未来 Reward 的累计。

6）**Critic（评价者）**：A model that estimates the expected future return from the current state. 估计当前状态未来预期 Return 的模型。

7）**Value（价值）**：The critic's estimate of the expected future return from a state. Critic 对当前状态未来 Return 的预测值。

8）**Advantage（优势）**：Measures how much better or worse an action is compared with a baseline. 衡量一个动作相比基准好多少或差多少。

9）**GAE（广义优势估计）**：A method for estimating advantage using rewards and value estimates. 利用 Reward 和 Value 来估计 Advantage 的方法。

10）**PPO（近端策略优化）**：A policy optimization algorithm that uses advantage to update the policy while limiting large changes. 利用 Advantage 更新 Policy，同时限制更新幅度的算法。

11）**GRPO（组相对策略优化）**：A policy optimization algorithm that estimates advantage by comparing rewards within a group. 通过比较组内 Reward 来计算 Advantage 并更新 Policy 的算法。

12）**Clipping（截断）**：Limits how much the policy can change in one update. 限制 Policy 一次更新的幅度。

13）**KL Divergence（KL 散度）**：Measures the difference between two probability distributions. 衡量两个概率分布之间的差异。

14）**State（状态）**：The information available to the model at the current step. 模型在当前这一步拥有的信息。

15）**Action（动作）**：The choice made by the model in the current state. 模型在当前状态下做出的选择；在 LLM 中通常是下一个 token。

16）**Old Policy（旧策略）**：The policy that generated the current batch of training data. 生成当前这批训练数据的 Policy。

17）**Reference Policy（参考策略）**：A fixed policy used as a reference during training. 训练时用于约束当前 Policy 的固定参考模型。

18）**Probability Ratio（概率比）**：Measures how an action's probability changes from the old policy to the new policy. 衡量某个 Action 在新旧 Policy 下的概率变化。

## 1）Policy / Actor 是什么

Policy 就是你真正想训练的模型。

比如输入：

\[ \text{What is the capital of France?} \]

模型生成：

\[ \text{Paris} \]

这个负责“做动作”的模型就叫：

\[ \boxed{\text{Policy}} \]

或者：

\[ \boxed{\text{Actor}} \]

在 LLM 里：

\[ \text{action} = \text{生成下一个 token} \]

所以可以记：

\[ \boxed{\text{Actor = 负责生成}} \]

------

## 2）Reward 是什么

模型生成完整回答以后，我们需要判断：

“这个回答好不好？”

于是得到一个分数：

\[ R \]

例如：

\[ \text{Answer A} \rightarrow R = 0.2 \]

\[ \text{Answer B} \rightarrow R = 0.8 \]

Reward 就是：

\[ \boxed{\text{回答得到的分数}} \]

它可能来自：

人工评分、Reward Model、规则验证器、数学答案是否正确等。

所以：

\[ \boxed{\text{Reward = 最终成绩}} \]

------

## 3）Reward Model 是什么

Reward Model 和 Reward 不是一回事。

Reward Model 是一个“打分模型”。

输入：

\[ \text{Prompt + Response} \]

输出：

\[ R=0.8 \]

所以：

\[ \boxed{\text{Reward Model = 老师}} \]

\[ \boxed{\text{Reward = 老师给的分数}} \]

例如：

\[ RM(q,y)=0.8 \]

这里 RM 是 Reward Model，0.8 才是 reward。

------

## 4）为什么有 Reward 还需要 Advantage

因为 reward 只告诉你：

“这个回答得了 8 分。”

但训练更关心：

“这个回答相对于正常水平，到底好不好？”

例如：

第一组大家都是：

\[ [1,2,3] \]

你得到：

\[ 3 \]

那很不错。

但另一组大家：

\[ [98,99,100] \]

你得到：

\[ 3 \]

就非常差。

所以我们需要一个“相对表现”：

\[ \boxed{\text{Advantage}} \]

可以理解成：

\[ \boxed{\text{Advantage = 比基准好多少}} \]

因此：

Reward 是绝对分数。

Advantage 是相对表现。

------

## 5）Advantage 是什么

最直觉的形式：

\[ A=\text{实际表现}-\text{预期表现} \]

如果：

\[ A>0 \]

说明比预期好。

如果：

\[ A<0 \]

说明比预期差。

例如：

\[ \text{实际 reward}=8 \]

预期：

\[ 6 \]

那么：

\[ A=8-6=2 \]

也就是说：

\[ \boxed{\text{Advantage = 比预期好 2}} \]

------

## 6）Critic 是什么

现在问题来了：

“预期表现”从哪里来？

PPO 里通常会有一个模型来预测\[ V(s) \]

这个模型就是：

\[ \boxed{\text{Critic}} \]

Critic 的工作不是生成答案，而是预测：

\[ \boxed{\text{从当前状态继续下去，预计能拿多少 reward}} \]

例如模型已经生成：

>   The capital of France is ...

Critic 可能预测：

\[ V(s_t)=0.7 \]

意思是：

“从现在这个状态继续生成，预计最终表现大约值 0.7。”

所以：

\[ \boxed{\text{Actor = 做}} \]

\[ \boxed{\text{Critic = 估值}} \]

------

## 7）Value 是什么

Critic 输出的东西叫\[ V(s_t) \]

叫做：

\[ \boxed{\text{Value}} \]

所以关系是：

\[ \boxed{\text{Critic 是模型}} \]\[ \boxed{V(s_t) 是 Critic 输出的值} } \]

就像：

Reward Model 是模型，

Reward 是它输出的分数。

同理：

Critic 是模型，

Value 是它输出的预测值。

------

## 8）GAE 是什么

有了 reward 和 critic 的 value 后，我们就可以估计 advantage。

其中一种常见方法叫：

\[ \boxed{\text{GAE}} \]

全称：

Generalized Advantage Estimation。

它首先计算：

\[ \delta_t = r_t+\gamma V(s_{t+1})-V(s_t) \]

你可以先简单理解成：

\[ \boxed{ \delta_t = \text{实际得到的东西} - \text{原本预测的东西} } \]

然后把多个时间步的信息结合起来：

\[ A_t^{GAE} = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} +\cdots \]

最终得到：

\[ \boxed{A_t} \]

所以：

\[ \boxed{ Reward + Critic \xrightarrow{GAE} Advantage } \]

注意：

GAE 不是模型。

GAE 也不是完整训练算法。

它只是：

\[ \boxed{\text{一种计算 Advantage 的方法}} \]

------

## 9）PPO 是什么

现在 advantage 已经算出来了。

例如：

\[ A_t=+0.8 \]

这说明：

“刚才这个行为挺好。”

接下来要更新 Policy，让以后更容易做出类似行为。

PPO 就负责这一步。

所以：

\[ \boxed{ Advantage \rightarrow PPO \rightarrow 更新 Policy } \]

但是 PPO 有一个特点：

不能更新得太猛。

它会计算新旧模型概率比例：

\[ r_t(\theta) = \frac{ \pi_\theta(a_t|s_t) }{ \pi_{\theta_{\text{old}}}(a_t|s_t) } \]

然后用 clipping：

\[ L^{PPO} = \min \left( r_tA_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t \right) \]

所以一句话：

\[ \boxed{\text{PPO = 用 Advantage 更新 Policy，但限制更新幅度}} \]

------

## 10）】经 PPO 流程是什么

把前面全部串起来：

\[ \text{Prompt} \]

↓

\[ \text{Actor / Policy 生成回答} \]

↓

\[ \text{Reward Model 给 Reward} \]

↓

\[ \text{Critic 给 } V(s) \]

↓

\[ \text{GAE 根据 Reward + Value 算 Advantage} \]

↓

\[ \text{PPO 根据 Advantage 更新 Policy} \]

也就是：

\[ \boxed{ Reward + Critic \xrightarrow{GAE} Advantage \xrightarrow{PPO} Policy } \]

这条是最重要的。

------

## 11）GRPO 又是什么

GRPO 做了一件很关键的事情：

它说：

“我能不能不用 Critic？”

可以。

对于同一个 prompt，生成多个回答：

\[ y_1,y_2,y_3,y_4 \]

得到 reward：

\[ R=[1,2,3,4] \]

然后直接比较这一组回答。

例如：

\[ A_i = \frac{ R_i-\operatorname{mean}(R) }{ \operatorname{std}(R)+10^{-6} } \]

这样就能得到 advantage。

比如：

\[ R_4=4 \]

平均：

\[ 2.5 \]

标准差约：

\[ 1.12 \]

那么：

\[ A_4 = \frac{4-2.5}{1.12} \approx1.34 \]

因此：

\[ \boxed{ Group Rewards \rightarrow Advantage } \]

不需要再用 critic 来预测 baseline。

所以 GRPO 的核心思想是：

\[ \boxed{ \text{用同组回答作为彼此的 baseline} } \]

------

## 12）PPO 和 GRPO 到底是什么关系

它们是同一个层级。

都是：

\[ \boxed{\text{Policy Optimization Algorithm}} \]

即“策略优化算法”。

可以写：

\[ \boxed{ PPO \quad vs \quad GRPO } \]

它们主要区别在 advantage 怎么来。

PPO 常见：

\[ Reward + Critic \xrightarrow{GAE} Advantage \]

GRPO：

\[ Group\ Rewards \xrightarrow{\text{Relative Comparison}} Advantage \]

然后两者都会：

\[ Advantage \rightarrow 更新 Policy \]

------

## 13）GAE 和 GRPO 不是同一级

这个特别重要。

GAE：

\[ \boxed{\text{Advantage estimation method}} \]

GRPO：

\[ \boxed{\text{Policy optimization algorithm}} \]

所以不能直接说：

“GAE vs GRPO”

它们不是一个层面的东西。

更合理的是：

\[ \boxed{\text{PPO vs GRPO}} \]

以及：

\[ \boxed{\text{GAE vs group-relative advantage estimation}} \]

------

## 14）那你之前看到“每个 token 都 +0.8”是什么

假设一整个回答得到：

\[ A=+0.8 \]

回答有：

\[ token_1,\ token_2,\ token_3,\ token_4 \]

有些 GRPO 实现中，会把这个 sequence-level advantage 用到每个 token：

\[ +0.8,\quad+0.8,\quad+0.8,\quad+0.8 \]

这并不是：

\[ P(token)+0.8 \]

而是：

“这些 token 都获得同一个正向训练信号。”

因此：

\[ A>0 \]

表示提高这些 token 被生成的倾向。

\[ A<0 \]

表示降低这些 token 被生成的倾向。

------

## 15）还有 Return 是什么

这个也很容易和 Reward 混。

Reward：

\[ r_t \]

是某一步得到的奖励。

Return：

\[ G_t \]

是从现在开始，未来 reward 的累计：

\[ G_t = r_t + \gamma r_{t+1} + \gamma^2r_{t+2} +\cdots \]

所以：

\[ \boxed{\text{Reward = 一次得分}} \]\[ \boxed{\text{Return = 后面所有 reward 的累计}} \]

Critic 实际上想预测的就是：

\[ V(s_t) \approx E[G_t] \]

即：

\[ \boxed{\text{预计未来总共还能赚多少}} \]

------

## 16）最终把所有概念放进一个层级图

最上层：

\[ \boxed{\text{Reinforcement Learning}} \]

里面有一个：

\[ \boxed{\text{Policy / Actor}} \]

负责生成行为。

然后环境或 Reward Model 给：

\[ \boxed{\text{Reward}} \]

为了判断这个 reward 到底好不好，需要：

\[ \boxed{\text{Advantage}} \]

Advantage 可以有不同算法算。

PPO 常见：

\[ \boxed{ Critic \rightarrow Value } \]

然后：

\[ \boxed{ Reward + Value \xrightarrow{GAE} Advantage } \]

最后：

\[ \boxed{ Advantage \xrightarrow{PPO} 更新 Policy } \]

而 GRPO：

\[ \boxed{ 一组 Responses \rightarrow 一组 Rewards } \]

然后：

\[ \boxed{ \frac{R_i-\text{mean}(R)} {\text{std}(R)} \rightarrow Advantage } \]

最后：

\[ \boxed{ Advantage \xrightarrow{GRPO} 更新 Policy } \]

所以最简洁的对照是：

| 概念           | 它是什么  | 作用                                     |
| -------------- | --------- | ---------------------------------------- |
| Policy / Actor | 模型      | 生成 token                               |
| Reward Model   | 模型      | 给回答打分                               |
| Reward         | 数值      | 回答得多少分                             |
| Critic         | 模型      | 预测未来 reward                          |
| Value \(V(s)\) | 数值      | Critic 的预测                            |
| Return         | 数值      | 未来 reward 总和                         |
| Advantage      | 数值      | 比预期好/差多少                          |
| GAE            | 计算方法  | 用 reward + value 算 advantage           |
| PPO            | 训练算法  | 用 advantage 更新 policy                 |
| GRPO           | 训练算法  | 用组内 reward 算 advantage 并更新 policy |
| Clip           | 技巧      | 防止 policy 更新过猛                     |
| KL             | 距离/约束 | 防止新模型偏离参考模型太远               |

最后只背这一句就够：

\[ \boxed{ \text{PPO: Reward + Critic → GAE → Advantage → PPO → Policy} } \]\[ \boxed{ \text{GRPO: Group Rewards → Relative Advantage → GRPO → Policy} } \]

也就是说，**PPO 和 GRPO 是“大方法”；GAE 是 PPO 里常见的 advantage 计算工具；Critic 是 GAE 常用来估计 baseline/value 的模型。**
