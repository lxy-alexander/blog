---
title: "code"
published: 2026-09-27
description: "code"
image: ""
tags: ["llm_inference","code"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-27T19:42:28.712.840273611Z"
---

# Cross-Entropy

Cross-Entropy 用来衡量模型预测概率和真实标签之间的差异。

Cross-Entropy 对单个样本的公式：

\[ L=-\log p_y \]

其中 \(p_y\) 是真实类别对应的预测概率。

模型输出 logits \(z\) 后，先通过 Softmax：

\[ p_i=\frac{e^{z_i}}{\sum_j e^{z_j}} \]

实际计算时通常先减去最大值，防止 `exp` 数值溢出：

\[ p_i= \frac{e^{z_i-\max(z)}} {\sum_j e^{z_j-\max(z)}} \]

因此：

\[ L=-\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) \]

对于一个 batch：

\[ L=-\frac{1}{N}\sum_{n=1}^{N}\log p_{n,y_n} \]

1）NumPy

```
import numpy as np  # 导入 NumPy

def cross_entropy(logits, labels):  # 定义 Cross-Entropy
    logits = logits - np.max(logits, axis=1, keepdims=True)  # 减去每个样本的最大值，防止 exp 溢出
    exp_logits = np.exp(logits)  # 计算指数
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # 计算 Softmax 概率
    probs = probs[np.arange(len(labels)), labels]  # 取出真实类别对应的概率
    loss = -np.mean(np.log(probs))  # 计算平均 Cross-Entropy
    return loss  # 返回损失

logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])  # 2 个样本，每个样本有 3 个类别的 logits
labels = np.array([0, 1])  # 2 个样本的真实类别分别是 class 0 和 class 1
loss = cross_entropy(logits, labels)  # 计算损失
print(loss)  # 输出结果
```

2）PyTorch

```python
import torch  # 导入 PyTorch
import torch.nn.functional as F  # 仅用于最后测试对比

def cross_entropy(logits, labels):  # 自己实现 Cross-Entropy
    logits = logits - torch.max(logits, dim=1, keepdim=True).values  # 减去最大值，防止 exp 溢出
    exp_logits = torch.exp(logits)  # 计算 e^logits
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)  # 计算 Softmax 概率
    probs = probs[torch.arange(len(labels)), labels]  # 取出每个样本真实类别对应的概率
    loss = -torch.mean(torch.log(probs))  # 计算平均 Cross-Entropy
    return loss  # 返回损失


logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])  # 2 个样本，每个样本有 3 个类别的 logits

labels = torch.tensor([0, 1])  # 2 个样本的真实类别分别是 class 0 和 class 1

loss = cross_entropy(logits, labels)  # 使用自己实现的 Cross-Entropy
print(loss)  # 输出自己实现的结果

loss_api = F.cross_entropy(logits, labels)  # 使用 PyTorch 官方 API
print(loss_api)  # 输出官方 API 的结果进行对比
```

```
import torch  # 导入 PyTorch
import torch.nn.functional as F  # 导入 PyTorch 函数接口

def cross_entropy(logits, labels):  # 定义 Cross-Entropy
    log_probs = F.log_softmax(logits, dim=1)  # 计算 Log-Softmax
    log_probs = log_probs[torch.arange(len(labels)), labels]  # 取出真实类别对应的 Log 概率
    loss = -torch.mean(log_probs)  # 计算平均 Cross-Entropy
    return loss  # 返回损失

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])  # 创建测试 logits
labels = torch.tensor([0, 1])  # 创建测试标签
loss = cross_entropy(logits, labels)  # 计算手写 Cross-Entropy
print(loss)  # 输出手写结果
print(F.cross_entropy(logits, labels))  # 和 PyTorch 官方实现对比
```

```python
import torch  # 导入 PyTorch
import torch.nn.functional as F  # 导入 PyTorch 函数接口

def cross_entropy(logits, labels):  # 定义 Cross-Entropy
    loss = F.cross_entropy(logits, labels)  # 直接调用 PyTorch 的 Cross-Entropy API
    return loss  # 返回损失


logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])  # 2 个样本，每个样本有 3 个类别的 logits

labels = torch.tensor([0, 1])  # 2 个样本的真实类别分别是 class 0 和 class 1

loss = cross_entropy(logits, labels)  # 计算 Cross-Entropy
print(loss)  # 输出损失
```





# SFT

