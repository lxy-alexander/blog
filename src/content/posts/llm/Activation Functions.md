---
title: "Activation Functions"
published: 2026-03-09
description: "Activation Functions"
image: ""
tags: ["llm","Activation Functions"]
category: llm
draft: false
lang: ""
---


# **I. Activation Functions (激活函数)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Activation Functions (激活函数) introduce <span style="color:#E8600A;font-weight:700">Non-linearity (非线性)</span> into Neural Networks (神经网络). Without them, multiple layers collapse into a <span style="color:#E8600A;font-weight:700">Linear Model (线性模型)</span>.  
Historically, new activation functions were proposed to address problems such as <span style="color:#E8600A;font-weight:700">Vanishing Gradient (梯度消失)</span>, <span style="color:#E8600A;font-weight:700">Dead Neurons (神经元死亡)</span>, and <span style="color:#E8600A;font-weight:700">Optimization Instability (优化不稳定)</span>.  
</div>

---

## <span style="color:#E8600A">1.</span> Sigmoid Function (Sigmoid 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Sigmoid Function (Sigmoid函数)</span> maps any real number to the interval **(0,1)**.

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

### 2) Motivation

Sigmoid was originally introduced for <span style="color:#E8600A;font-weight:700">Probability Modeling (概率建模)</span> in <span style="color:#E8600A;font-weight:700">Logistic Regression (逻辑回归)</span>.

### 3) Python Example

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

print(sigmoid([-2,-1,0,1,2]))
```

### 4) Limitation

<span style="color:#C0392B;font-weight:600">Vanishing Gradient (梯度消失)</span>

When inputs are large, the derivative becomes nearly **0**, preventing gradients from propagating through deep networks.

### 5) Consequence

This motivated the creation of <span style="color:#2980B9">Tanh Activation (Tanh 激活函数)</span>.

---

## <span style="color:#E8600A">2.</span> Tanh Function (Tanh 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Hyperbolic Tangent Function (双曲正切函数)</span> outputs values in **(-1,1)**.

$$
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

### 2) Improvement over Sigmoid

<span style="color:#2980B9">Key idea:</span>

Tanh produces <span style="color:#E8600A;font-weight:700">Zero-Centered Activations (零中心激活)</span>.

This improves optimization during <span style="color:#E8600A;font-weight:700">Gradient Descent (梯度下降)</span>.

### 3) Python Example

```python
import numpy as np

def tanh(x):
    return np.tanh(x)

print(tanh([-2,-1,0,1,2]))
```

### 4) Remaining Problem

Tanh still suffers from

<span style="color:#C0392B;font-weight:600">Vanishing Gradient (梯度消失)</span>.

### 5) Consequence

This motivated the introduction of <span style="color:#2980B9">ReLU (Rectified Linear Unit，修正线性单元)</span>.

---

## <span style="color:#E8600A">3.</span> ReLU Function (ReLU 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Rectified Linear Unit (修正线性单元)</span> is defined as

$$
ReLU(x)=max(0,x)
$$

### 2) Motivation

ReLU was introduced to address

<span style="color:#E8600A;font-weight:700">Vanishing Gradient (梯度消失)</span>

because its derivative is **1 when x > 0**.

### 3) Python Example

```python
import numpy as np

def relu(x):
    return np.maximum(0,x)

print(relu([-2,-1,0,1,2]))
```

### 4) Limitation

<span style="color:#C0392B;font-weight:600">Dying ReLU Problem (ReLU死亡问题)</span>

Neurons receiving negative inputs may output **0 permanently**.

### 5) Consequence

This motivated improved variants such as

* <span style="color:#E8600A;font-weight:700">Leaky ReLU (带泄漏ReLU)</span>
* <span style="color:#E8600A;font-weight:700">ELU (Exponential Linear Unit，指数线性单元)</span>

---

## <span style="color:#E8600A">4.</span> Leaky ReLU (Leaky ReLU 激活函数)

### 1) Definition

Leaky ReLU introduces a small slope for negative inputs.

$$
f(x)=
\begin{cases}
x & x>0 \
\alpha x & x\le0
\end{cases}
$$

### 2) Motivation

It solves

<span style="color:#E8600A;font-weight:700">Dying ReLU (神经元死亡)</span>

by allowing gradients for negative inputs.

### 3) Python Example

```python
import numpy as np

def leaky_relu(x,alpha=0.01):
    return np.where(x>0,x,alpha*x)

print(leaky_relu([-2,-1,0,1,2]))
```

---

## <span style="color:#E8600A">5.</span> ELU Function (ELU 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Exponential Linear Unit (指数线性单元)</span> is defined as

$$
ELU(x)=
\begin{cases}
x & x>0 \
\alpha(e^x-1) & x\le0
\end{cases}
$$

### 2) Motivation

ELU further improves ReLU by:

* Allowing **negative outputs**
* Maintaining **non-zero gradients**

This reduces bias shift during training.

### 3) Python Example

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0,-1.0,0.0,1.0,2.0])
print(F.elu(x))
```

---

## <span style="color:#E8600A">6.</span> GELU Function (GELU 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Gaussian Error Linear Unit (高斯误差线性单元)</span> is defined as

$$
GELU(x)=x\Phi(x)
$$

where <span style="color:#E8600A;font-weight:700">Φ(x)</span> is the <span style="color:#E8600A;font-weight:700">Gaussian CDF (高斯累积分布函数)</span>.

Approximation:

$$
GELU(x)\approx0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right)
$$

### 2) Motivation

GELU introduces

<span style="color:#E8600A;font-weight:700">Smooth Probabilistic Activation (平滑概率激活)</span>.

Instead of hard gating like ReLU, it performs <span style="color:#E8600A;font-weight:700">Probabilistic Gating (概率门控)</span>.

### 3) Python Example

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0,-1.0,0.0,1.0,2.0])
print(F.gelu(x))
```

### 4) Usage

Widely used in

* <span style="color:#E8600A;font-weight:700">BERT</span>
* <span style="color:#E8600A;font-weight:700">GPT</span>
* <span style="color:#E8600A;font-weight:700">Transformer Models (Transformer模型)</span>

---

## <span style="color:#E8600A">7.</span> Swish Function (Swish 激活函数)

### 1) Definition

The <span style="color:#E8600A;font-weight:700">Swish Activation (Swish 激活函数)</span> is defined as

$$
Swish(x)=x \cdot \sigma(x)
$$

### 2) Motivation

Swish was discovered through <span style="color:#E8600A;font-weight:700">Neural Architecture Search (神经架构搜索)</span>.

It provides

* smooth gradients
* improved performance in deep models.

### 3) Python Example

```python
import torch

def swish(x):
    return x * torch.sigmoid(x)

print(swish(torch.tensor([-2.,-1.,0.,1.,2.])))
```
---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Activation Functions (激活函数) evolved from <span style="color:#E8600A;font-weight:700">Sigmoid → Tanh → ReLU → Leaky ReLU → ELU → GELU → Swish</span>, where each new function was introduced to solve optimization problems such as <span style="color:#E8600A;font-weight:700">Vanishing Gradient (梯度消失)</span>, <span style="color:#E8600A;font-weight:700">Dead Neurons (神经元死亡)</span>, and improve gradient smoothness in Deep Neural Networks (深度神经网络).
</div>

---

# **II. Why Activation Functions Introduce Non-linearity (为什么激活函数能引入非线性)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
The key purpose of Activation Functions (激活函数) is to introduce <span style="color:#E8600A;font-weight:700">Non-linearity (非线性)</span> into Neural Networks (神经网络).  
If a neural network only performs <span style="color:#E8600A;font-weight:700">Linear Transformations (线性变换)</span>, then stacking multiple layers still results in an equivalent single linear transformation.
</div>

---

## <span style="color:#E8600A">1.</span> Linear Composition Problem (线性组合问题)

Each layer in a Neural Network (神经网络) typically computes

$$
y = Wx + b
$$

where

- <span style="color:#E8600A;font-weight:700">W</span> is the <span style="color:#E8600A;font-weight:700">Weight Matrix (权重矩阵)</span>  
- <span style="color:#E8600A;font-weight:700">b</span> is the <span style="color:#E8600A;font-weight:700">Bias (偏置)</span>

Stacking two linear layers gives

$$
y_1 = W_1x + b_1
$$

$$
y_2 = W_2y_1 + b_2
$$

Substituting

$$
y_2 = W_2(W_1x + b_1) + b_2
$$

which simplifies to

$$
y_2 = W'x + b'
$$

<div style="background:#F5F5F5;border-left:4px solid #2980B9;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<b style="color:#2980B9">Observation:</b><br>
Multiple linear layers collapse into a single <span style="color:#E8600A;font-weight:700">Linear Transformation (线性变换)</span>.  
Therefore the neural network behaves like a <span style="color:#E8600A;font-weight:700">Linear Model (线性模型)</span>.
</div>

---

## <span style="color:#E8600A">2.</span> Role of Non-linear Functions (非线性函数的作用)

Activation Functions apply a

<span style="color:#E8600A;font-weight:700">Non-linear Mapping (非线性映射)</span>

$$
y = f(Wx+b)
$$

Stacking layers produces

$$
f(W_2 f(W_1 x))
$$

Because the function <span style="color:#E8600A;font-weight:700">f(x)</span> is non-linear, the network can no longer be simplified into a single linear transformation.

This enables neural networks to approximate complex functions using

<span style="color:#E8600A;font-weight:700">Universal Function Approximation (通用函数逼近)</span>.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
The <span style="color:#E8600A;font-weight:700">Universal Approximation Theorem (通用逼近定理)</span> states that a neural network with at least one non-linear activation function can approximate any continuous function.
</div>

---

## <span style="color:#E8600A">3.</span> Example: XOR Problem (异或问题)

The classic <span style="color:#E8600A;font-weight:700">XOR Problem (异或问题)</span> cannot be solved using a <span style="color:#E8600A;font-weight:700">Linear Classifier (线性分类器)</span>.

However, when we introduce <span style="color:#E8600A;font-weight:700">Non-linear Activation Functions (非线性激活函数)</span>, a neural network can represent the XOR decision boundary.

---

# **III. Evolution of Activation Functions (激活函数演化)**

| Era | Activation | Problem Addressed |
|---|---|---|
| 1980s | Sigmoid | Probability modeling |
| 1990s | Tanh | Zero-centered outputs |
| 2010 | ReLU | Solve vanishing gradient |
| 2013 | Leaky ReLU | Solve dying ReLU |
| 2015 | ELU | Improve negative activations |
| 2017 | GELU | Smooth probabilistic activation |
| 2019 | Swish | Improve gradient smoothness |

---
<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Activation Functions (激活函数) introduce <span style="color:#E8600A;font-weight:700">Non-linearity (非线性)</span>, allowing Neural Networks (神经网络) to represent complex functions, while the evolution from <span style="color:#E8600A;font-weight:700">Sigmoid → Swish</span> reflects improvements in gradient flow and training stability.
</div>
