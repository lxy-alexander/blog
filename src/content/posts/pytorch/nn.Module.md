---
title: "nn.Module"
published: 2026-04-27
description: "nn.Module"
image: ""
tags: ["pytorch","nn.Module"]
category: pytorch
draft: false
lang: ""
createdAt: "2026-04-27T16:54:08.912.501496281Z"
---
# `nn.Module` (神经网络模块基类)

>   `nn.Module` is ==the base class (基类) for all PyTorch models== — it provides ==parameter tracking== (参数追踪), ==device management== (设备管理), and ==serialization== (序列化), so subclasses only need to ==define `__init__` and `forward`.==

------

## 1. Lifecycle (生命周期)

`nn.Module` enforces a two-method contract (两方法约定) that separates structure from computation.

### 1) `__init__` — Structure (结构定义)

`__init__` registers submodules (子模块) and parameters (参数) into ==PyTorch's internal registry== (内部注册表) via `super().__init__()`. Skipping `super().__init__()` leaves the registry uninitialized — every subsequent attribute assignment silently fails to register.

### 2) `forward` — Computation (计算定义)

`forward` defines ==the computation graph (计算图) traced by autograd (自动微分)== on each call. Call the module as `model(x)` rather than `model.forward(x)` — the `__call__` wrapper fires registered hooks (钩子) before and after `forward`.

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)  # auto-registered (自动注册)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))

model = MLP(4, 2)
print(model(torch.randn(3, 4)).shape)  # torch.Size([3, 2])
```

------

## 2. Parameter Management (参数管理)

`nn.Module` distinguishes three kinds of named tensors stored inside a module.

### 1) `nn.Parameter` — Learnable (可学习参数)

`nn.Parameter` wraps a tensor so that `requires_grad=True` by default and it appears in `model.parameters()`. Use it for weights that the optimizer (优化器) must update — plain tensors assigned as attributes are invisible to the optimizer.

### 2) `register_buffer` — Non-learnable State (非可学习状态)

`register_buffer` attaches a tensor to the module that moves with `.to(device)` but is excluded from `parameters()`. Prefer it over plain attributes for fixed tensors like running statistics (运行统计量) in BatchNorm.

```python
import torch
import torch.nn as nn

class NormLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))        # learnable (可学习)
        self.register_buffer("running_mean", torch.zeros(dim))  # non-learnable (非可学习)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) * self.weight

model = NormLayer(4)
print(dict(model.named_parameters()).keys())  # weight only
print(dict(model.named_buffers()).keys())     # running_mean only
```

### 3) `named_parameters` vs `parameters` (命名参数 vs 参数迭代器)

`parameters()` yields tensors for the optimizer; `named_parameters()` yields `(name, tensor)` pairs for debugging or selective freezing (选择性冻结). Freeze a layer by setting `param.requires_grad = False` — the optimizer skips tensors where `requires_grad` is false.

------

## 3. Hooks (钩子)

Hooks intercept (拦截) the forward and backward passes without modifying `forward` itself.

### 1) Forward Hook (前向钩子)

`register_forward_hook(fn)` fires after `forward` completes, receiving `(module, input, output)`. Use it for activation logging (激活值记录) or feature extraction (特征提取) without altering model code.

### 2) Backward Hook (反向钩子)

`register_full_backward_hook(fn)` fires during the backward pass (反向传播), receiving `(module, grad_input, grad_output)`. Use it to inspect or clip gradients (梯度裁剪) at the module level; the trade-off is a slight overhead on every backward call.

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
activations = {}

def fwd_hook(module, inp, out):
    activations["linear"] = out.detach()

handle = model.register_forward_hook(fwd_hook)
model(torch.randn(3, 4))
print(activations["linear"].shape)  # torch.Size([3, 2])
handle.remove()  # always remove hooks when done (用完及时移除)
```

>   **Note:** Always call `handle.remove()` after use — unreleased hooks accumulate (积累) and slow down every forward pass.

------

## 4. `state_dict` and Serialization (`state_dict` 与序列化)

`state_dict` is the canonical (规范) way to save and restore model weights in PyTorch.

### 1) Save and Load (保存与加载)

`state_dict()` returns an `OrderedDict` of all parameters and buffers keyed by their registered names. Prefer `torch.save(model.state_dict(), path)` over pickling the entire model — it decouples weights from the class definition (解耦权重与类定义), making loading robust across code refactors.

### 2) `load_state_dict` — `strict` Flag (`strict` 标志)

`load_state_dict(sd, strict=True)` raises an error on any key mismatch (键不匹配); set `strict=False` when loading a pretrained backbone (预训练骨干网络) into a model with extra heads — missing or unexpected keys are silently ignored.

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
torch.save(model.state_dict(), "/tmp/weights.pt")

# Restore on any device (在任意设备恢复)
new_model = nn.Linear(4, 2)
new_model.load_state_dict(torch.load("/tmp/weights.pt", map_location="cpu"))
print(new_model(torch.randn(3, 4)).shape)  # torch.Size([3, 2])
```

------

## 5. Training vs Eval Mode (训练模式 vs 推理模式)

`.train()` and `.eval()` toggle (切换) the behavior of stateful layers like Dropout (随机失活) and BatchNorm (批归一化).

### 1) `.train()` / `.eval()` — Mode Switch (模式切换)

`.train()` enables Dropout and uses per-batch statistics in BatchNorm; ==`.eval()` disables Dropout and switches BatchNorm to its running statistics (运行统计量).== 

==Forgetting `.eval()` at inference (推理) is one of the most common bugs in PyTorch== — Dropout randomly zeroes activations and BatchNorm uses noisy batch stats instead of the learned ones.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))

model.train()
out_train = model(torch.randn(3, 4))  # dropout active (激活)

model.eval()
with torch.no_grad():
    out_eval = model(torch.randn(3, 4))  # dropout disabled (禁用)

print(out_train.shape, out_eval.shape)  # both torch.Size([3, 2])
```

>   **Note:** ==`torch.no_grad()` disables gradient tracking (梯度追踪)== for memory efficiency but does not switch layer behavior — always pair it with `.eval()` at inference.

------

>   **Summary:** `nn.Module` tracks parameters via `nn.Parameter` and `register_buffer`, intercepts passes via hooks, serializes via `state_dict`, and gates layer behavior via `.train()` / `.eval()` — mastering these five mechanisms covers the vast majority of real-world PyTorch interview questions.