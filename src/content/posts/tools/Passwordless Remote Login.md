---
title: "Passwordless Remote Login"
published: 2026-01-30
description: "Passwordless Remote Login"
image: ""
tags: ["tools","Passwordless Remote Login"]
category: tools
draft: false
lang: ""
---

> 目标：在 Mac/Windows/Linux 本机登录远程服务器（如 `spiedie.binghamton.edu`）时实现**免输入密码**，并让 VS Code Remote-SSH 更稳定。

---

## 1. 核心知识：SSH 免密登录原理（公钥/私钥认证）

SSH “免密登录”不是不验证身份，而是把“输入密码认证”换成“密钥认证”。

### 1.1 两个关键文件
- **私钥（Private Key）**：保存在本机  
  例：`~/.ssh/id_ed25519`  
  ✅ 绝对不能泄露  
- **公钥（Public Key）**：可以发给服务器  
  例：`~/.ssh/id_ed25519.pub`  
  ✅ 可以公开/复制到服务器

### 1.2 服务器如何记住你？
服务器会把你的公钥写入：

```bash
~/.ssh/authorized_keys
```

之后服务器就会允许“持有对应私钥的人”登录。

### 1.3 登录认证发生了什么（简化流程）

1.  你发起连接：`ssh user@host`
2.  服务器从 `authorized_keys` 找到你的公钥
3.  服务器发一个随机挑战（challenge）
4.  你用 **私钥** 在本机对挑战签名（私钥不离开本机）
5.  服务器用 **公钥** 验证签名
6.  验证通过 → 登录成功（不再输入账号密码）

------

## 2. 免密登录操作流程（推荐步骤）

以下示例用你的服务器信息：

-   用户名：`xli49`
-   主机：`spiedie.binghamton.edu`

------

### Step 1：本机生成 SSH Key

先检查有没有 key：

```bash
ls ~/.ssh/id_ed25519 ~/.ssh/id_rsa 2>/dev/null
```

没有就生成（推荐 ed25519）：

```bash
ssh-keygen -t ed25519 -C "xli49@spiedie.binghamton.edu"
```

>   ✅ 想“完全免输入”：passphrase 直接回车留空
>   ✅ 更安全：设置 passphrase（可配合 Keychain / ssh-agent）

------

### Step 2：把公钥拷贝到服务器（免密的关键）

✅ 推荐（有 `ssh-copy-id` 时）：

```bash
ssh-copy-id xli49@spiedie.binghamton.edu
```

输入一次密码即可。

#### 如果没有 `ssh-copy-id`（手动方式）

```bash
cat ~/.ssh/id_ed25519.pub | ssh xli49@spiedie.binghamton.edu \
"mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```

------

### Step 3：测试免密是否成功

```bash
ssh xli49@spiedie.binghamton.edu
```

✅ 不再要求输入密码 → 成功

------

## 3. 配置 `~/.ssh/config`（更方便 + 更稳）

编辑：

```bash
nano ~/.ssh/config
```

推荐写法（用域名当 Host）：

```sshconfig
Host spiedie.binghamton.edu
    HostName spiedie.binghamton.edu
    User xli49
    ServerAliveInterval 300
    ServerAliveCountMax 120
```

之后连接直接：

```bash
ssh spiedie.binghamton.edu
```

------

## 4. `IdentityFile` 是否需要？

### 4.1 一般情况下不需要

因为 SSH 默认会自动尝试：

-   `~/.ssh/id_ed25519`
-   `~/.ssh/id_rsa`
-   ssh-agent 里已经加载的 key

所以 key 在默认位置时，通常不用写 `IdentityFile`。

### 4.2 必须写 `IdentityFile` 的情况

✅ 当你有多把 key，SSH 可能选错
✅ key 不在默认路径
✅ 服务器要求使用特定 key

示例：

```sshconfig
Host spiedie.binghamton.edu
    HostName spiedie.binghamton.edu
    User xli49
    IdentityFile ~/.ssh/id_ed25519
```

------

## 5. Remote-SSH 经常掉线：保活原理与设置（推荐开启）

掉线常见原因：

-   学校网络/防火墙会清理长时间“空闲连接”
-   远程长时间无输出被认为 idle

解决：启用 SSH 心跳包（KeepAlive）

```sshconfig
ServerAliveInterval 30
ServerAliveCountMax 120
```

含义：

-   每 30 秒发一次心跳包
-   最多允许 120 次无响应（约 1 小时）才断开

------

## 6. 排错工具

### 6.1 看看 SSH 到底用的是哪把 key

```bash
ssh -v xli49@spiedie.binghamton.edu
```

重点看类似：

-   `Offering public key: ...`
-   `Authentication succeeded`

### 6.2 服务器端检查默认 shell（避免错误配置）

远程执行：

```bash
echo $SHELL
getent passwd xli49 | cut -d: -f7
which zsh
```

------

