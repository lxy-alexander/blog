---
title: "PR"
published: 2026-09-20
description: "PR"
image: ""
tags: ["llm_inference","PR"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-09-20T04:08:11.542.267467359Z"
---

你现在的简化流程如下。

### 1. 认证

继续使用 HTTPS，无需配置 SSH 或 GPG：

````
Signing Key 不是从 GitHub 下载的，而是在本机生成，然后把公钥添加到 GitHub。建议使用 SSH Signing Key，比配置 GPG 简单。

生成并保存在持久化目录：

```bash
install -d -m 700 /mnt/lxy/.ssh

ssh-keygen -t ed25519 -a 100 \
  -f /mnt/lxy/.ssh/github_signing \
  -C "lxy_alexander@outlook.com"

chmod 600 /mnt/lxy/.ssh/github_signing
chmod 644 /mnt/lxy/.ssh/github_signing.pub
```

显示公钥：

```bash
cat /mnt/lxy/.ssh/github_signing.pub
```

进入 GitHub：

```text
Settings
→ SSH and GPG keys
→ New SSH key
→ Key type: Signing Key
```

粘贴 `.pub` 文件的完整内容。不要上传没有 `.pub` 后缀的私钥。

然后配置 vLLM 仓库：

```bash
cd /mnt/lxy/vllm

git config --local user.name "lxy-alexander"
git config --local user.email "lxy_alexander@outlook.com"
git config --local gpg.format ssh
git config --local user.signingkey /mnt/lxy/.ssh/github_signing
git config --local commit.gpgsign true
```

提交时仍然要保留 vLLM 的 DCO：

```bash
git commit -s -m "[Bugfix][MoE] Restore WNA16 runtime dispatch"
```

此时：

- `commit.gpgsign=true` 自动进行 SSH 加密签名；
- `-s` 添加 vLLM 所需的 `Signed-off-by`；
- 推送到 GitHub 后，commit 应显示 **Verified**。


先把 `uv` 安装到持久化目录 `/mnt/lxy/bin`，这样更换 GPU 实例后文件还在：

```bash
mkdir -p /mnt/lxy/bin

curl -LsSf https://astral.sh/uv/install.sh |
  env UV_INSTALL_DIR=/mnt/lxy/bin sh

export PATH="/mnt/lxy/bin:$PATH"

uv --version
```

然后在 vLLM 仓库创建 Python 3.12 环境：

```bash
cd /mnt/lxy/vllm

uv venv --python 3.12
source .venv/bin/activate
```

确认环境：

```bash
which uv
which python
uv --version
python --version
```

预计分别指向：

```text
/mnt/lxy/bin/uv
/mnt/lxy/vllm/.venv/bin/python
```

之后重新运行你的安装命令即可。

每次创建新 GPU 实例后，只需重新设置 PATH 和激活环境：

```bash
export PATH="/mnt/lxy/bin:$PATH"
cd /mnt/lxy/vllm
source .venv/bin/activate
```

不需要再次下载 `uv`，前提是 `/mnt/lxy` 确实是持久化磁盘。
````

只要 `origin` 是：

```
git clone https://github.com/lxy-alexander/vllm.git
```

就可以正常推送。

### 2. 配置提交身份

```
git config --local user.name "lxy-alexander"
git config --local user.email "lxy_alexander@outlook.com"
```

### 3. 同步分支

这台机器没有配置名为 `upstream` 的远端。`upstream` 只是本地别名，不会随 GitHub 仓库自动创建。

先执行一次：

```
cd /mnt/lxy/vllm

git remote -v

git remote add upstream https://github.com/vllm-project/vllm.git
git fetch upstream main
```

然后同步分支：

```
git switch fix/moe-wna16-runtime-dispatch
git rebase upstream/main
git push --force-with-lease -u origin fix/moe-wna16-runtime-dispatch
```

以后只需要：

```
git switch fix/moe-wna16-runtime-dispatch
git fetch upstream main
git rebase upstream/main
git push --force-with-lease
```

远端应当是：

```
origin    你的 fork：lxy-alexander/vllm
upstream  官方仓库：vllm-project/vllm
```

可用下面命令确认：

```
git remote -v
```

如果提示 `remote upstream already exists`，则修正地址：

```
git remote set-url upstream https://github.com/vllm-project/vllm.git
```



### 4. 修改并检查

```
git status
git diff
```

只添加本次修改的文件，不要直接 `git add .`：

```
git add <本次修改的文件>
```

### 5. 运行测试

```
.venv/bin/python -m pytest tests/quantization/test_moe_wna16.py -v
```

然后运行相关 pre-commit：

```
.venv/bin/pre-commit run --files <修改的文件>
```

### 6. 提交

vLLM 需要 DCO，使用小写 `-s`；不需要 GPG 的大写 `-S`：

```
git commit -s \
  -m "[Bugfix][MoE] Restore WNA16 runtime dispatch" \
  -m "Co-authored-by: OpenAI Codex <noreply@openai.com>"
```

### 7. 推送

```
git push
```

如果后来 rebase 了已有提交：

```
git push --force-with-lease
```

### 8. 提交 PR 前检查重复工作

```
gh issue view 53241 --repo vllm-project/vllm --comments

gh pr list --repo vllm-project/vllm --state open \
  --search "53241 in:body"

gh pr list --repo vllm-project/vllm --state open \
  --search "WNA16 runtime dispatch"
```

如果 `gh` 没有认证：

```
gh auth login
```

最终概括：**HTTPS 负责推送，`git commit -s` 负责 vLLM 的 DCO；SSH 和 GPG 都不是必需的。**
