---
title: "Compile&Run vLLM"
published: 2026-02-09
description: "Compile&Run vLLM"
image: ""
tags: ["vllm","Compile&Run vLLM"]
category: vllm
draft: false
lang: ""
---





## ② sudo（纯用户安装）

可以手动装到家目录：

```
mkdir -p $HOME/local
cd $HOME/local

wget https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz
tar xf ccache-4.10.2-linux-x86_64.tar.xz

ln -s $HOME/local/ccache-4.10.2-linux-x86_64/ccache $HOME/.local/bin/ccache
```

确保 PATH：

```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

验证：

```
ccache --version
```
