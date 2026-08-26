---
title: "Triton Contributor"
published: 2026-08-26
description: "Triton Contributor"
image: ""
tags: ["content","posts","Triton Contributor"]
category: content / posts
draft: false
lang: ""
createdAt: "2026-08-26T04:45:58.226.704296521Z"
---





conda create -n triton-toolchain -y \

​    python=3.12 \

​    "cmake>=3.20,<4" ninja \

​    gcc_linux-64=12 gxx_linux-64=12 \

​    -c conda-forge



conda activate triton-toolchain

