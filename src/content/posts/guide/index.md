---
title: Build Guides
published: 2024-04-01
description: "How to buid the blog website"
image: "./cover.jpeg"
tags: ["Blogging", "Build Guides"]
category: Guides
draft: false
---


一、准备环境

1. 安装 Node.js（版本 >= 18）
2. 全局安装 pnpm

   npm install -g pnpm

3. 已创建 Astro 博客项目
4. 博客内容以 Markdown 文件为主
5. 项目已推送至 GitHub 仓库

---

二、安装项目依赖

进入项目根目录，执行：

pnpm install

---

三、本地运行博客（开发模式）

执行：

pnpm dev

终端会显示：

Local http://localhost:4321

在浏览器中打开该地址即可访问博客。

该模式说明：
- 仅在本机可访问
- 支持热更新
- 仅用于开发和预览

---

四、本地构建静态站点

执行：

pnpm build

构建完成后会生成 dist 目录。

dist 目录中只包含：
- HTML
- CSS
- JavaScript

这是最终用于部署的静态文件。

---

五、配置 Astro 以适配 GitHub Pages

编辑 astro.config.mjs。

如果仓库名不是 username.github.io：

import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://你的用户名.github.io",
  base: "/仓库名/",
  output: "static",
});

如果仓库名是 username.github.io：

import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://你的用户名.github.io",
  output: "static",
});

---

六、创建 GitHub Actions 自动部署流程

在项目根目录创建文件：

.github/workflows/deploy.yml

内容如下：

name: Deploy Astro to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 18
      - run: npm install -g pnpm
      - run: pnpm install
      - run: pnpm build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/deploy-pages@v4

---

七、开启 GitHub Pages

1. 打开 GitHub 仓库 Settings
2. 进入 Pages
3. Source 选择 GitHub Actions
4. 保存设置

---

八、触发自动部署

执行：

git add .
git commit -m "deploy blog"
git push

GitHub Actions 会自动运行：
- 安装依赖
- 构建项目
- 发布到 GitHub Pages

访问地址：

https://你的用户名.github.io/仓库名/

---

九、使用 Markdown 写博客

Markdown 文件可放在：
- src/pages
- src/content

开发模式下修改会即时生效，部署时会自动编译为静态页面。

---

十、去除页面多余 icon

1. 去除 favicon  
   删除或注释 layout 中的：

   <link rel="icon" href="/favicon.svg" />

   或直接删除 public 目录下的 favicon 文件。

2. 去除 Header / Footer icon  
   在 src/components 或 src/layouts 中删除类似：

   <Icon name="github" />

   或 img 图标元素。

3. Markdown 页面默认不包含 icon，无需额外处理。

---

十一、完整流程汇总

pnpm install
pnpm dev
pnpm build
git push
GitHub Pages 自动更新

该流程一经配置即可长期使用，无需频繁维护。
