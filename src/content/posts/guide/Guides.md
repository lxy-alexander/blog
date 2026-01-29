---
title: Build Guides
published: 2026-01-01
description: "How to buid the blog website"
image: "./cover.jpeg"
tags: ["Build Guides"]
category: guides
draft: false
---

## 🚀  博客构建全流程指南

这是一份基于 **Fuwari** 模板与 **Astro** 框架的博客构建全流程指南。本指南整合了从本地环境搭建到 GitHub Pages 自动化部署的完整步骤。

### 1. 环境准备

在开始之前，请确保你的设备已安装以下工具：

* **Node.js**: 版本需  18。
* **pnpm**: 推荐的包管理器。安装命令：`npm install -g pnpm`
* **Git**: 用于代码版本管理。

---

### 2. 初始化项目

你可以通过以下两种方式之一创建博客：

* **方式 A（推荐）**: 访问 [Fuwari 模板页](https://github.com/saicaca/fuwari/generate) 直接生成新仓库。
* **方式 B (命令行)**:

```sh
pnpm create fuwari@latest

```



创建完成后，克隆仓库至本地并安装依赖：

```sh
pnpm install

```

---

### 3. 本地开发与配置

在正式发布前，你需要进行个性化设置。

* **基础配置**: 编辑 `src/config.ts` 修改站点标题、作者、社交链接等。
* **运行预览**: 执行 `pnpm dev`。
* 访问 `http://localhost:4321` 查看实时效果。
* 该模式支持**热更新**，修改文件后浏览器会自动刷新。


* **清理图标**: 若需删除默认图标，可在 `public/` 目录下替换 `favicon.svg`，或在 `src/components` 中移除相关的 `<Icon />` 标签。

---

### 4. 撰写内容

Fuwari 使用 Markdown 存储文章，位置在 `src/content/posts/`。

* **创建新文章**:

```sh
pnpm new-post <文件名>

```


* **配置 Frontmatter**: 在 `.md` 文件顶部配置元数据：

```yaml
---
title: 我的第一篇文章
published: 2026-01-22
description: 文章描述
image: ./cover.jpg
tags: [教程, Astro]
category: 技术
draft: false
---

```



---

### 5. GitHub Pages 部署配置

为了让全球用户都能访问，我们需要配置自动化部署。

#### A. 修改 Astro 配置

编辑 `astro.config.mjs`，确保路径正确：

```javascript
export default defineConfig({
  site: "https://<你的用户名>.github.io",
  base: "/<仓库名>/", // 如果仓库名是 username.github.io，则此项留空或填 "/"
});

```

#### B. 创建部署脚本 (GitHub Actions)

在项目根目录创建 `.github/workflows/deploy.yml`，并粘贴以下核心配置：

```sh
name: Deploy to GitHub Pages

on:
  # 每次推送到 `main` 分支时触发这个“工作流程”
  # 如果你使用了别的分支名，请按需将 `main` 替换成你的分支名
  push:
    branches: [ main ]
  # 允许你在 GitHub 上的 Actions 标签中手动触发此“工作流程”
  workflow_dispatch:

# 允许 job 克隆 repo 并创建一个 page deployment
permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout your repository using git
        uses: actions/checkout@v5
      - name: Install, build, and upload your site
        uses: withastro/action@v5
        # with:
          # path: . # 存储库中 Astro 项目的根位置。（可选）
          # node-version: 20 # 用于构建站点的特定 Node.js 版本，默认为 20。（可选）
          # package-manager: pnpm@latest # 应使用哪个 Node.js 包管理器来安装依赖项和构建站点。会根据存储库中的 lockfile 自动检测。（可选）
          # build-cmd: pnpm run build # 用于构建你的网站的命令。默认运行软件包的构建脚本或任务。（可选）
        # env:
          # PUBLIC_POKEAPI: 'https://pokeapi.co/api/v2' # 对变量值使用单引号。（可选）

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```



> 该脚本会在你每次 `git push` 时自动完成安装、构建并发布。

#### C. 开启 GitHub 设置

1. 进入 GitHub 仓库 **Settings** -> **Pages**。
2. 在 **Build and deployment** 下的 **Source** 选项中，选择 **GitHub Actions**。

---

### 6. 发布上线

执行以下命令将代码推送到 GitHub：

```sh
git add .
git commit -m "Initial blog setup"
git push origin main
```

**检查进度**: 在 GitHub 仓库的 **Actions** 选项卡中可以看到部署进度。完成后，你的博客将运行在 `https://<用户名>.github.io/<仓库名>/`。

---

### 💡 常用命令速查表

| 命令                   | 作用                                   |
| ---------------------- | -------------------------------------- |
| `pnpm dev`             | 启动本地开发服务器                     |
| `pnpm build`           | 执行本地构建（生成 `dist` 静态文件夹） |
| `pnpm new-post <name>` | 快速创建新文章模板                     |
| `pnpm format`          | 自动格式化美化代码                     |

### 💡 常用Markdown语法速查表

| 语法          | 作用 |
| ------------- | ---- |
| `==content==` | 高亮 |
