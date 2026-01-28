---
title: "Astro"
published: 2026-01-28
description: "Astro"
image: ""
tags: ["guide","Astro"]
category: guide
draft: false
lang: ""
---



## mark/highlight

1）安装我们自己插件需要的依赖

```shell
npm i unist-util-visit
```

2）在你的项目里创建一个文件（路径一定要对）
`src/plugins/remark-mark.mjs`

文件内容完整复制进去：

```js
import { visit } from "unist-util-visit";

export default function remarkMark() {
  return (tree) => {
    visit(tree, "text", (node, index, parent) => {
      if (!node.value || !parent) return;

      const next = node.value.replace(/==(.+?)==/g, "<mark>$1</mark>");
      if (next === node.value) return;

      node.type = "html";
      node.value = next;
    });
  };
}
```

3）打开项目根目录的 `astro.config.mjs`，在 import 区域加上这一行（一定要放在 `export default defineConfig(...)` 之前）

```js
import remarkMark from "./src/plugins/remark-mark.mjs";
```

4）把 `remarkMark` 加进 `remarkPlugins`（建议放第一个）

```js
markdown: {
  remarkPlugins: [
    remarkMark,
    remarkMath,
    remarkReadingTime,
    remarkExcerpt,
    remarkGithubAdmonitionsToDirectives,
    remarkDirective,
    remarkSectionize,
    parseDirectiveNode,
  ],
  rehypePlugins: [
    // ...
  ],
},
```
