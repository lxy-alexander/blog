---
title: "Git"
published: 2026-08-18
description: "Git"
image: ""
tags: ["llm_inference","Git"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-18T23:22:41.060.411662047Z"
---



## 修复临时 cherry-pick 到当前分支

如果只是想继续你自己的 `fix/remount-put-timeout` 工作，最合适的是把 #3515 的修复临时 cherry-pick 到当前分支：

```
git fetch upstream pull/3515/head:pr-3515

git switch fix/remount-put-timeout
git cherry-pick 02f918a9eb1aae0b8e5f43b2b17da1385e9c1838
```

然后重新编译。等 #3515 合入 `main` 后，再执行：

```
git fetch upstream
git rebase upstream/main
```

如果只是想验证 #3515，不修改自己的分支：

```
git fetch upstream pull/3515/head:pr-3515
git switch pr-3515
```

不建议再从 `a2055ae7` 开发，它只是临时绕过故障，会缺少后续主线提交。

结论：

-   做你自己的功能：留在 `fix/remount-put-timeout`，cherry-pick #3515。
-   验证上游修复：切到 `pr-3515`。
-   不需要再单独实现同一份 bug 修复，因为 #3515 已经在处理。

