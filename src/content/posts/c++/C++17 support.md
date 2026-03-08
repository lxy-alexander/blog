---
title: "C++17 support"
published: 2026-03-08
description: "C++17 support"
image: ""
tags: ["c++","C++17 support"]
category: c++
draft: false
lang: ""
---


# **I. Check C++17 Support (验证编译器是否支持 C++17)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
To verify whether <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">g++</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clang++</code> supports C++17, use the <span style="color:#E8600A;font-weight:700">Preprocessor (预处理器)</span> to dump the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__cplusplus</code> macro value. If it outputs <code>201703L</code>, C++17 is active.
</div>

---

## <span style="color:#E8600A">1.</span> **The Command (检测命令)**

```shell
g++     -std=c++17 -x c++ -dM -E - < /dev/null | grep __cplusplus
clang++ -std=c++17 -x c++ -dM -E - < /dev/null | grep __cplusplus
```

Expected output:

```
#define __cplusplus 201703L
```

---

## <span style="color:#E8600A">2.</span> **Flag Breakdown (参数逐一解析)**

### 1) Compiler & standard (编译器与标准)

| Flag | Meaning |
|---|---|
| `g++` / `clang++` | Invoke the C++ compiler frontend (C++ 编译器前端) |
| `-std=c++17` | Use the <span style="color:#E8600A;font-weight:700">C++17 Language Standard (C++17 语言标准)</span> |

### 2) Input control (输入控制)

| Flag | Meaning |
|---|---|
| `-x c++` | Force treat input as C++ source (强制当作 C++ 源码) |
| `-` | Read source from <span style="color:#2980B9">stdin (标准输入)</span> |
| `< /dev/null` | Feed an <span style="color:#2980B9">empty input (空输入)</span> to stdin |

### 3) Preprocessor output (预处理输出)

| Flag | Meaning |
|---|---|
| `-dM` | Dump all <span style="color:#E8600A;font-weight:700">Predefined Macros (预定义宏)</span> |
| `-E` | Run <span style="color:#E8600A;font-weight:700">Preprocessor only (仅预处理阶段)</span>, skip compilation |

### 4) Filtering (过滤)

| Flag | Meaning |
|---|---|
| `\|` | <span style="color:#2980B9">Pipe (管道)</span> — pass output of left command to right command |
| `grep __cplusplus` | Show only the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__cplusplus</code> macro line |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>The <code>__cplusplus</code> macro is a standard integer that encodes the active C++ standard version: <code>199711L</code> = C++98, <code>201103L</code> = C++11, <code>201402L</code> = C++14, <code>201703L</code> = C++17, <code>202002L</code> = C++20.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">-dM -E - &lt; /dev/null</code> dumps all predefined macros without compiling any real file — pipe to <code>grep __cplusplus</code> to confirm which C++ standard is active.
</div>