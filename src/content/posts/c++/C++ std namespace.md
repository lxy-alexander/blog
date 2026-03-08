---
title: "C++ std namespace"
published: 2026-03-08
description: "C++ std namespace"
image: ""
tags: ["c++","C++ std namespace"]
category: c++
draft: false
lang: ""
---


# **I. `std` Namespace (`std` 命名空间)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std</code> is the <span style="color:#E8600A;font-weight:700">Namespace (命名空间)</span> that contains all symbols defined by the <span style="color:#2980B9">C++ Standard Library (C++ 标准库)</span> — including <code>vector</code>, <code>string</code>, <code>cout</code>, <code>sort</code>, and more.
</div>

---

## <span style="color:#E8600A">1.</span> **Two ways to use it**

### 1) Explicit prefix (显式前缀) — recommended in production

```cpp
std::vector<int> v;
std::cout << "hello\n";
std::sort(v.begin(), v.end());
```

### 2) `using namespace std` — convenient for competitive programming

```cpp
using namespace std;
vector<int> v;
cout << "hello\n";
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>using namespace std</code> in header files is considered <span style="color:#C0392B;font-weight:600">bad practice (不良实践)</span> — it pollutes the global namespace and can cause <span style="color:#C0392B;font-weight:600">Name Collision (命名冲突)</span> for anyone including your header.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::</code> prefix is always safe; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">using namespace std</code> is fine in <code>.cpp</code> files but <span style="color:#C0392B;font-weight:600">never put it in headers</span>.
</div>