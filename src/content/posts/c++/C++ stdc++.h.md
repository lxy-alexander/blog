---
title: "C++ stdc++.h"
published: 2026-03-07
description: "C++ stdc++.h"
image: ""
tags: ["c++","C++ stdc++.h"]
category: c++
draft: false
lang: ""
---

# **I. `#include <bits/stdc++.h>` (万能头文件)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">#include &lt;bits/stdc++.h&gt;</code> is a <span style="color:#E8600A;font-weight:700">Global Header File (全局头文件)</span> provided by <span style="color:#2980B9">GCC/Clang</span> compilers. It includes all common C++ standard library headers in one line — popular in competitive programming (竞技编程) for speed, but <span style="color:#C0392B;font-weight:600">not recommended in production code (生产代码)</span>.
</div>

---

## <span style="color:#E8600A">1.</span> **What it does (作用)**

```cpp
#include <bits/stdc++.h>
using namespace std;
```

Equivalent to including all of: `<vector>`, `<string>`, `<algorithm>`, `<map>`, `<set>`, `<iostream>`, `<cmath>`, etc. — in one line.

---

## <span style="color:#E8600A">2.</span> **When to use (适用场景)**

| Scenario | Recommendation |
|---|---|
| Competitive programming (竞技编程) | ✅ Use it — saves time |
| Production / real projects (生产代码) | <span style="color:#C0392B;font-weight:600">❌ Avoid</span> — increases compile time, non-portable |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>This header is <span style="color:#C0392B;font-weight:600">not part of the C++ standard</span> — it is a GCC/Clang extension and may not work on MSVC (Visual Studio).
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">&lt;bits/stdc++.h&gt;</code> is a GCC-only shortcut — great for competitive programming, <span style="color:#C0392B;font-weight:600">avoid in real projects</span>.
</div>