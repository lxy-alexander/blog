---
title: "C++ Integer Division Trunction"
published: 2026-03-08
description: "C++ Integer Division Trunction"
image: ""
tags: ["c++","C++ Integer Division Trunction"]
category: c++
draft: false
lang: ""
---


# **I. Integer Division Truncation (整数除法截断)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
In C++, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int / int</code> performs <span style="color:#E8600A;font-weight:700">Integer Division (整数除法)</span> — it <span style="color:#C0392B;font-weight:600">truncates the decimal part</span>. To get a floating-point result, cast at least one operand to <code>double</code> or <code>float</code> before dividing.
</div>

---

## <span style="color:#E8600A">1.</span> **The Problem (问题)**

```cpp
int maxSum = 51;
int k = 4;
return maxSum / k;        // ❌ returns 12, not 12.75
```

`51 / 4 = 12` because both operands are `int` — the decimal `.75` is lost.

---

## <span style="color:#E8600A">2.</span> **Two Fixes (两种修复方式)**

### 1) C-style cast (C 风格强制转换)

```cpp
return (double)maxSum / k;     // ✅ 12.75
```

### 2) Multiply by 1.0 (乘以浮点数)

```cpp
return maxSum * 1.0 / k;       // ✅ 12.75
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Cast only one operand — as soon as one side is <code>double</code>, C++ automatically promotes (隐式提升) the other to <code>double</code> before dividing.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int / int</code> always truncates — use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(double)a / b</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">a * 1.0 / b</code> to get a real division result.
</div>