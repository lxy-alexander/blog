---
title: "C++ Reference in Parameters"
published: 2026-03-07
description: "C++ Reference in Parameters"
image: ""
tags: ["c++","C++ Reference in Parameters"]
category: c++
draft: false
lang: ""
---

# **I. Reference in Function Parameters (函数参数中的引用)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Passing by <span style="color:#E8600A;font-weight:700">Reference (引用)</span> avoids copying the entire object. The rule of thumb: <span style="color:#E8600A;font-weight:700">Modify → <code>&</code></span>, <span style="color:#E8600A;font-weight:700">Read only → <code>const &</code></span>, <span style="color:#2980B9">small primitive type → pass by value</span>.
</div>

---

## <span style="color:#E8600A">1.</span> **Why use references (为什么用引用)**

### 1) Avoid copying (避免拷贝)

```cpp
void f(vector<int> v);   // ❌ copies the entire vector
void f(vector<int>& v);  // ✅ no copy, direct access
```

### 2) In-place modification (原地修改)

```cpp
void increment(int& x) { x++; }  // modifies the original
```

---

## <span style="color:#E8600A">2.</span> **Decision Rule (判断规则)**

| Intent | Parameter style | Example |
|---|---|---|
| Modify the argument (修改参数) | `T&` | `void sort(vector<int>& v)` |
| Read only, avoid copy (只读，避免拷贝) | `const T&` | `void print(const string& s)` |
| Small type, just copy (小类型直接传值) | `T` | `void f(int x)` |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>"Small types (小类型)" means primitives like <code>int</code>, <code>char</code>, <code>double</code> — copying them is essentially free. For large objects like <code>vector</code>, <code>string</code>, always use a reference.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<span style="color:#E8600A;font-weight:700">Modify → <code>&</code></span> · <span style="color:#E8600A;font-weight:700">Read only → <code>const &</code></span> · <span style="color:#2980B9">small primitive → value</span>
</div>