---
title: "C++ auto&int&size_t"
published: 2026-03-07
description: "C++ auto&int&size_t"
image: ""
tags: ["c++","C++ auto&int&size_t"]
category: c++
draft: false
lang: ""
---

# **I. `auto` / `int` / `size_t` (类型选择)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Three commonly confused types for variables and indices: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">auto</code> lets the compiler deduce the type, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int</code> is a signed general integer, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">size_t</code> is an <span style="color:#E8600A;font-weight:700">Unsigned Type (无符号类型)</span> for sizes and indices.
</div>

---

## <span style="color:#E8600A">1.</span> **When to use each (使用场景)**

| Type | Use when |
|---|---|
| `auto` | Type is obvious from context, or complex type (e.g. iterators, lambdas) |
| `int` | General signed integer, counters, arithmetic |
| `size_t` | Sizes returned by containers (`.size()`), array indices |

---

## <span style="color:#E8600A">2.</span> **Common pitfall: mixing `int` and `size_t` (混用陷阱)**

```cpp
vector<int> v = {1, 2, 3};
for (int i = v.size() - 1; i >= 0; i--)  // ⚠️ v.size() is size_t (unsigned), if v.size() == 0, v.size() - 1 will be a large number
    
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>size_t</code> is <span style="color:#C0392B;font-weight:600">unsigned (无符号)</span> — if you do <code>size_t i = 0; i--</code>, it wraps around to a huge positive number. Always use <code>int</code> for loop counters that may go negative.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">auto</code> = let compiler decide · <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int</code> = general integer · <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">size_t</code> = sizes/indices — <span style="color:#C0392B;font-weight:600">never use <code>size_t</code> in loops that count down to 0</span>.
</div>

你可以加一个 **Section 3: Best practice examples**，保持风格一致：

---

## <span style="color:#E8600A">3.</span> **Best practice examples**

### 1）Forward iteration

When iterating forward with `.size()`, use `size_t`.

```cpp
vector<int> v = {1,2,3};

for (size_t i = 0; i < v.size(); i++)
{
    cout << v[i] << " ";
}
```

**Why**

* `.size()` returns `size_t`
* avoids signed/unsigned comparison warnings

---

### 2）Reverse iteration

When the loop variable may become negative, use `int`.

```cpp
vector<int> v = {1,2,3};

for (int i = (int)v.size() - 1; i >= 0; i--)
{
    cout << v[i] << " ";
}
```

**Why**

* `size_t` cannot represent negative values
* using `int` allows the condition `i >= 0`

---

### 3）Range-based loop (Modern C++)

If you don't need the index, use a range-based loop.

```cpp
vector<int> v = {1,2,3};

// const auto& x
for (auto &x : v) // do not use auto x, it will do value copy
{
    cout << x << " ";
}
```

**Advantages**

* simpler syntax
* avoids index type issues
* recommended in modern C++

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 Practical rule</span><br>
Forward loops → <code>size_t</code><br>
Reverse loops → <code>int</code><br>
No index needed → <code>range-based for</code>
</div>
