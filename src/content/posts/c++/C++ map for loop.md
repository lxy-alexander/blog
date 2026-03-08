---
title: "C++ map for loop"
published: 2026-03-08
description: "C++ map for loop"
image: ""
tags: ["c++","C++ map for loop"]
category: c++
draft: false
lang: ""
---


# **I. Map For Loop (`map` 遍历方式)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
When iterating over <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">std::map</code> or <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">std::unordered_map</code>, each element is a <span style="color:#E8600A;font-weight:700">Key-Value Pair (键值对)</span> of type <code>std::pair&lt;const Key, Value&gt;</code>. Use <code>auto</code> or C++17 <span style="color:#E8600A;font-weight:700">Structured Binding (结构化绑定)</span> for clean, readable code.
</div>

---

## <span style="color:#E8600A">1.</span> **Three iteration styles (三种遍历写法)**

### 1) `auto` with `.first` / `.second`

```cpp
for (const auto& n : cnt)
    print_key_value(n.first, n.second);
```

### 2) Explicit pair type (显式类型)

```cpp
for (const std::pair<const char, int>& n : cnt)
    print_key_value(n.first, n.second);
```

### 3) C++17 Structured Binding (结构化绑定) — recommended ✅

```cpp
for (const auto& [ch, freq] : cnt)
    print_key_value(ch, freq);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Always use <code>const auto&</code> (not <code>auto</code>) to avoid copying each pair — map elements can be large. Structured binding requires <span style="color:#C0392B;font-weight:600">C++17</span>.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">for (const auto& [key, val] : map)</code> (C++17) for the cleanest map iteration — <code>auto</code> + containers = perfect match.
</div>