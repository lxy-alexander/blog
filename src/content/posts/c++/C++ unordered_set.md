---
title: "C++ unordered_set"
published: 2026-03-08
description: "C++ unordered_set"
image: ""
tags: ["c++","C++ unordered_set"]
category: c++
draft: false
lang: ""
---


# **I. `unordered_set` (无序集合)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">unordered_set</code> stores unique elements with <span style="color:#E8600A;font-weight:700">O(1) average lookup (平均 O(1) 查找)</span> using a Hash Table (哈希表). It is ideal for fast membership checks — e.g. checking if a character is a vowel.
</div>

---

## <span style="color:#E8600A">1.</span> **Two approaches for membership check (两种成员检查方式)**

### 1) `unordered_set` — clean and scalable

```cpp
unordered_set<char> vowels = {'a', 'e', 'i', 'o', 'u'};

if (vowels.count(c)) { /* c is a vowel */ }
// or
if (vowels.find(c) != vowels.end()) { /* c is a vowel */ }
```

### 2) Manual boolean function — fast for tiny fixed sets

```cpp
bool isVowel(char c) {
    return c=='a' || c=='e' || c=='i' || c=='o' || c=='u';
}
```

---

## <span style="color:#E8600A">2.</span> **When to choose which (如何选择)**

| Approach | Best for |
|---|---|
| `unordered_set` | Dynamic sets, large sets, or when elements may change |
| Manual function | Tiny fixed sets (like 5 vowels), maximum speed |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>unordered_set</code> uses <span style="color:#2980B9">hashing (哈希)</span> — it has no guaranteed order. If you need sorted iteration, use <code>std::set</code> instead (O(log n) lookup).
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">unordered_set</code> = O(1) lookup via hashing, <span style="color:#C0392B;font-weight:600">no order</span>; use <code>set</code> if you need sorted order.
</div>