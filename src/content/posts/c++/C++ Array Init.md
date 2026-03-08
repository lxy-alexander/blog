---
title: "C++ Array Init"
published: 2026-03-07
description: "C++ Array Init"
image: ""
tags: ["c++","C++ Array Init"]
category: c++
draft: false
lang: ""
---

# **I. 1D Boolean Array Init (一维标记数组初始化)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Three common types for a 1D boolean/flag array: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vector&lt;bool&gt;</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vector&lt;char&gt;</code>, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vector&lt;uint8_t&gt;</code>. For competitive programming, <span style="color:#E8600A;font-weight:700"><code>vector&lt;char&gt;</code> is the recommended default</span> — fast, stable, and avoids the pitfalls of <code>vector&lt;bool&gt;</code>.
</div>

---

## <span style="color:#E8600A">1.</span> **Comparison Table (对比表)**

| Type | Memory per element | Speed | Pitfalls | Recommended |
|---|---|---|---|---|
| `vector<bool>` | 1 bit | May be slow | <span style="color:#C0392B;font-weight:600">Special behavior, avoid</span> | ⭐⭐ |
| `vector<char>` | 1 byte | ✅ Fast & stable | Minor: char may be signed | ⭐⭐⭐⭐⭐ |
| `vector<uint8_t>` | 1 byte | ✅ Fast & stable | Slightly unfamiliar syntax | ⭐⭐⭐⭐ |

---

## <span style="color:#E8600A">2.</span> **`vector<bool>` (bit-compressed, 位压缩)**

```cpp
int n = 10;
vector<bool> vis(n, false);

vis[3] = true;
if (vis[3]) { /* vis[3] == true */ }
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>vector&lt;bool&gt;</code> is internally <span style="color:#C0392B;font-weight:600">bit-compressed (位压缩)</span> — <code>vis[i]</code> returns a proxy object, not a real <code>bool&amp;</code>. This causes subtle bugs with <code>auto&</code>, <code>std::ref</code>, etc.
</div>

---

## <span style="color:#E8600A">3.</span> **`vector<char>` (recommended, 推荐)**

```cpp
int n = 10;
vector<char> vis(n, 0);

vis[3] = 1;
if (vis[3]) { /* vis[3] != 0 */ }
```

✅ Most stable and common pattern for competitive programming.

---

## <span style="color:#E8600A">4.</span> **`vector<uint8_t>` (explicit semantics, 语义明确)**

```cpp
#include <cstdint>
int n = 10;
vector<uint8_t> vis(n, 0);

vis[3] = 1;
if (vis[3]) { /* vis[3] != 0 */ }
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vector&lt;char&gt; vis(n, 0)</code> as your default flag array — <span style="color:#C0392B;font-weight:600">avoid <code>vector&lt;bool&gt;</code></span> due to its proxy-object behavior.
</div>

---

# **II. 2D Boolean Array Init (二维标记数组初始化)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
For an <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">n × m</code> boolean grid, you can use nested vectors or a <span style="color:#E8600A;font-weight:700">1D array simulating 2D (一维模拟二维)</span>. The 1D approach offers <span style="color:#E8600A;font-weight:700">better cache performance (缓存友好)</span> due to contiguous memory layout.
</div>

---

## <span style="color:#E8600A">1.</span> **`vector<vector<bool>>`**

```cpp
int n = 3, m = 4;
vector<vector<bool>> vis(n, vector<bool>(m, false));

vis[1][2] = true;
if (vis[1][2]) { /* true */ }
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>vector&lt;bool&gt;</code> has <span style="color:#C0392B;font-weight:600">special proxy-object behavior (代理对象行为)</span> even in 2D — avoid when possible.
</div>

---

## <span style="color:#E8600A">2.</span> **`vector<vector<char>>` (recommended, 推荐)**

```cpp
int n = 3, m = 4;
vector<vector<char>> vis(n, vector<char>(m, 0));

vis[1][2] = 1;
if (vis[1][2]) { /* non-zero = true */ }
```

✅ Most stable and common for competitive programming.

---

## <span style="color:#E8600A">3.</span> **`vector<vector<uint8_t>>`**

```cpp
#include <cstdint>
int n = 3, m = 4;
vector<vector<uint8_t>> vis(n, vector<uint8_t>(m, 0));

vis[1][2] = 1;
if (vis[1][2]) { /* non-zero = true */ }
```

---

## <span style="color:#E8600A">4.</span> **1D array simulating 2D**

When `n * m` is large, use a flat 1D array for <span style="color:#E8600A;font-weight:700">contiguous memory (连续内存)</span> and better cache performance:

```cpp
int n = 3, m = 4;
vector<char> vis(n * m, 0);

// index mapping: (i, j) → i * m + j
auto id = [&](int i, int j) {
    return i * m + j;
};

vis[id(1, 2)] = 1;
if (vis[id(1, 2)]) { /* true */ }
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>The <span style="color:#2980B9">Index Mapping Formula (坐标映射公式)</span> is <code>i * m + j</code> where <code>m</code> = number of columns (列数). The lambda captures <code>m</code> by reference via <code>[&]</code>.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
For large grids, prefer <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">vector&lt;char&gt; vis(n * m, 0)</code> with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">id(i,j) = i*m+j</code> — <span style="color:#E8600A;font-weight:700">contiguous memory (连续内存)</span> is faster than nested vectors.
</div>