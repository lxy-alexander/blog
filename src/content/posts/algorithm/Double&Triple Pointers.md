---
title: "Double&Triple Pointers"
published: 2026-01-29
description: "Double&Triple Pointers"
image: ""
tags: ["algorithm","Double&Triple Pointers"]
category: algorithm
draft: false
lang: ""
---



# Double Pointers

------

## Check if `p` is a subsequence of `s` 

Check whether `p` is a subsequence of `s`  after some characters of `s` have been removed.

### ✅ 最优雅（推荐）——只写一个 if

```python
j = 0
for i in range(m):
    if not removed[i] and j < len(p) and s[i] == p[j]:
        j += 1
return j == len(p)
```

```cpp
int j = 0;
for (int i = 0; i < m; i++) {
    if (!removed[i] && j < (int)p.size() && s[i] == p[j]) {
        j++;
    }
}
return j == (int)p.size();
```

------



### ✅ 更“清爽”的 continue 版（逻辑最直观）

```python
j = 0
for i in range(m):
    if removed[i]:
        continue
    if j < len(p) and s[i] == p[j]:
        j += 1
return j == len(p)
```

`(int)` 是为了避免 `int` 和 `size_t(无符号)` 混着比较导致警告/潜在bug。在 C++ 里如果你把 `j` 定义成 `size_t`（无符号），然后==写了 `j--`，会出现一个非常坑的现象：不会变成 -1，而是“下溢”变成一个超级大的数。==

```cpp
int j = 0;
for (int i = 0; i < m; i++) {
    if (removed[i]) continue;
    if (j < (int)p.size() && s[i] == p[j]) {
        j++;
    }
}
return j == (int)p.size();
```

------

### ✅ 用 for ch in s 更优雅（但要处理 removed）

```python
j = 0
for ch in s:
    if j < len(p) and ch == p[j]:
        j += 1
return j == len(p)
```

### 

```cpp
int j = 0;
for (char ch : s) {
    if (j < (int)p.size() && ch == p[j]) {
        j++;
    }
}
return j == (int)p.size();
```

------

