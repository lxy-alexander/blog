---
title: C++ Interview
published: 2025-01-15
description: "C++ Interview"
image: "./cover.jpeg"
tags: ["Blogging", "C++ Interview"]
category: Guides
draft: false
---

# C++ Interview

## Compile Method

- **If you are doing low-level development on Linux:<u>** **g++** is the preferred choice</u>, as its compatibility and stability have been proven over decades.↳
- **If you are developing on Mac:** **clang++** is the default option (even if you type `g++`, <u>the command on macOS is typically just a wrapper for `clang`</u>).↳
- **If you care about the development experience (error messages):** Use **clang++** (paired with **clangd**) during the development phase; it helps you locate syntax errors much faster.



## Printf & Cout

- Use **`std::cout`** by default in modern C++ because it is type-safe and integrates naturally with C++ objects and templates.
- Use **`printf`** only when working with C APIs, low-level/system code, or when you need strict formatting control and predictable performance.



## Why use **reference** in function parameters (C++)?

- To avoid copying (performance) ⚡
- requires in-place modification
- **Modify → `&`**.     **Read only → `const &`**.    **Small copy → pass by value**

```
void f(vector<int> v);    // copies the whole vector ❌

void f(vector<int>& v);  // no copy ✅
```



## auto/int/size_t

- **`auto`** → let compiler deduce
- **`int`** → general integer
- **`size_t`** → size / index



## std

`std` is the **namespace** that contains **all symbols defined by the C++ standard library**.



## Integer division

**Fix:**

```
return (double)maxSum / k;
// or
return maxSum * 1.0 / k;
```

`int / int` truncates decimals, so you got **12.00000** instead of **12.75000**.





## Std::max

`std::max` needs both arguments to be the **same type**.  You’re passing `int` (`ans`) and `double` (`sum * 1.0 / k`).↳



