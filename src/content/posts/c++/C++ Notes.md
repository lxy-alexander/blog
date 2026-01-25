---
title: C++ Notes
published: 2026-01-20
description: "C++ Notes"
image: ""
tags: ["C++ Notes"]
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



## std::max

`std::max` needs both arguments to be the **same type**.  You’re passing `int` (`ans`) and `double` (`sum * 1.0 / k`).↳



## map for loop

**Use `auto` when the type is obvious from the initializer or when the exact type is not important.**  Containers + iterators = **perfect for `auto`**.

```c++
for (const auto& n : cnt)
    print_key_value(n.first, n.second);

for (const std::pair<const char, int>& n : cnt)
    print_key_value(n.first, n.second);

// or even better (C++17):
for (const auto& [ch, freq] : cnt)
    print_key_value(ch, freq);
```



## unordered_set

`unordered_set<char> vowels = {'a','e','i','o','u'};`

OR

`bool isVowel(char c) {return c=='a'||c=='e'||c=='i'||c=='o'||c=='u';}`







