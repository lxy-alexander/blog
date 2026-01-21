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



## Container

### vector

```
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {2, 3, 1, 4};
    v.push_back(5);
    v.push_back(6);
    v[2] = -1;
    for (int x : v) {
        std::cout << x << ' '; // 2 3 -1 4 5 6 
    }
    std::cout << '\n';
}
```



### array

C++ array with length `n`  

int a[n];  Not allowed (standard C++) . 

vector<int> a(n); 👉 Use vector when the size is decided at runtime.

- **Runtime size → `vector`**
- *Compile-time size → `int a[5]` / `std::array`**
- **Fixed size → array**
- *Variable size → vector**









