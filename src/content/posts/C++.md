---
title: C++
published: 2026-01-20
description: "C++"
image: ""
tags: ["Blogging", "C++"]
category: Guides
draft: false
---

# C++

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









