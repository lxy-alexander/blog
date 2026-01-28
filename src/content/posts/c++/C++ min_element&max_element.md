---
title: "C++ min_element&max_element"
published: 2026-01-27
description: "C++ min_element&max_element"
image: ""
tags: ["c++","C++ min_element&max_element"]
category: c++
draft: false
lang: ""
---

### 

## **Integer types**

```cpp
std::numeric_limits<int>::min()
```

returns the <u>**most negative value**</u> (same as `INT_MIN`).



##  **Floating-point types**

```cpp
std::numeric_limits<float>::min()
```

It does **NOT** mean the most negative value
It means the **smallest positive normalized value**

That’s why for floating-point types you should always use:

```cpp
lowest()
```

------



##  One-line takeaway

**For integers, `min()` gives the smallest value; for floating-point numbers, `lowest()` gives the most negative value.**

| Type   | C-style Max | C-style Min (negative) | C++-style Max                        | C++-style Min (negative)                |
| ------ | ----------- | ---------------------- | ------------------------------------ | --------------------------------------- |
| int    | `INT_MAX`   | `INT_MIN`              | `std::numeric_limits<int>::max()`    | `std::numeric_limits<int>::min()`       |
| float  | `FLT_MAX`   | `-FLT_MAX`             | `std::numeric_limits<float>::max()`  | `std::numeric_limits<float>::lowest()`  |
| double | `DBL_MAX`   | `-DBL_MAX`             | `std::numeric_limits<double>::max()` | `std::numeric_limits<double>::lowest()` |





## Why does `::max()` have `::`?

Static member functions (静态成员函数) belong to the class, so you call them with `ClassName::func()`. Non-static member functions belong to objects, so you call them with `obj.func()`.

-   **static member function（静态成员函数）**
-   **scope resolution operator `::`（作用域解析运算符）**
-   **namespace（命名空间）**
-   **type（类型）**
-   **object（对象）**

Because `max()` is a **static member function** inside the class template `std::numeric_limits<T>`. So you must access it using the **scope resolution operator** `::`





```c++
#include <algorithm>
#include <climits>
#include <cstdio>
#include <vector>

using namespace std;

// decomposition declarations = structured binding
// C++17 extension = 这是 C++17 才支持 的特性

void findMinMaxInContainer() {
    vector<int> a = {1, 2, 3, 4};
    int n = a.size();
    int mn = *min_element(a.begin(), a.end());
    int mx = *max_element(a.begin(), a.end());
    auto [mn_it, mx_it] = minmax_element(a.begin(), a.end());
    printf("min:%d\n", *mn_it);
    printf("max:%d\n", *mx_it);
}

void findMinMaxInArray() {
    int a[] = {1, 2, 3,4};
    int n = sizeof(a) / sizeof(a[0]);
    int mn = *min_element(a, a + n); // left-closed, right-open
    printf("min:%d\n", mn); // cout << mn << endl;
    int mx = *max_element(a, a + n);
    printf("max:%d\n", mx);
}

void findMinMaxbyNormalWayInIntArray() {
    int mn = INT_MAX; // 2³¹ − 1
    int mx = INT_MIN; // - 2³¹
    int a[] =  {1, 2, 3,4};
    for (int x : a) {
        mn = max(mn, x);
        mx = min(mn, x);
    }
    printf("min:%d\n", mn); // cout << mn << endl;
    printf("max:%d\n", mx);
}

int main() {
    findMinMaxbyNormalWayInIntArray();
    findMinMaxInArray();
    findMinMaxInContainer();
}
```

