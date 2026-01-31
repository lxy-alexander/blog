---
title: C++ Notes
published: 2026-01-20
description: "C++ Notes"
image: ""
tags: ["C++ Notes"]
category: c++
draft: false
---



## nth_element

```
nth_element(nums.begin(), nums.begin() + (k - 1), nums.end(), greater<int>());
```

nums[k-1] 就是第 k 大元素 ✅

对所有 i < k-1：nums[i] >= nums[k-1]

对所有 i > k-1：nums[i] <= nums[k-1]

⚠️ 但是：[0 ... k-2] 这部分 不保证有序，[k ... n-1] 这部分 也不保证有序。只是“分区”好了，不是排序好了。

### 1）最常用的两个

```
less<int>()
```

表示 **升序（小在前）**：`a < b`。等价于 `sort(nums.begin(), nums.end())`

```
greater<int>()
```

表示 **降序（大在前）**：`a > b`

------

### 2）自定义比较器（lambda）

你也可以自己写规则，比如降序：

```
auto cmp = [](int a, int b) {
    return a > b;
};
```

然后：

```
sort(nums.begin(), nums.end(), cmp);
```







## minmax_element

Return pair of iterators, `mn` is an iterator pointing to the smallest element, and `mx` is an iterator pointing to the largest element.

如果 nums 是 const 呢？mn` / `mx` 是 `std::vector<int>::const_iterator

```c++
std::pair<
	std::vector<int>::iterator,
	std::vector<int>::iterator
>
```

```c++
auto [mn, mx] = minmax_element(nums.begin(), nums.end()); 
```



## `#include <bits/stdc++.h>` 

**It is **the global header file provided by GCC/Clang compiler, it includes common standard header files.



## 一维数组初始化false/0

| 写法              | 每个元素占用 | 优点               | 缺点                           | 推荐度 |
| ----------------- | ------------ | ------------------ | ------------------------------ | ------ |
| `vector<bool>`    | 1 bit        | 最省内存           | 行为特殊，可能慢/坑            | ⭐⭐     |
| `vector<char>`    | 1 byte       | 快、稳定、最常用   | char 可能有符号（但做0/1没事） | ⭐⭐⭐⭐⭐  |
| `vector<uint8_t>` | 1 byte       | 快、稳定、语义明确 | 写起来稍微陌生                 | ⭐⭐⭐⭐   |

### `vector<bool> vis(n, false);`

```cpp
int n = 10;

// 初始化：长度 n，全是 false
vector<bool> vis(n, false);

// 标记：把下标 3 标成 true
vis[3] = true;

// 判断：下标 3 是否被标记
if (vis[3]) {
    // vis[3] == true
}
```

📌 用法很像普通 bool 数组，但它内部是 bit 压缩的。

------

### ==`vector<char> vis(n, 0);`==

```cpp
int n = 10;

// 初始化：长度 n，全是 0
vector<char> vis(n, 0);

// 标记：把下标 3 标成 1
vis[3] = 1;

// 判断：下标 3 是否被标记
if (vis[3]) {
    // vis[3] != 0
}
```

📌 这是刷题最稳、最常用的标记数组写法。

------

### `vector<uint8_t> vis(n, 0);`

```cpp
#include <cstdint>

int n = 10;

// 初始化：长度 n，全是 0
vector<uint8_t> vis(n, 0);

// 标记：把下标 3 标成 1
vis[3] = 1;

// 判断：下标 3 是否被标记
if (vis[3]) {
    // vis[3] != 0
}
```





## 二维数组初始化false/0

假设我们要一个 `n 行 m 列` 的二维标记数组 `vis`：

------

### 1）二维 `vector<vector<bool>>`（省内存，但行为特殊）

```cpp
int n = 3, m = 4;

// 初始化：n 行 m 列，全是 false
vector<vector<bool>> vis(n, vector<bool>(m, false));

// 标记 (1,2) 为 true
vis[1][2] = true;

// 判断 (1,2) 是否被标记
if (vis[1][2]) {
    // true
}
```

📌 注意：`vector<bool>` 行为特殊，二维也一样特殊。

------

### 2）二维 `vector<vector<char>>`

```cpp
int n = 3, m = 4;

// 初始化：n 行 m 列，全是 0
vector<vector<char>> vis(n, vector<char>(m, 0));

// 标记 (1,2) 为 1
vis[1][2] = 1;

// 判断 (1,2) 是否被标记
if (vis[1][2]) {
    // 非 0 就算 true
}
```

✅ 刷题最常用、稳定、快。

------

### 3）二维 `vector<vector<uint8_t>>`（byte 二维数组）

```cpp
#include <cstdint>
int n = 3, m = 4;

// 初始化：n 行 m 列，全是 0
vector<vector<uint8_t>> vis(n, vector<uint8_t>(m, 0));

// 标记 (1,2) 为 1
vis[1][2] = 1;

// 判断 (1,2) 是否被标记
if (vis[1][2]) {
    // 非 0 就算 true
}
```

------

### ==4）更快的二维写法：用一维数组模拟二维（性能最强✅）==

当 n*m 很大时，推荐这一种（内存连续更快）：

```cpp
int n = 3, m = 4;

// 初始化：一维长度 n*m，全是 0
vector<char> vis(n * m, 0);

// 计算二维坐标 (i,j) 的一维下标
auto id = [&](int i, int j) {
    return i * m + j;
};

// 标记 (1,2)
vis[id(1,2)] = 1;

// 判断
if (vis[id(1,2)]) {
    // true
}
```

------







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







