---
title: "Lambda Expression(anonymous function)"
published: 2026-01-29
description: "Lambda Expression(anonymous function)"
image: ""
tags: ["c++","Lambda Expression(anonymous function)"]
category: c++
draft: false
lang: ""
---



# **C++11 lambda expression** 

==如果不用 `auto`，lambda 原本的类型是一个 **编译器生成的匿名类类型（unnamed closure type）**。==

------

## 1) Basic Syntax

```cpp
[capture](parameters) { body }
```

-   capture（捕获）
-   parameters（参数）
-   body（函数体）

------

## 2) Capture List `[]`（捕获列表）

Control lambda how to use external variables (how external variables come in)

-   `[]`：capture nothing（不捕获）
-   `[=]`：capture by value（按值捕获 / 拷贝）
-   `[&]`：capture by reference（按引用捕获）
-   `[x]`：capture x by value（x 按值捕获）
-   `[&x]`：capture x by reference（x 按引用捕获）

------

## 3) Return Type（返回类型）

（auto-deduced（自动推导）

也可以显式写：

```cpp
[](int a, int b) -> long long { return 1LL*a+b; }
```

-   return type（返回类型）
-   explicit（显式）

------

## 4) `mutable`（允许修改值捕获变量）

默认值捕获变量不可改（read-only）

```cpp
int x = 1;
auto f = [x]() mutable { x++; };
```

------

## 5) Immediately Invoked Lambda（立即执行 lambda）

写完立刻调用：

```cpp
int ans = [&]() { return 42; }();
```

------

## 6) Lambda as Functor（函数对象）

lambda 本质是一个对象（object），可以像函数一样调用：

```cpp
auto f = [](int x){ return x+1; };
f(10);
```

------

## 7) Use with STL Algorithms（配合 STL 算法）

最常用场景，比如排序：

```cpp
sort(v.begin(), v.end(), [](int a,int b){ return a>b; });
```

常见算法：

-   `sort`（排序）
-   `find_if`（按条件查找）
-   `count_if`（按条件计数）

------

## 8) Capture `this`（捕获 this 指针）

类成员函数里常用：

```cpp
auto f = [this]() { return value; };
```

------

## 9) Common Pitfall（常见坑）：Dangling Reference（悬空引用）

引用捕获时，变量生命周期结束会炸：

-   dangling reference（悬空引用）
-   lifetime（生命周期）

------







If you want, I can also give you a **quick cheat sheet** (10 lines) for interviews / competitive programming.

This is a **C++11 lambda expression** (an anonymous function).

```cpp
auto id = [&](int i, int j) {
    return i * m + j;
};
```

### 1) `auto id = ...`

This defines a variable named `id`.
Its type is automatically deduced by the compiler (because the type of a lambda is complicated and unnamed).

### 2) `[&](int i, int j) { ... }`

This part is the lambda function:

-   `[](parameters) { body }` is the basic lambda syntax
-   `[&]` means **capture external variables by reference**
    -   so the lambda can directly use `m`, which is defined outside
    -   capturing by reference means if `m` changes outside, the lambda sees the updated value

### 3) Purpose: convert 2D coordinates into a 1D index

It maps a 2D coordinate `(i, j)` into a 1D array index:

```cpp
i * m + j
```

Here `m` usually represents the number of columns.

For example, if a `n × m` matrix is stored in a 1D array:

-   element at row `i`, column `j` → index = `i * m + j`

------

✅ Equivalent code (without lambda):

```cpp
int id(int i, int j) {
    return i * m + j;
}
```

But this version requires `m` to be a global variable or a class member, while the lambda is more flexible.

If you want, I can also explain the difference between `[=]`, `[&]`, `[this]`, etc.
