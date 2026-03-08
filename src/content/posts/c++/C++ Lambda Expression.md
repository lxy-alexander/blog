---
title: "C++ Lambda Expression"
published: 2026-03-07
description: "C++ Lambda Expression"
image: ""
tags: ["c++","C++ Lambda Expression"]
category: c++
draft: false
lang: ""
---

# **I. C++11 Lambda Expression (Lambda 表达式)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
A <span style="color:#E8600A;font-weight:700">Lambda Expression (Lambda 表达式)</span> is an anonymous inline function introduced in C++11. Without <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">auto</code>, its true type is a compiler-generated <span style="color:#E8600A;font-weight:700">Unnamed Closure Type (匿名闭包类型)</span> — unique, hidden, and impossible to write by hand. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">auto</code> to let the compiler deduce it.
</div>

---

## <span style="color:#E8600A">1.</span> **Basic Syntax (基本语法)**

```cpp
[capture](parameters) { body }
```

- <span style="color:#E8600A;font-weight:700">Capture (捕获)</span> — controls which external variables enter the lambda
- <span style="color:#E8600A;font-weight:700">Parameters (参数)</span> — function arguments, same as a regular function
- <span style="color:#E8600A;font-weight:700">Body (函数体)</span> — executable statements

---

## <span style="color:#E8600A">2.</span> **Capture List (捕获列表)** `[]`

Controls how the lambda accesses variables from the surrounding <span style="color:#2980B9">Scope (作用域)</span>:

| Syntax | Meaning |
|---|---|
| `[]` | Capture nothing (不捕获) |
| `[=]` | Capture all by value (按值捕获 / 拷贝) |
| `[&]` | Capture all by reference (按引用捕获) |
| `[x]` | Capture `x` by value (x 按值捕获) |
| `[&x]` | Capture `x` by reference (x 按引用捕获) |

---

## <span style="color:#E8600A">3.</span> **Return Type (返回类型)**

Return type is <span style="color:#2980B9">auto-deduced (自动推导)</span> by default. Can also be written explicitly (显式):

```cpp
[](int a, int b) -> long long { return 1LL * a + b; }
```

---

## <span style="color:#E8600A">4.</span> **`mutable` (允许修改值捕获变量)**

By default, variables captured by value are <span style="color:#C0392B;font-weight:600">read-only (只读)</span> inside the lambda. Add <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">mutable</code> to allow modification:

```cpp
int x = 1;
auto f = [x]() mutable { x++; };  // modifies the local copy, not the original
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">mutable</code> only modifies the <strong>local copy</strong> inside the lambda. The original variable outside is <span style="color:#C0392B;font-weight:600">not affected</span>.
</div>

---

## <span style="color:#E8600A">5.</span> **Immediately Invoked Lambda (立即执行 Lambda)**

Define and call in one expression:

```cpp
int ans = [&]() { return 42; }();
```

---

## <span style="color:#E8600A">6.</span> **Lambda as Functor (函数对象)**

A lambda is essentially an <span style="color:#E8600A;font-weight:700">Object (对象)</span> that can be called like a function:

```cpp
auto f = [](int x) { return x + 1; };
f(10);  // returns 11
```

---

## <span style="color:#E8600A">7.</span> **Use with STL Algorithms (配合 STL 算法)**

The most common use case — e.g. custom sort (排序):

```cpp
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
```

Common algorithms (常用算法):

- <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">sort</code> — sorting (排序)
- <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">find_if</code> — find by condition (按条件查找)
- <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">count_if</code> — count by condition (按条件计数)

---

## <span style="color:#E8600A">8.</span> **Capture `this` (捕获 this 指针)**

Used inside a class member function to access class members:

```cpp
auto f = [this]() { return value; };
```

---

## <span style="color:#E8600A">9.</span> **Common Pitfall (常见坑): Dangling Reference (悬空引用)**

When capturing by reference, if the original variable's <span style="color:#E8600A;font-weight:700">Lifetime (生命周期)</span> ends before the lambda is called, the reference becomes a <span style="color:#C0392B;font-weight:600">Dangling Reference (悬空引用)</span> — undefined behavior (未定义行为).

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Never return a lambda that captures a local variable <span style="color:#C0392B;font-weight:600">by reference</span> — the local variable will be destroyed when the function returns, leaving the reference dangling.
</div>

---

## <span style="color:#E8600A">10.</span> **Real-world Example: 2D → 1D Index (坐标压缩)**

A common competitive programming pattern — convert a 2D coordinate into a 1D array index:

```cpp
auto id = [&](int i, int j) {
    return i * m + j;
};
```

### 1) `auto id = ...`

The variable `id` holds the lambda. Its type is <span style="color:#2980B9">auto-deduced (自动推导)</span> since the true Closure Type (闭包类型) is unnamed.

### 2) `[&](int i, int j) { ... }`

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[&]</code> captures all external variables by reference — so `m` (number of columns, 列数) is accessible directly.

### 3) `i * m + j`

Maps a 2D coordinate `(i, j)` to a 1D index — for an `n × m` matrix stored in a 1D array, element at row `i`, column `j` → index `= i * m + j`.

```cpp
// Equivalent without lambda (requires m to be global):
int id(int i, int j) {
    return i * m + j;
}
```

The lambda version is <span style="color:#2980B9">more flexible</span> — `m` can be a local variable.

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
A lambda is a compiler-generated <span style="color:#E8600A;font-weight:700">Closure Type (闭包类型)</span>; use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">auto</code> to hold it, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[&]</code> to capture by reference, and always watch out for <span style="color:#C0392B;font-weight:600">Dangling References (悬空引用)</span>.
</div>