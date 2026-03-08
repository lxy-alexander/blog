---
title: "C++ numeric_limits"
published: 2026-03-07
description: "C++ numeric_limits"
image: ""
tags: ["c++","C++ numeric_limits"]
category: c++
draft: false
lang: ""
---

# **I. Min & Max in C++ (最值查询完全指南)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
C++ offers three layers for finding min/max: <span style="color:#E8600A;font-weight:700">Sentinel Values (哨兵值)</span> with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">INT_MAX</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">INT_MIN</code>, <span style="color:#E8600A;font-weight:700">STL Algorithms (STL 算法)</span> with <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">min_element</code> / <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">max_element</code>, and the type-safe <span style="color:#E8600A;font-weight:700">numeric_limits (数值极限模板)</span>. A critical pitfall: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">numeric_limits&lt;float&gt;::min()</code> is <span style="color:#C0392B;font-weight:600">NOT</span> the most negative float — use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">lowest()</code> instead.
</div>

---

## <span style="color:#E8600A">1.</span> **`numeric_limits` — Integer vs Float (整数 vs 浮点)**

### 1) Integer Types (整数类型)

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">min()</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">max()</code> behave <span style="color:#2980B9">intuitively (符合直觉)</span> for integers:

```cpp
std::numeric_limits<int>::min()   // most negative → same as INT_MIN
std::numeric_limits<int>::max()   // most positive → same as INT_MAX
```

### 2) Floating-point Types (浮点类型) ⚠️

```cpp
std::numeric_limits<float>::min()   // ⚠️ smallest POSITIVE normalized value ≈ 1.175e-38
```

<span style="color:#C0392B;font-weight:600">This is NOT the most negative float.</span> For the most negative value, always use:

```cpp
std::numeric_limits<float>::lowest()   // ✅ most negative float ≈ -3.4e+38
std::numeric_limits<double>::lowest()  // ✅ most negative double
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Per the IEEE 754 (浮点标准) spec, <code>min()</code> was defined as the smallest <em>normalized</em> positive value — designed to measure precision, not the negative boundary. This is the <span style="color:#C0392B;font-weight:600">opposite</span> of integer behavior and a very common bug.
</div>

### 3) Quick Reference Table (速查表)

| Type | C Max | C Most Negative | C++ Max | C++ Most Negative |
|---|---|---|---|---|
| `int` | `INT_MAX` | `INT_MIN` | `numeric_limits<int>::max()` | `numeric_limits<int>::min()` |
| `float` | `FLT_MAX` | `-FLT_MAX` | `numeric_limits<float>::max()` | `numeric_limits<float>::lowest()` |
| `double` | `DBL_MAX` | `-DBL_MAX` | `numeric_limits<double>::max()` | `numeric_limits<double>::lowest()` |

---

## <span style="color:#E8600A">2.</span> **Why `::max()` Uses `::` (为什么用双冒号)**

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">max()</code> is a <span style="color:#E8600A;font-weight:700">Static Member Function (静态成员函数)</span> of the class template <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">std::numeric_limits&lt;T&gt;</code>. It belongs to the **class**, not to any object instance.

| Call style | When to use |
|---|---|
| `ClassName::func()` | <span style="color:#E8600A;font-weight:700">Static member function (静态成员函数)</span> — belongs to the class |
| `obj.func()` | Non-static member function — belongs to an object instance (对象实例) |

So <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::numeric_limits&lt;int&gt;::max()</code> uses the <span style="color:#E8600A;font-weight:700">Scope Resolution Operator (作用域解析运算符)</span> `::` to access a static function inside the <span style="color:#2980B9">Namespace (命名空间)</span> `std` and class `numeric_limits<int>`.

---

## <span style="color:#E8600A">3.</span> **STL Algorithms for Min/Max (STL 最值算法)**

### 1) On `vector` (动态数组)

```cpp
vector<int> a = {1, 2, 3, 4};

int mn = *min_element(a.begin(), a.end());
int mx = *max_element(a.begin(), a.end());

// C++17: get both in one call
auto [mn_it, mx_it] = minmax_element(a.begin(), a.end());
printf("min:%d\n", *mn_it);
printf("max:%d\n", *mx_it);
```

### 2) On raw array (原始数组)

```cpp
int a[] = {1, 2, 3, 4};
int n = sizeof(a) / sizeof(a[0]);

int mn = *min_element(a, a + n);   // left-closed, right-open (左闭右开)
int mx = *max_element(a, a + n);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span><code>min_element</code> / <code>max_element</code> return <span style="color:#E8600A;font-weight:700">Iterators (迭代器)</span>, not values directly — always dereference with <code>*</code>.
</div>

### 3) Manual scan with sentinel (手动遍历 + 哨兵值)

```cpp
int mn = INT_MAX;   // 2³¹ − 1
int mx = INT_MIN;   // −2³¹
int a[] = {1, 2, 3, 4};

for (int x : a) {
    mn = min(mn, x);
    mx = max(mx, x);
}
```

---

## <span style="color:#E8600A">4.</span> **C++17: Structured Binding (结构化绑定)**

<code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">minmax_element</code> returns a <span style="color:#E8600A;font-weight:700">Pair of Iterators (迭代器对)</span>. C++17's <span style="color:#E8600A;font-weight:700">Structured Binding (结构化绑定)</span> lets you unpack it cleanly:

```cpp
// returns std::pair<vector<int>::iterator, vector<int>::iterator>
auto [mn, mx] = minmax_element(nums.begin(), nums.end());
// mn → iterator to smallest element
// mx → iterator to largest element
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Structured binding <code>auto [a, b] = ...</code> is a <span style="color:#C0392B;font-weight:600">C++17 feature (C++17 特性)</span>. Make sure to compile with <code>-std=c++17</code> or later.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
For integers <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">min()</code> = most negative; for floats use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">lowest()</code> instead. Use <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">minmax_element</code> + <span style="color:#E8600A;font-weight:700">Structured Binding (结构化绑定)</span> for clean STL-style min/max on containers.
</div>