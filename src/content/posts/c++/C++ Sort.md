---
title: "C++ sort"
published: 2026-01-31
description: "C++ sort"
image: ""
tags: ["c++","C++ sort"]
category: c++
draft: false
lang: ""
---


# **I. C++ Sorting Patterns with `std::sort`**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">

The <span style="color:#E8600A;font-weight:700">Sorting (排序)</span> operation is one of the most common tasks in algorithm design.
In C++, the function <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::sort()</code> from the <span style="color:#E8600A;font-weight:700">Standard Template Library (标准模板库, STL)</span> provides an efficient way to reorder containers.

Understanding <span style="color:#E8600A;font-weight:700">Comparator Functions (比较器函数)</span>, multi-key sorting, and container transformations allows flexible sorting of vectors, pairs, strings, and maps.

</div>

---

## 1. **Basic Vector Sorting**

### 1) Sort a Vector In-Place (原地排序)

The function <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">sort(begin, end)</code> sorts elements directly inside the container.

<span style="color:#2980B9">Therefore</span>, the original vector will be modified.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> a = {3, 1, 2};
    sort(a.begin(), a.end());
    // a = [1,2,3]
}
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
The default ordering uses the <span style="color:#E8600A;font-weight:700">Less Comparator (小于比较器)</span>, meaning the result is <span style="color:#E8600A;font-weight:700">Ascending Order (升序)</span>.
</div>

---

### 2) Return a New Sorted Vector (不修改原数据)

Sometimes the original container must remain unchanged.

<span style="color:#2980B9">In this case</span>, create a <span style="color:#E8600A;font-weight:700">Copy (副本)</span> before sorting.

```cpp
vector<int> a = {3, 1, 2};
vector<int> b = a;             // copy
sort(b.begin(), b.end());      // sort copy
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
This pattern is common in <span style="color:#E8600A;font-weight:700">Functional Programming Style (函数式编程风格)</span> where original data should remain immutable.
</div>

---

## 2. **Descending Order Sorting**

### 1) Method A — `greater<>`

The template comparator <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">greater<T></code> reverses the sorting order.

```cpp
sort(a.begin(), a.end(), greater<int>());
```

This produces a <span style="color:#E8600A;font-weight:700">Descending Order (降序)</span> result.

---

### 2) Method B — Custom Comparator (自定义比较器)

A <span style="color:#E8600A;font-weight:700">Lambda Expression (Lambda表达式)</span> can define custom logic.

```cpp
sort(a.begin(), a.end(), [](int x, int y){
    return x > y;
});
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
The comparator must satisfy the rule of <span style="color:#E8600A;font-weight:700">Strict Weak Ordering (严格弱序)</span>, otherwise the behavior of <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">std::sort</code> becomes undefined.
</div>

---

## 3. **Sorting Pairs / Tuples by Key**

### 1) Sort by First Key then Second Key

Containers like <span style="color:#E8600A;font-weight:700">Pair (键值对)</span>
or <span style="color:#E8600A;font-weight:700">Tuple (元组)</span>
often require multi-field sorting.

Example: sort by first element descending, then second descending.

```cpp
void sortByKey() {
    std::vector<std::pair<int, int>> arr = {{1, 5}, {2, 5}, {2, 3}, {3, 4}};
    std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
        return a.first != b.first ? a.first > b.first : a.second > b.second;
    });

    for (auto &p: arr){
        printf("(%d, %d)", p.first, p.second);
    }
    printf("\n");
}
```

<span style="color:#2980B9">Logic</span>:

1）Compare primary key
2）If equal → compare secondary key

---

## 4. **Multi-Key Sorting Patterns**

Multiple fields may require different ordering directions.

| Pattern     | First Key  | Second Key | Example Result         |
| ----------- | ---------- | ---------- | ---------------------- |
| Asc + Asc   | ascending  | ascending  | `{1,5}{2,3}{2,5}{3,4}` |
| Asc + Desc  | ascending  | descending | `{1,5}{2,5}{2,3}{3,4}` |
| Desc + Asc  | descending | ascending  | `{3,4}{2,3}{2,5}{1,5}` |
| Desc + Desc | descending | descending | `{3,4}{2,5}{2,3}{1,5}` |

---

### 1) Ascending + Ascending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first < b.first : a.second < b.second;
});
```

---

### 2) Ascending + Descending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second;
});
```

---

### 3) Descending + Ascending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
});
```

---

### 4) Descending + Descending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first > b.first : a.second > b.second;
});
```

---

## 5. **Sorting Strings by Length**

### 1) Length Descending + Lexicographic Ascending

This example combines

* <span style="color:#E8600A;font-weight:700">Length Comparison (长度比较)</span>
* <span style="color:#E8600A;font-weight:700">Lexicographic Order (字典序)</span>

```cpp
void sortByStringLength() {

    std::vector<std::string> words =  {"apple", "bat", "banana", "app"};

    std::sort(words.begin(), words.end(),
    [](const std::string &a, const std::string &b){

        return a.size() != b.size()
            ? a.size() > b.size()
            : a < b;
    });

    for (auto &p : words) {
        printf("%s ", p.c_str());
    }

    printf("\n");
}
```

Result

```
banana apple app bat
```

---

## 6. **Sorting Map / Counter by Value**

Maps cannot be sorted directly because they are <span style="color:#E8600A;font-weight:700">Associative Containers (关联容器)</span>.

<span style="color:#2980B9">Therefore</span> we first convert them into a vector.

---

### 1) Python Example

```python
sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
```

Meaning:

* value descending
* key ascending

---

### 2) C++ Implementation

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {

    vector<int> nums = {1,1,1,2,2,3,3,4};

    unordered_map<int,int> cnt;

    for (int x : nums) {
        cnt[x]++;
    }

    vector<pair<int,int>> items(cnt.begin(), cnt.end());

    sort(items.begin(), items.end(),
    [](auto &a, auto &b){

        if (a.second != b.second)
            return a.second > b.second; // freq desc

        return a.first < b.first; // key asc
    });

}
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
This pattern is frequently used in <span style="color:#E8600A;font-weight:700">Frequency Analysis (频率统计)</span> problems such as <span style="color:#E8600A;font-weight:700">Top-K Elements (前K高频元素)</span>.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
The key to mastering <span style="color:#E8600A;font-weight:700">Sorting (排序)</span> in C++ is understanding how <span style="color:#E8600A;font-weight:700">Comparators (比较器)</span> control ordering, allowing flexible multi-key and custom sorting using <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">std::sort</code>.
</div>


