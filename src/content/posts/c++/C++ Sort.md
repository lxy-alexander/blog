---
title: "C++ Sort"
published: 2026-01-31
description: "C++ Sort"
image: ""
tags: ["c++","C++ Sort"]
category: c++
draft: false
lang: ""
---

## 1) Sort a Vector (In-Place)

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> a = {3, 1, 2};
    sort(a.begin(), a.end());
    // a = [1,2,3]
}
```

------

## 2) Return a New Sorted Vector (Do Not Modify the Original)

```cpp
vector<int> a = {3, 1, 2};
vector<int> b = a;             // copy
sort(b.begin(), b.end());      // sort copy
```

------

## 3) Sort in Descending Order

### Method A: `greater<>`

```cpp
sort(a.begin(), a.end(), greater<int>());
```

### Method B: custom comparator

```cpp
sort(a.begin(), a.end(), [](int x, int y){
    return x > y;
});
```

------

## 4) Sort by Key (Tuples / Pairs / Struct)

### Example: sort by second element of pair

```cpp
void sortByKey() {
    std::vector<std::pair<int, int>> arr = {{1, 5}, {2, 5}, {2, 3}, {3, 4}};
    std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
        return a.first != b.first ? a.first > b.first : a.second > b.second; // descending, descending
    });
    for (auto &p: arr){
        printf("(%d, %d)", p.first, p.second);
    }
    printf("\n");
}
```

------

## 5) Sort by multiple key and value

### 1. Ascending, Ascending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first < b.first : a.second < b.second; // ascending, ascending
});
for (auto &p: arr){
    printf("{%d, %d}", p.first, p.second);
}
printf("\n"); // {1, 5}{2, 3}{2, 5}{3, 4}
```



### 2. Ascending + Descending 

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second; // ascending, descending
});
for (auto &p: arr){
    printf("{%d, %d}", p.first, p.second);
}
printf("\n"); // {1, 5}{2, 5}{2, 3}{3, 4}
```



### 3. Descending, Ascending

```cpp
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second; // descending, ascending
});
for (auto &p: arr){
    printf("{%d, %d}", p.first, p.second);
}
printf("\n"); // {3, 4}{2, 3}{2, 5}{1, 5}
```



### 4.Descending, Descending

```cpp
std::vector<std::pair<int, int>> arr = {{1, 5}, {2, 5}, {2, 3}, {3, 4}};
std::sort(arr.begin(), arr.end(), [](auto &a, auto&b) {
return a.first != b.first ? a.first > b.first : a.second > b.second; // descending, descending
});
for (auto &p: arr){
printf("{%d, %d}", p.first, p.second);
}
printf("\n"); // {3, 4}{2, 5}{2, 3}{1, 5}
```



## 4. Sort strings by length descending, then lexicographically ascending

```cpp
void sortByStringLength() {
    // Sort strings by length descending, then lexicographically ascending
    std::vector<std::string> words =  {"apple", "bat", "banana", "app"};
    std::sort(words.begin(), words.end(), [](const std::string &a, const std::string &b){
        return a.size() != b.size() ? a.size() > b.size() : a < b;
    });
    for (auto &p : words) {
        printf("%s ", p.c_str());
    }
    printf("\n");
}
```

Result:

```
banana apple app bat
```





## 5. Sort a map / unordered_map / Counter by value, then key

Python:

```python
sorted(cnt.items(), key=lambda x: (-x[1], x[0])) # descending, ascending
```

### C++ (unordered_map  + vector)

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> nums = {1,1,1,2,2,3,3,4};

    unordered_map<int,int> cnt;
    for (int x : nums) cnt[x]++;

    vector<pair<int,int>> items(cnt.begin(), cnt.end());

    sort(items.begin(), items.end(), [](auto &a, auto &b){
        if (a.second != b.second) return a.second > b.second; // freq desc
        return a.first < b.first; // num asc
    });

    // items: [(1,3),(2,2),(3,2),(4,1)]
}
```

------





