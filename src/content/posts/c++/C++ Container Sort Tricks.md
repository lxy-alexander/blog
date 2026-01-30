---
title: "C++ Container Sort Tricks"
published: 2026-01-29
description: "C++ Container Sort Tricks"
image: ""
tags: ["c++","C++ Container Sort Tricks"]
category: c++
draft: false
lang: ""
---



## 1-D `vector<int>`  ascending order / descending order

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<int> v = {5, 2, 9, 1};

    sort(v.begin(), v.end()); // 升序

    for (int x : v) cout << x << " ";

    sort(v.rbegin(), v.rend()); // 降序

    for (int x : v) cout << x << " ";
}
```

✅ 输出：

```
Ascending: 1 2 5 9
Descending: 9 5 2 1
```



## 二维 `vector<vector<int>>` 默认 sort字典序

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<vector<int>> a = {
        {2, 3},
        {1, 5},
        {2, 0},
        {1, 2}
    };

    sort(a.begin(), a.end()); // 默认字典序排序

    cout << "Lexicographical sort:\n";
    for (auto &row : a) {
        cout << row[0] << " " << row[1] << "\n";
    }
}
```

✅ 输出：

```
Lexicographical sort:
1 2
1 5
2 0
2 3
```

------

## `vector<pair<int,int>>` 默认 sort（字典序）

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<pair<int,int>> a = {
        {2, 3},
        {1, 5},
        {2, 0},
        {1, 2}
    };

    sort(a.begin(), a.end()); // 默认字典序

    cout << "Pair sort (default):\n";
    for (auto &p : a) {
        cout << p.first << " " << p.second << "\n";
    }
}
```

✅ 输出：

```
Pair sort (default):
1 2
1 5
2 0
2 3
```

------

# ✅ 例子4：二维 `vector<vector<int>>` 按第1列升序（带输出）

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<vector<int>> a = {
        {2, 3},
        {1, 5},
        {2, 0},
        {1, 2}
    };

    sort(a.begin(), a.end(), [](auto &p, auto &q) {
        return p[1] < q[1];
    });

    cout << "Sort by column 1 ascending:\n";
    for (auto &row : a) {
        cout << row[0] << " " << row[1] << "\n";
    }
}
```

✅ 输出：

```
Sort by column 1 ascending:
2 0
1 2
2 3
1 5
```

------

# ✅ 例子5：二维 `vector<vector<int>>` 按第0列降序（带输出）

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<vector<int>> a = {
        {2, 3},
        {1, 5},
        {2, 0},
        {1, 2}
    };

    sort(a.begin(), a.end(), [](auto &p, auto &q) {
        return p[0] > q[0];
    });

    cout << "Sort by column 0 descending:\n";
    for (auto &row : a) {
        cout << row[0] << " " << row[1] << "\n";
    }
}
```

✅ 输出（可能有多种顺序，只要第0列是降序就对）：

```
Sort by column 0 descending:
2 3
2 0
1 5
1 2
```

------

# ✅ 例子6：二维 `vector<vector<int>>` 按行元素和升序（带输出）

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<vector<int>> a = {
        {2, 3},  // sum = 5
        {1, 5},  // sum = 6
        {2, 0},  // sum = 2
        {1, 2}   // sum = 3
    };

    sort(a.begin(), a.end(), [](auto &p, auto &q) {
        return p[0] + p[1] < q[0] + q[1];
    });

    cout << "Sort by sum ascending:\n";
    for (auto &row : a) {
        cout << row[0] << " " << row[1]
             << " (sum=" << row[0] + row[1] << ")\n";
    }
}
```

✅ 输出：

```
Sort by sum ascending:
2 0 (sum=2)
1 2 (sum=3)
2 3 (sum=5)
1 5 (sum=6)
```

------

如果你想我再给你一个最常用的：
✅ **按第0列升序，第1列降序**（双条件排序）我也可以继续补上。

