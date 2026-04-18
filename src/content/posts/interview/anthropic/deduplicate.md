---
title: "deduplicate"
published: 2026-04-17
description: "deduplicate"
image: ""
tags: ["interview","anthropic","deduplicate"]
category: interview / anthropic
draft: false
lang: ""
---

# I. File Deduplication (文件去重)

## 1. Problem Statement (题目描述)

==Given a Directory Tree (目录树), find and group files with identical Byte Content (字节内容) and output duplicate file paths where group size ≥ 2.==

**Task Description (任务说明):**

-   Input: A root directory (根目录) containing files and subdirectories
-   Definition: Duplicate Files (重复文件) have exactly the same byte content
-   Output: Groups of file paths (文件路径组), each group contains identical files
-   Format: One line per group, paths separated by spaces (空格分隔路径)

**Example (示例):**

```
/a/1.txt content "hello"
/b/2.txt content "hello"
/c/3.txt content "world"
```

**Output:**

```
/a/1.txt /b/2.txt
```

---

## 2. Core Approach (核心思路)

### 1) Directory Traversal (目录遍历)

Use DFS/BFS (深度优先/广度优先搜索) to visit all files in the directory tree.  ==What should we do if the file cannot be opened?==

### 2) Hashing Files (文件哈希)

Use a Hash Function (哈希函数) to convert file content into a Hash Value (哈希值) for quick comparison.

### 3) Grouping Duplicates (分组重复文件)

Use a Hash Map (哈希表) to map `hash -> list of file paths` and collect duplicates.

---



## 3. Code Implementation (代码实现)

### 1) Python Example (可独立运行)

```python
import os
import hashlib

def get_file_hash(file_path, chunk_size=4096):
    hasher = hashlib.md5()  # MD5哈希函数
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(root_dir):
    hash_map = {}  # 哈希值 -> 文件路径列表

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_hash = get_file_hash(file_path)

            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(file_path)

    for paths in hash_map.values():
        if len(paths) >= 2:
            print(" ".join(paths))

if __name__ == "__main__":
    # Ensure example_dir exists with test files
    find_duplicates("./example_dir")
```

```python
import os
import hashlib
from collections import defaultdict


def get_file_hash(file_path, chunk_size=4096):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates(root_dir):
    size_map = defaultdict(list)   # 文件大小 -> 文件路径列表
    hash_map = defaultdict(list)   # 文件哈希 -> 文件路径列表

    # 1）先遍历所有文件，按文件大小分组
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                size_map[file_size].append(file_path)
            except (OSError, PermissionError) as e:
                print(f"无法读取文件大小: {file_path}, 错误: {e}")

    # 2）只对“大小相同”的文件计算哈希
    for file_size, paths in size_map.items():
        if len(paths) < 2:
            continue

        for file_path in paths:
            try:
                file_hash = get_file_hash(file_path)
                hash_map[file_hash].append(file_path)
            except (OSError, PermissionError) as e:
                print(f"无法读取文件内容: {file_path}, 错误: {e}")

    # 3）输出真正重复的文件
    found = False
    for paths in hash_map.values():
        if len(paths) >= 2:
            found = True
            print(" ".join(paths))

    if not found:
        print("没有找到重复文件")


if __name__ == "__main__":
    find_duplicates("./example_dir")
```

```python
import os
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_file_hash(file_path, chunk_size=4096):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_file_worker(file_path):
    try:
        file_hash = get_file_hash(file_path)
        return file_path, file_hash, None
    except (OSError, PermissionError) as e:
        return file_path, None, e


def find_duplicates(root_dir, max_workers=8):
    size_map = defaultdict(list)

    # 1）先按文件大小分组
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                file_size = os.path.getsize(file_path)
                size_map[file_size].append(file_path)
            except (OSError, PermissionError) as e:
                print(f"无法读取文件大小: {file_path}, 错误: {e}")

    duplicate_groups = []

    # 2）只对大小相同的文件组计算哈希
    for file_size, paths in size_map.items():
        if len(paths) < 2:
            continue

        hash_map = defaultdict(list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(hash_file_worker, file_path) for file_path in paths]

            for future in as_completed(futures):
                file_path, file_hash, error = future.result()
                if error is not None:
                    print(f"无法读取文件内容: {file_path}, 错误: {error}")
                    continue
                hash_map[file_hash].append(file_path)

        # 3）收集真正重复的文件
        for same_files in hash_map.values():
            if len(same_files) >= 2:
                duplicate_groups.append(same_files)

    # 4）输出结果
    if duplicate_groups:
        print("找到重复文件：")
        for i, group in enumerate(duplicate_groups, 1):
            print(f"\n第{i}组重复文件：")
            for path in group:
                print(path)
    else:
        print("没有找到重复文件")


if __name__ == "__main__":
    find_duplicates("./example_dir", max_workers=8)
```



---

## 4. Complexity Analysis (复杂度分析)

### 1) Time Complexity (时间复杂度)

The Time Complexity (时间复杂度) is $$O(N \cdot S)$$ where N is number of files and S is average file size.

### 2) Space Complexity (空间复杂度)

The Space Complexity (空间复杂度) is $$O(N)$$ due to storing hash mappings.

---



## 5. Optimization Strategies (优化策略)

### 1) I/O Bound Optimization (I/O瓶颈优化)

Reduce Disk I/O (磁盘读写) by filtering files using File Size (文件大小) before hashing.

### 2) CPU Bound Optimization (CPU瓶颈优化)

Reduce Hash Computation (哈希计算) cost by using faster hash functions or parallel processing.



# II. Detect Duplicate Files (文件去重-按大小+哈希)

## 1. Problem Statement (题目描述)

==Given File Metadata (文件元数据) and a Content Reading Interface (文件读取接口), detect Duplicate Files (重复文件) where two files are duplicates iff their contents are identical (内容完全相同).==

**Requirements (要求):**

-   Input: A list `files`, each contains:
    -   `path` (路径)
    -   `size` (文件大小，字节)
-   Helper Functions (辅助函数):
    -   `read_file(path) -> bytes` (读取文件内容)
    -   `hash_bytes(data) -> str` (计算哈希值)
-   Output: `List[List[str]]`, each group contains duplicate file paths (每组至少2个文件)

**Constraints (约束):**

-   Number of Files (文件数量) up to $$10^6$$
-   Files may be very large (大文件GB级)
-   Need Streaming Processing (流式处理) to avoid loading entire file into memory

**Optimization Requirement (优化要求):**

-   Stage 1: Group by File Size (按文件大小分组)
-   Stage 2: For same size files, compute Content Hash (内容哈希)

**Example (示例):**

Input:

```id="ex1"
[a.txt size=3 content=abc,
 b.txt size=3 content=abc,
 c.txt size=3 content=abd,
 d.txt size=10 content=0123456789]
```

Output:

```id="ex2"
[[a.txt, b.txt]]
```

---

## 2. Core Idea (核心思路)

### 1) Two-Stage Filtering (两阶段过滤)

First use File Size (文件大小) to prune candidates, then use Hash Function (哈希函数) to confirm duplicates.

### 2) Performance Insight (性能关键点)

This approach reduces expensive I/O (磁盘读取) and Hash Computation (哈希计算).

---

## 3. Algorithm Steps (算法步骤)

### 1) Step Flow (步骤流程)

1.  Build Size Map (大小映射): `size -> list of paths`
2.  Filter groups with size ≥ 2
3.  For each group, compute Hash (计算哈希)
4.  Build Hash Map (哈希映射): `hash -> list of paths`
5.  Collect groups with size ≥ 2

---

## 4. Code Implementation (代码实现)

### 1) Python Example (可独立运行)

```python
import hashlib
from collections import defaultdict

# Mock read_file (模拟读取函数)
def read_file(path):
    data_map = {
        "a.txt": b"abc",
        "b.txt": b"abc",
        "c.txt": b"abd",
        "d.txt": b"0123456789"
    }
    return data_map[path]

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def find_duplicates(files):
    size_map = defaultdict(list)

    # Stage 1: group by file size
    for f in files:
        size_map[f["size"]].append(f["path"])

    result = []

    # Stage 2: group by content hash
    for paths in size_map.values():
        if len(paths) < 2:
            continue

        hash_map = defaultdict(list)
        for path in paths:
            data = read_file(path)
            h = hash_bytes(data)
            hash_map[h].append(path)

        for group in hash_map.values():
            if len(group) >= 2:
                result.append(group)

    return result

if __name__ == "__main__":
    files = [
        {"path": "a.txt", "size": 3},
        {"path": "b.txt", "size": 3},
        {"path": "c.txt", "size": 3},
        {"path": "d.txt", "size": 10},
    ]

    print(find_duplicates(files))
```

---

## 5. Complexity Analysis (复杂度分析)

### 1) Time Complexity (时间复杂度)

The Time Complexity (时间复杂度) is $$O(N + K \cdot S)$$ where K is number of candidate files and S is file size.

### 2) Space Complexity (空间复杂度)

The Space Complexity (空间复杂度) is $$O(N)$$ for storing mappings.

---

## 6. System Design Discussion (系统设计讨论)

### 1) Large File Handling (大文件处理)

Use Streaming Hashing (流式哈希) to process files in chunks to avoid Memory Overflow (内存溢出).

### 2) I/O Bound Optimization (I/O瓶颈优化)

Use Concurrent I/O (并发I/O) and Batch Processing (批处理) to reduce disk latency.

### 3) CPU Bound Optimization (CPU瓶颈优化)

Use Parallel Hashing (并行哈希) with Multi-processing (多进程) to speed up computation.

### 4) Real-time Detection (实时检测)

Use File System Watcher (文件系统监听器) and Incremental Indexing (增量索引) to detect duplicates dynamically.





# III. Find Duplicate Files by Content (按内容查找重复文件)

## 1. Problem Statement (题目描述)

Given a Directory Structure (目录结构), find all files with duplicate content where file content can be compared by a Hash String (哈希字符串).

**Requirements (要求):**

-   Input: A list of strings `paths`, each string contains:
    -   Directory Path (目录路径)
    -   File Name (文件名)
    -   File Content (文件内容)
-   Output: `List[List[str]]`, each group contains file paths with identical content (内容相同的文件路径分组)

**Example (示例):**

Input:

```python
[
    "root/a 1.txt(abcd) 2.txt(efgh)",
    "root/c 3.txt(abcd)",
    "root/c/d 4.txt(efgh)",
    "root 4.txt(1234)"
]
```

Output:

```python
[
    ["root/a/1.txt", "root/c/3.txt"],
    ["root/a/2.txt", "root/c/d/4.txt"]
]
```

**Constraints (约束):**

-   Each input string length (每个输入字符串长度) is less than $$300$$
-   Number of files (文件数量) is less than $$10^4$$

**Extra Example (额外示例):**

Input:

```python
["root/a 1.txt(abcd) 2.txt(efgh)"]
```

Output:

```python
[]
```

------

## 2. Core Idea (核心思路)

### 1) Hash Map Grouping (哈希表分组)

Use a Hash Map (哈希表) to map Content (内容) to Full Paths (完整路径), because the same content should belong to the same group.

### 2) String Parsing (字符串解析)

Split each record into Directory (目录) and File Info (文件信息), then extract File Name (文件名) and Content (内容) from each file token.

------

## 3. Algorithm Steps (算法步骤)

### 1) Step Flow (步骤流程)

1.  Traverse each path string
2.  Split it by spaces into Directory (目录) and File Entries (文件项)
3.  For each file entry, parse File Name (文件名) and Content (内容)
4.  Build Full Path (完整路径)
5.  Store it in Hash Map (哈希表): `content -> list of full paths`
6.  Return groups whose size is at least 2

------

## 4. Code Implementation (代码实现)

### 1) Python Example (可独立运行)

```python
from collections import defaultdict


def find_duplicate(paths):
    content_map = defaultdict(list)

    for record in paths:
        parts = record.split(" ")
        directory = parts[0]

        for file_info in parts[1:]:
            left = file_info.find("(")
            right = file_info.rfind(")")

            file_name = file_info[:left]
            content = file_info[left + 1:right]
            full_path = directory + "/" + file_name

            content_map[content].append(full_path)

    return [group for group in content_map.values() if len(group) >= 2]


if __name__ == "__main__":
    paths = [
        "root/a 1.txt(abcd) 2.txt(efgh)",
        "root/c 3.txt(abcd)",
        "root/c/d 4.txt(efgh)",
        "root 4.txt(1234)"
    ]

    result = find_duplicate(paths)
    print(result)

    extra_input = ["root/a 1.txt(abcd) 2.txt(efgh)"]
    print(find_duplicate(extra_input))
```

```python
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def parse_record(record):
    parts = record.split(" ")
    directory = parts[0]

    local_map = defaultdict(list)

    for file_info in parts[1:]:
        left = file_info.find("(")
        right = file_info.rfind(")")

        file_name = file_info[:left]
        content = file_info[left + 1:right]
        full_path = directory + "/" + file_name

        local_map[content].append(full_path)

    return local_map


def find_duplicate(paths, max_workers=4):
    content_map = defaultdict(list)

    # 多线程解析
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(parse_record, paths)

    # 合并结果
    for local_map in results:
        for content, file_list in local_map.items():
            content_map[content].extend(file_list)

    return [group for group in content_map.values() if len(group) >= 2]


if __name__ == "__main__":
    paths = [
        "root/a 1.txt(abcd) 2.txt(efgh)",
        "root/c 3.txt(abcd)",
        "root/c/d 4.txt(efgh)",
        "root 4.txt(1234)"
    ]

    result = find_duplicate(paths, max_workers=4)
    print(result)

    extra_input = ["root/a 1.txt(abcd) 2.txt(efgh)"]
    print(find_duplicate(extra_input))
```



------

## 5. Complexity Analysis (复杂度分析)

### 1) Time Complexity (时间复杂度)

The Time Complexity (时间复杂度) is $$O(N \cdot K)$$, where $N$ is the number of files and $K$ is the average parsing cost.

### 2) Space Complexity (空间复杂度)

The Space Complexity (空间复杂度) is $$O(N \cdot K)$$, because we store Content (内容) and File Paths (文件路径) in a Hash Map (哈希表).

------

## 6. Interview Notes (面试要点)

### 1) Why Hash Map (为什么用哈希表)

A Hash Map (哈希表) is the most direct way to group files by the same Content (内容).

### 2) Why Not Compare Every Pair (为什么不两两比较)

Pairwise Comparison (两两比较) is too slow at $$O(N^2)$$, so grouping by key is the standard optimization.

### 3) Edge Case (边界情况)

If every file has unique content, the answer is an empty list because no group has at least two files.

------





