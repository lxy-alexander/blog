---
title: "Find Duplicate Files"
published: 2026-03-09
description: "Find Duplicate Files"
image: ""
tags: ["interview","anthropic","Find Duplicate Files"]
category: interview / anthropic
draft: false
lang: ""
---

# **I. Problem Description**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Given a <span style="color:#E8600A;font-weight:700">directory structure (目录结构)</span>, find all files with <span style="color:#E8600A;font-weight:700">duplicate content (重复内容)</span>. Each file's content is simplified using a <span style="color:#E8600A;font-weight:700">hash function (哈希函数)</span> into a comparable string. The goal is to group all files sharing identical content and return each group as a list of full paths. </div>

## 1. Input Format

<span style="color:#E8600A">1.</span> The input is a **list of strings (字符串列表)**, where each string represents one directory and its files.

<span style="color:#E8600A">2.</span> Each string follows the pattern:

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">"directory filename1(content1) filename2(content2) ..."</code>

<span style="color:#E8600A">3.</span> The <span style="color:#2980B9">first token (第一个标记)</span> is the directory path; the remaining tokens are `filename(content)` pairs.

**Example Input:**

```
[
  "root/a 1.txt(abcd) 2.txt(efgh)",
  "root/c 3.txt(abcd)",
  "root/c/d 4.txt(efgh)",
  "root 4.txt(1234)"
]
```

## 2. Output Format

<span style="color:#E8600A">1.</span> Output a **list of groups (分组列表)**. Each group contains the full paths of all files sharing the same content.

<span style="color:#E8600A">2.</span> Files with <span style="color:#2980B9">unique content (唯一内容)</span> (no duplicates) are **excluded** from the result.

**Example Output:**

```
[
  ["root/a/1.txt", "root/c/3.txt"],
  ["root/a/2.txt", "root/c/d/4.txt"]
]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The file <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">root/4.txt(1234)</code> does not appear in the output because its content <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">1234</code> is unique — no other file shares it.</div>

## 3. Constraints

| Constraint (约束条件)                                    | Value                        |
| -------------------------------------------------------- | ---------------------------- |
| Max length of each input string (每条输入字符串最大长度) | < 300                        |
| Max number of files (最大文件数量)                       | < 10⁴                        |
| Content is pre-hashed (内容已哈希)                       | Yes — treat as opaque string |

------

# **II. Implementation**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> The core idea is to use a <span style="color:#E8600A;font-weight:700">Hash Map (哈希表)</span> that maps each <span style="color:#E8600A;font-weight:700">content string (内容字符串)</span> to a list of full file paths. After parsing all input strings, we collect only the groups whose size is <span style="color:#E8600A;font-weight:700">≥ 2</span>, i.e., files with duplicated content. The Time Complexity (时间复杂度) is <span style="color:#E8600A;font-weight:700">O(N)</span> where N is the total number of files. </div>

## 1. Algorithm Walkthrough

### 1) Parse Each Input String

<span style="color:#E8600A">1.</span> <span style="color:#2980B9">Split (分割)</span> the string by spaces to get tokens.

<span style="color:#E8600A">2.</span> The <span style="color:#2980B9">first token</span> is the **directory path (目录路径)**.

<span style="color:#E8600A">3.</span> For each remaining token, extract the **filename (文件名)** and **content (内容)** using the pattern <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">name(content)</code>.

<span style="color:#E8600A">4.</span> Construct the full path as <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">directory + "/" + filename</code>.

### 2) Build the Hash Map

<span style="color:#E8600A">1.</span> Use a dictionary / unordered_map: <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">content → [path1, path2, ...]</code>

<span style="color:#E8600A">2.</span> For each file, append its full path to the list keyed by its content string.

### 3) Filter Duplicates

<span style="color:#E8600A">1.</span> Iterate through the map entries.

<span style="color:#E8600A">2.</span> Include only entries where the list has <span style="color:#E8600A;font-weight:700">length ≥ 2</span>.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> We use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">find('(')</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">find(')')</code> to extract the content between parentheses. In Python, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">split('(')[1].rstrip(')')</code> is a clean alternative.</div>

## 2. Python Implementation

```python
from collections import defaultdict
from typing import List

class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        # Map: file content → list of file paths
        groups = defaultdict(list)

        for entry in paths:
            parts = entry.split(' ')
            directory = parts[0]
            files = parts[1:]

            for file in files:
                left = file.index('(')

                file_name = file[:left]
                file_content = file[left + 1:-1]

                full_path = f"{directory}/{file_name}"
                groups[file_content].append(full_path)

        return [
            file_paths
            for file_paths in groups.values()
            if len(file_paths) > 1
        ]
```

:::note

defaultdict(int)     # 默认 0
defaultdict(list)    # 默认 []
defaultdict(set)     # 默认 set()

:::



## 3. Complexity Analysis

| Metric (指标)                 | Value                                                       | Explanation                                       |
| ----------------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| Time Complexity (时间复杂度)  | <span style="color:#E8600A;font-weight:700">O(N × L)</span> | N = total files; L = avg token length for hashing |
| Space Complexity (空间复杂度) | <span style="color:#E8600A;font-weight:700">O(N × L)</span> | Hash map stores all file paths                    |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In practice, because the content is already a <span style="color:#E8600A;font-weight:700">hash string (哈希字符串)</span> (pre-hashed), the string comparison cost L is bounded and small. If raw file content were used, hashing would be essential to reduce comparison cost from O(file_size) to O(1).</div>

## 4. Follow-up Considerations

### 1) If the system had millions of files (百万级文件)?

<span style="color:#E8600A">1.</span> <span style="color:#2980B9">First-pass filter (第一轮过滤)</span>: Group files by **file size (文件大小)**. Only files of identical size can be duplicates — this avoids hashing most files.

<span style="color:#E8600A">2.</span> <span style="color:#2980B9">Second-pass filter (第二轮过滤)</span>: For size-matched groups, compute a **fast hash (快速哈希)** (e.g., MD5 of first 1KB) to narrow candidates.

<span style="color:#E8600A">3.</span> <span style="color:#2980B9">Full hash (完整哈希)</span>: Only perform a full SHA-256 hash on remaining candidates to confirm true duplicates.

### 2) What if we cannot read all files into memory (无法全部读入内存)?

<span style="color:#C0392B;font-weight:600">⚠ Warning:</span> Never load entire file contents into RAM when dealing with large files. Use <span style="color:#E8600A;font-weight:700">streaming hash (流式哈希)</span> — read and hash file in chunks (块), updating the hash incrementally.

### 3) What if duplicate detection must be real-time (实时检测)?

<span style="color:#E8600A">1.</span> Maintain a <span style="color:#E8600A;font-weight:700">persistent hash store (持久化哈希存储)</span> (e.g., Redis or a database) that maps content hash → file list.

<span style="color:#E8600A">2.</span> On each new file write, compute its hash and query the store to detect duplicates immediately.

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Use a <span style="color:#E8600A;font-weight:700">Hash Map (哈希表)</span> keyed by content string to group file paths in a single O(N) pass, then return only groups with 2 or more entries as duplicates.</div>
