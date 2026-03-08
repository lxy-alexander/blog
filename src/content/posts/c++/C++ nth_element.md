---
title: "C++ nth_element"
published: 2026-03-07
description: "C++ nth_element"
image: ""
tags: ["c++","C++ nth_element"]
category: c++
draft: false
lang: ""
---

# **I. `nth_element` (第K大/小元素)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nth_element</code> is a <span style="color:#E8600A;font-weight:700">Partial Sort (部分排序)</span> algorithm — it guarantees only that <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nums[k-1]</code> holds the correct k-th element, but the rest of the array is <span style="color:#C0392B;font-weight:600">not sorted</span>. Average time complexity is <span style="color:#E8600A;font-weight:700">O(n)</span>, faster than full sort O(n log n).
</div>

---

## <span style="color:#E8600A">1.</span> **Basic Usage (基本用法)**

```cpp
nth_element(nums.begin(), nums.begin() + (k - 1), nums.end(), greater<int>());
// nums[k-1] is the k-th largest element ✅
```

### 1) Partition Guarantee (分区保证)

After the call:

- All elements at index `i < k-1` satisfy: `nums[i] >= nums[k-1]`
- All elements at index `i > k-1` satisfy: `nums[i] <= nums[k-1]`

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>Segments <code>[0 ... k-2]</code> and <code>[k ... n-1]</code> are <span style="color:#C0392B;font-weight:600">not guaranteed to be sorted</span> — only partitioned. Do not rely on their order.
</div>

---

## <span style="color:#E8600A">2.</span> **Comparators (比较器)**

### 1) Built-in comparators (内置比较器)

```cpp
less<int>()     // ascending (升序): a < b  — same as default sort
greater<int>()  // descending (降序): a > b
```

### 2) Custom comparator via lambda (自定义比较器)

```cpp
auto cmp = [](int a, int b) {
    return a > b;  // descending (降序)
};

sort(nums.begin(), nums.end(), cmp);
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px">
<span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nth_element</code> <span style="color:#E8600A;font-weight:700">partitions (分区)</span>, it does NOT sort — use it when you only need the k-th value, not a fully sorted array.
</div>