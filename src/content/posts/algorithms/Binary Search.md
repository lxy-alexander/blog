---
title: Binary Search
published: 2026-01-24
description: "Binary Search"
image: ""
tags: ["Binary Search"]
category: algorithm
draft: false

---

>   Binary search maintains a answer window: Each step removes the half that doesn't contain the answer. When the window shrinks to empty, the left boundary is the answer

# Note

## Why does `end = lowerBound(nums, target + 1) - 1` work?

`lowerBound(nums, target)` returns the index of the first element that is `>= target`.

`lowerBound(nums, target + 1)` finds the **first element that is `>= target + 1`**, Since all numbers are integers, **`>= target + 1` is the same as `> target`**

```
nums   = [1, 2, 2, 2, 4, 6]
target = 2
```

`lowerBound(nums, 3)` → points to `4` (index `4`) ,  `end = 4 - 1 = 3` => Index `3` is the last `2`, which is exactly the last position of `target`.



## What if `>= target + 1` doesn’t exist?

```
nums = [1, 2, 2, 2]
target = 2
```

When if `>= target + 1` doesn't exist,  it will return `n`



## Why do we write `left + (right - left) / 2`?

you may not know how large the input array can be. If the array length reaches the maximum value of an `int`, then `left + right` may overflow.

In Python, since integers do not overflow, you can just add them directly.





##  How can I tell which type of binary search I wrote?

Look at the condition in the `while` loop:

-   If it is `left <= right`, then it’s a **closed interval** (`[left, right]`)↳
-   If it is `left < right`, then it’s a **left-closed right-open interval** (`[left, right)`)↳
-   If it is `left + 1 < right`, then it’s an **open interval** (`(left, right)`)



## `left <= right`

✅ high 指向：最后一个 `< target` 的位置

因为 high 一直在往左收缩，直到它停在“不够 target”的最后一格

✅ low 指向：第一个 `>= target` 的位置

因为 low 一直在往右跳过所有 `< target` 的位置





# Binary Search on Sorted Array

## Find the boundary of element in sorted array

[34. Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with `O(log n)` runtime complexity.

 

**Example 1:**

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

**Example 2:**

```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

**Example 3:**

```
Input: nums = [], target = 0
Output: [-1,-1]
```



```c++
class Solution {
private:
    int lowerBound(vector<int>& nums, int n, int target) {
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] >= target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int n = nums.size();
        int start = lowerBound(nums, n, target);
        if (start == n or nums[start] != target) {
            return {-1, -1};
        }
        // find the first element that is greater than target, then use its index to minus 1, we can get the the end position of target
        int end = lowerBound(nums, n, target + 1) - 1;
        return {start, end};
    }
};
```

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def lower_bound(nums, n, target):
            l = 0
            r = n - 1
            while l <= r:
                m = (l + r) // 2
                if nums[m] >= target:
                    r -= 1
                else:
                    l += 1
            return l
        
        n = len(nums)
        start = lower_bound(nums, n, target)
        if start == n or nums[start] != target:
            return [-1, -1]
        end = lower_bound(nums, n, target + 1) - 1
        return [start, end]

```

