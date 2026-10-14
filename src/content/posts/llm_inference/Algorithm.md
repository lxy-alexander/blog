---
title: "Algorithm"
published: 2026-10-11
description: "Algorithm"
image: ""
tags: ["llm_inference","Algorithm"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-10-11T04:03:29.449.257117440Z"
---





# 摩尔投票法

## 169. 多数元素 Majority Element

出现次数 **> n/2**，题目保证一定存在。

```python
class Solution:
    def majorityElement(self, nums: list[int]) -> int:
        cand, count = 0, 0
        for num in nums:
            if count == 0:
                cand = num
            count += 1 if num == cand else -1
        return candidate
```



## 面试题 17.10. 主要元素 Find Majority Element LCCI

出现次数 **> n/2**，但**不保证存在**，不存在时返回 -1。所以投票结束后要再遍历一次**验证**。

```python
class Solution:
    def majorityElement(self, nums: list[int]) -> int:
        cand, count = 0, 0
        for num in nums:
            if count == 0:
                cand = num
            count += 1 if num == cand else -1
        # 验证 verify
        return cand if nums.count(cand) > len(nums) // 2 else -1
```



## 229. 多数元素 II Majority Element II

找出所有出现次数 **> n/3** 的元素。这样的元素最多只有 **2 个**，所以维护**两个候选人**。当前元素和两个候选人都不同时，三者**同时抵消**（两个票数各减 1）。最后同样需要验证。

```python
class Solution:
    def majorityElement(self, nums: list[int]) -> list[int]:
        cand1, cand2, cnt1, cnt2 = 0, 1, 0, 0
        for num in nums:
            if num == cand1:
                cnt1 += 1
            elif num == cand2:
                cnt2 += 1
            elif cnt1 == 0:
                cand1, cnt1 = num, 1
            elif cnt2 == 0:
                cand2, cnt2 = num, 1
            else:
                cnt1 -= 1
                cnt2 -= 1
        # 验证 verify
        # return [c for c in (cand1, cand2) if nums.count(c) > len(nums) // 3]
        res = []
        for c in (cand1, cand2):              # 依次检查两个候选人
            if nums.count(c) > len(nums) // 3:  # 实际出现次数是否 > n/3
                res.append(c)                   # 满足就加入结果
        return res
```





# 轮转 / 旋转

Given an integer array `nums`, rotate the array to the right by `k` steps, where `k` is non-negative.
