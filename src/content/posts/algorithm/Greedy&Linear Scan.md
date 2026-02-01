---
title: "Greedy&Linear Scan"
published: 2026-01-27
description: "Greedy&Linear Scan"
image: ""
tags: ["algorithm","Greedy&Linear Scan"]
category: algorithm
draft: false
lang: ""
---

>This algorithm is a **linear scan (one-pass) greedy counting algorithm**. 
>
>-   **Linear Scan / One-pass Traversal**
>-   **Greedy** (always keeps the best answer so far)
>-   Can also be seen as a simple **state tracking** approach



### [485. Max Consecutive Ones](https://leetcode.com/problems/max-consecutive-ones/)

Given a binary array `nums`, return *the maximum number of consecutive* `1`*'s in the array*.

**Example 1:**

```
Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
```

**Example 2:**

```
Input: nums = [1,0,1,1,0,1]
Output: 2
```

**Constraints:**

-   `1 <= nums.length <= 105`
-   `nums[i]` is either `0` or `1`.

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        mx = 0
        cnt = 0
        for x in nums:
            if x == 1:
                cnt += 1
            else:
                mx = max(mx, cnt)
                cnt = 0
            mx = max(mx, cnt)
        return mx
```

```c++
class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int ans = 0;
        int cnt = 0;
        for (int x : nums) {
            if (x == 0) {
                ans = max(ans, cnt);
                cnt = 0;
            } else {
                cnt += 1;
            }
        }
        ans = max(ans, cnt);
        return ans;
    }
};
```





### [1446. Consecutive Characters](https://leetcode.com/problems/consecutive-characters/)

The **power** of the string is the maximum length of a non-empty substring that contains only one unique character.

Given a string `s`, return *the **power** of* `s`.

**Example 1:**

```
Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.
```

**Example 2:**

```
Input: s = "abbcccddddeeeeedcba"
Output: 5
Explanation: The substring "eeeee" is of length 5 with the character 'e' only.
```

**Constraints:**

-   `1 <= s.length <= 500`
-   `s` consists of only lowercase English letters.

```python
class Solution:
    def maxPower(self, s: str) -> int:
        ans = 1
        cnt = 1
        for i in range(len(s)):
            if s[i] == s[i - 1]:
                cnt += 1
            else:
                cnt = 1
            ans = max(ans, cnt)
        return ans
```

```c++
class Solution {
public:
    int maxPower(string s) {
        int ans = 0;
        int l = 0;
        int n = s.size();
        map<char, int> cnt;
        for (int r = 0; r < n; r++) {
            cnt[s[r]]++;
            while (cnt.size() > 1) {
                cnt[s[l]]--;
                if (cnt[s[l]] == 0) {
                    cnt.erase(s[l]);
                }
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```









