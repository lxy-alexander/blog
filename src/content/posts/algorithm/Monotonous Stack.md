---
title: "Monotonous Stack"
published: 2026-06-09
description: "Monotonous Stack"
image: ""
tags: ["algorithm","Monotonous Stack"]
category: algorithm
draft: false
lang: ""
createdAt: "2026-06-10T00:32:22.985.734388522Z"
---

## [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

Given an array of integers `temperatures` represents the daily temperatures, return *an array* `answer` *such that* `answer[i]` *is the number of days you have to wait after the* `ith` *day to get a warmer temperature*. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

 

**Example 1:**

```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

**Example 2:**

```
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
```

**Example 3:**

```
Input: temperatures = [30,60,90]
Output: [1,1,0]
```

 

**Constraints:**

-   `1 <= temperatures.length <= 105`
-   `30 <= temperatures[i] <= 100`



Scan from right to left, pop all days that are not warmer than today, then the stack top is the nearest warmer day.

The temperature at the top of the stack is it is the closest temperature.

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        st = []
        for i in range(n - 1, -1, -1):
            t = temperatures[i]
            while st and t >= temperatures[st[-1]]:
                st.pop()
            if st:
                ans[i] = st[-1] - i
            st.append(i)
        return ans
```









## [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return *the area of the largest rectangle in the histogram*.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/histogram)

```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
```

Keep an increasing stack of indices. When a shorter bar appears, pop taller bars and compute their max rectangle. For each popped bar, current i is the right boundary and the new stack top is the left boundary.

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        st = []
        ans = 0
        heights.append(0)

        for i, h in enumerate(heights):
            while st and h < heights[st[-1]]:
                j = st.pop()
                if st:
                    width = i - st[-1] - 1
                else:
                    width = i
                ans = max(ans, width * heights[j])
            st.append(i)
        
        return ans
```





## [85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

Given a `rows x cols` binary `matrix` filled with `0`'s and `1`'s, find the largest rectangle containing only `1`'s and return *its area*.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/maximal)

```
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.
```

**Example 2:**

```
Input: matrix = [["0"]]
Output: 0
```

**Example 3:**

```
Input: matrix = [["1"]]
Output: 1
```

 

**Constraints:**

-   `rows == matrix.length`
-   `cols == matrix[i].length`
-   `1 <= rows, cols <= 200`
-   `matrix[i][j]` is `'0'` or `'1'`.

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        def get_largest_rectangle(heights):
            st = []
            mx = 0
            for i, h in enumerate(heights):
                while st and heights[st[-1]] > h:
                    j = st.pop()
                    if st:
                        width = i - st[-1] - 1
                    else:
                        width = i
                    mx = max(mx, width * heights[j])
                st.append(i)
            return mx
            

        ans = 0
        m, n = len(matrix), len(matrix[0])
        heights = [0] * (n + 1)

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '0':
                    heights[j] = 0
                else:
                    heights[j] += 1
                
            ans = max(ans, get_largest_rectangle(heights))
        return ans

```











## [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is **the smallest in lexicographical order** among all possible results.

 

**Example 1:**

```
Input: s = "bcabc"
Output: "abc"
```

**Example 2:**

```
Input: s = "cbacdcbc"
Output: "acdb"
```

 

**Constraints:**

-   `1 <= s.length <= 104`
-   `s` consists of lowercase English letters.



```python
from collections import Counter

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # Count how many times each character still appears later
        cnt = Counter(s)

        # Stack stores the current lexicographically smallest result
        st = []

        # check the current character if it is included in the result
        used = set()

        for ch in s:
            # Current character is being used now, so reduce its remaining count
            cnt[ch] -= 1

            # Skip if this character is already in the result
            if ch in used:
                continue

            # If the current character is smaller than the stack top,
            # and the stack top appears again later,
            # we can safely remove the stack top to get a smaller result.
            while st and ch < st[-1] and cnt[st[-1]] > 0:
                used.remove(st.pop())

            # Add current character to the result
            used.add(ch)
            st.append(ch)

        return ''.join(st)
```









## [402. Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

Given string num representing a non-negative integer `num`, and an integer `k`, return *the smallest possible integer after removing* `k` *digits from* `num`.

 

**Example 1:**

```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
```

**Example 2:**

```
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
```

**Example 3:**

```
Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.
```

 

**Constraints:**

-   `1 <= k <= num.length <= 105`
-   `num` consists of only digits.
-   `num` does not have any leading zeros except for the zero itself.



```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        # Stack stores the current smallest possible number
        st = []

        for d in num:
            # If current digit is smaller than the previous digit,
            # remove the previous digit to make the number smaller.
            # We can only remove while k > 0.
            while st and d < st[-1] and k > 0:
                k -= 1
                st.pop()

            # Add current digit
            st.append(d)

        # If we still need to remove digits,
        # remove them from the end.
        while k > 0:
            st.pop()
            k -= 1

        # Remove leading zeros.
        # If nothing is left, return "0".
        return ''.join(st).lstrip('0') or '0'
```







## [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:

1.  Open brackets must be closed by the same type of brackets.
2.  Open brackets must be closed in the correct order.
3.  Every close bracket has a corresponding open bracket of the same type.

 

**Example 1:**

**Input:** s = "()"

**Output:** true

**Example 2:**

**Input:** s = "()[]{}"

**Output:** true

**Example 3:**

**Input:** s = "(]"

**Output:** false

**Example 4:**

**Input:** s = "([])"

**Output:** true

**Example 5:**

**Input:** s = "([)]"

**Output:** false

 

```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2:
            return False
        st = []
        mp = {'(': ')', '[': ']', '{': '}'}
        for c in s:
            if c in mp:
                st.append(mp[c])
            elif not st or c != st.pop():
                return False
        return not st
```







## [1047. Remove All Adjacent Duplicates In String](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

You are given a string `s` consisting of lowercase English letters. A **duplicate removal** consists of choosing two **adjacent** and **equal** letters and removing them.

We repeatedly make **duplicate removals** on `s` until we no longer can.

Return *the final string after all such duplicate removals have been made*. It can be proven that the answer is **unique**.

 

**Example 1:**

```
Input: s = "abbaca"
Output: "ca"
Explanation: 
For example, in "abbaca" we could remove "bb" since the letters are adjacent and equal, and this is the only possible move.  The result of this move is that the string is "aaca", of which only "aa" is possible, so the final string is "ca".
```

**Example 2:**

```
Input: s = "azxxzy"
Output: "ay"
```

 

**Constraints:**

-   `1 <= s.length <= 105`
-   `s` consists of lowercase English letters.



```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        st = []
        for c in s:
            if st and c == st[-1]:
                st.pop()
            else:
                st.append(c)
        return ''.join(st)
```







## [1944. Number of Visible People in a Queue](https://leetcode.com/problems/number-of-visible-people-in-a-queue/)

There are `n` people standing in a queue, and they numbered from `0` to `n - 1` in **left to right** order. You are given an array `heights` of **distinct** integers where `heights[i]` represents the height of the `ith` person.

A person can **see** another person to their right in the queue if everybody in between is **shorter** than both of them. More formally, the `ith` person can see the `jth` person if `i < j` and `min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1])`.

Return *an array* `answer` *of length* `n` *where* `answer[i]` *is the **number of people** the* `ith` *person can **see** to their right in the queue*.

![image-20260611023431158](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260611023431158)

```
Input: heights = [10,6,8,5,11,9]
Output: [3,1,2,1,1,0]
Explanation:
Person 0 can see person 1, 2, and 4.
Person 1 can see person 2.
Person 2 can see person 3 and 4.
Person 3 can see person 4.
Person 4 can see person 5.
Person 5 can see no one since nobody is to the right of them.
```

**Example 2:**

```
Input: heights = [5,1,2,3,10]
Output: [4,1,1,1,0]
```

 

**Constraints:**

-   `n == heights.length`
-   `1 <= n <= 105`
-   `1 <= heights[i] <= 105`
-   All the values of `heights` are **unique**.



```python
class Solution:
    def canSeePersonsCount(self, heights: List[int]) -> List[int]:
        n = len(heights)
        st = []
        ans = [0] * n
        for i in range(n - 1, -1, -1):
            h = heights[i]
            while st and h > st[-1]:
                st.pop()
                ans[i] += 1
            if st:
                ans[i] += 1
            st.append(h)
        return ans
```







## [42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/rainwatertrap)

```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
```

**Example 2:**

```
Input: height = [4,2,0,3,2,5]
Output: 9
```

 

**Constraints:**

-   `n == height.length`
-   `1 <= n <= 2 * 104`
-   `0 <= height[i] <= 105`



```
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        n = len(height)
        st = []
        for i, h in enumerate(height):
            while st and h > height[st[-1]]:
                btm = st.pop()
                if not st:
                    break
                left = st[-1]
                width = i - left - 1
                dh = min(height[left], h) - height[btm]
                ans += width * dh
            st.append(i)
        return ans
```

