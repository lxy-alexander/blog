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

## 二分的前提：单调性

二分能用的核心条件：

-   存在一个函数/条件 `check(x)`
-   具有单调性：
    -   要么 `False False ... True True`
    -   要么 `True True ... False False`

## 怎么设置左闭右闭 `[l, r]`：

>   **r 设成 x 的最大可能取值，而不是 n-1**

-   如果 **x 是数组下标** → `r = n-1`
-   如果 **x 是答案且能到 n** → `r = n`
-   如果 **x 是答案且能到 maxVal** → `r = maxVal`





## 求最少和最多核心区别：你在找哪个边界？

二分答案的 `check(x)` 必须单调：

### 求最少（最小可行值）

目标：找 **最小的 x** 使得 `check(x)=True`

单调形状一般是：

```
F F F F T T T T
        ↑
     第一个 True
```

你要找的是 **第一个 True**

------

### 求最多（最大可行值）

目标：找 **最大的 x** 使得 `check(x)=True`

单调形状一般是：

```
T T T T F F F F
        ↑
     最后一个 True
```

你要找的是 **最后一个 True**









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



## left <= right

✅ high 指向：最后一个 `< target` 的位置

因为 high 一直在往左收缩，直到它停在“不够 target”的最后一格

✅ low 指向：第一个 `>= target` 的位置

因为 low 一直在往右跳过所有 `< target` 的位置



## ⌈b/a⌉ = ⌊(b+a-1)/a⌋

**C++**

```
ceil_div = (b + a - 1) / a = (b - 1) / a + 1;
```

**Python**

```
ceil_div = (b + a - 1) // a = (b - 1) / a + 1
```



## 左右边界怎么设置（通用口诀）

✅ left：最小“合法可能值”

-   时间类：`1`
-   速度类：`1`
-   容量类：`max(...)`
-   divisor：`1`
-   最小距离：`1` 或 `0`

✅ right：一个“肯定可行”的上界

-   1011：`sum(weights)`（一天运完）
-   2187：`min(time) * totalTrips`（最快车跑完全部）
-   875：`max(piles)`（速度=最大堆，1小时1堆）
-   1283：`max(nums)`（d=max(nums) 时每项最多1）

```python
l = 最小合法值
r = 最大可行值
while l <= r:
    mid = (l+r)//2
    if check(mid):
        r = mid - 1
    else:
        l = mid + 1
return l
```





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





### [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)

Given an array of integers `nums` and an integer `threshold`, we will choose a positive integer `divisor`, divide all the array by it, and sum the division's result. Find the **smallest** `divisor` such that the result mentioned above is less than or equal to `threshold`.

Each result of the division is rounded to the nearest integer greater than or equal to that element. (For example: `7/3 = 3` and `10/2 = 5`).

The test cases are generated so that there will be an answer.

 

**Example 1:**

```
Input: nums = [1,2,5,9], threshold = 6
Output: 5
Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1. 
If the divisor is 4 we can get a sum of 7 (1+1+2+3) and if the divisor is 5 the sum will be 5 (1+1+1+2). 
```

**Example 2:**

```
Input: nums = [44,22,33,11,1], threshold = 5
Output: 44
```

 

**Constraints:**

-   `1 <= nums.length <= 5 * 104`
-   `1 <= nums[i] <= 106`
-   `nums.length <= threshold <= 106`



#### 什么时候缩l & r

✅ `S(m) <= threshold`：mid **可行**，答案可能更小 → **往左找** → `r = m - 1`

✅ `S(m) > threshold`：mid **太小**，必须增大 → **往右找** → `l = m + 1` 



#### 为什么退出循环后答案是 `l`？

退出条件是：

```
l=r+1
```

`r` 停在 **最后一个不满足条件的位置**（最后一个 False）

`l` 就是 **它右边第一个位置**（第一个 True）

Because we always move `l` right when `m` is invalid, and move `r` left when `m` is valid, the search ends with `r` at the last invalid value and `l` at the first valid value — therefore `l` is the smallest feasible answer.

因为当 m 无效时我们总是将 l 右移，而当 m 有效时我们将 r 左移，所以搜索结束时 r 在最后一个无效值，l 在第一个有效值——因此 l 是最小的可行答案。

```python
# 这道题你并需要排序，因为题目没有要求是数组中元素，只需要最小值和数组的最大值进行二分法
# (x + m - 1) // m 这个是ceil的实现， import math  math.ceil(x / m) 也可以
# 
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        l, r = 1, max(nums)
        while l <= r:
            m = (l + r) // 2
            total = 0
            if sum((x + m - 1) // m for x in nums) <= threshold:
                r = m - 1
            else:
                l = m + 1
        return l 
```



## Find the least

### [2187. Minimum Time to Complete Trips](https://leetcode.com/problems/minimum-time-to-complete-trips/)

You are given an array `time` where `time[i]` denotes the time taken by the `ith` bus to complete **one trip**.

Each bus can make multiple trips **successively**; that is, the next trip can start **immediately after** completing the current trip. Also, each bus operates **independently**; that is, the trips of one bus do not influence the trips of any other bus.

You are also given an integer `totalTrips`, which denotes the number of trips all buses should make **in total**. Return *the **minimum time** required for all buses to complete **at least*** `totalTrips` *trips*.

**Example 1:**

```
Input: time = [1,2,3], totalTrips = 5
Output: 3
Explanation:
- At time t = 1, the number of trips completed by each bus are [1,0,0]. 
  The total number of trips completed is 1 + 0 + 0 = 1.
- At time t = 2, the number of trips completed by each bus are [2,1,0]. 
  The total number of trips completed is 2 + 1 + 0 = 3.
- At time t = 3, the number of trips completed by each bus are [3,1,1]. 
  The total number of trips completed is 3 + 1 + 1 = 5.
So the minimum time needed for all buses to complete at least 5 trips is 3.
```

**Example 2:**

```
Input: time = [2], totalTrips = 1
Output: 2
Explanation:
There is only one bus, and it will complete its first trip at t = 2.
So the minimum time needed to complete 1 trip is 2.
```

**Constraints:**

-   `1 <= time.length <= 105`
-   `1 <= time[i], totalTrips <= 107`





#### 左边界left 怎么设

-   `left`：不需要 `l` 一开始一定 False, 可以是最小可能答案，可以是是最小可能答案-1，也可以设置为1（多二分几次），但是一般`l` 直接设成“合法的最小值”。
-   `right`：一定够（True）

```
left = min(time) 
```



#### 什么时候可以 `left = lowerBound - 1`？

check 在该范围内是“合法的”，比如 2187 中 `T=min(time)-1` 是合法时间，算 trips 不会出错。





#### 右边界 right 怎么设？

一般保证在 `right` 时间内一定能跑够 `totalTrips`。

最稳的上界：**最快的车一个人跑完所有 trips**：

-   最快车用时 `min(time)`
-   跑 `totalTrips` 趟需要 `min(time) * totalTrips`

```
right = min(time) * totalTrips
```



```python
# 必须保证在 right 时间内一定能跑够 totalTrips， 跑 totalTrips 趟需要 min(time) * totalTrips

class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        min_t = min(time)
        l = min_t
        r = min_t * totalTrips
        while l <= r:
            m = (l + r) // 2
            if sum(m // x for x in time) >= totalTrips:
                r = m - 1
            else:
                l = m + 1
        return l #返回l就好了，这个l就是我们所要的time t

```





### [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)



A conveyor belt has packages that must be shipped from one port to another within `days` days.

传送带上的包裹必须在几天内从一个港口运送到另一个港口。

传送带上的第 i 个包裹的重量为weights[i]。 每天，我们都会在传送带上将包裹装载到船上（按照重量顺序）。 我们装载的重量不得超过船舶的最大承重能力。

The `ith` package on the conveyor belt has a weight of `weights[i]`. Each day, we load the ship with packages on the conveyor belt (in the order given by `weights`). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within `days` days.

 

**Example 1:**

```
Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation: A ship capacity of 15 is the minimum to ship all the packages in 5 days like this:
1st day: 1, 2, 3, 4, 5
2nd day: 6, 7
3rd day: 8
4th day: 9
5th day: 10

Note that the cargo must be shipped in the order given, so using a ship of capacity 14 and splitting the packages into parts like (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) is not allowed.
```

**Example 2:**

```
Input: weights = [3,2,2,4,1,4], days = 3
Output: 6
Explanation: A ship capacity of 6 is the minimum to ship all the packages in 3 days like this:
1st day: 3, 2
2nd day: 2, 4
3rd day: 1, 4
```

**Example 3:**

```
Input: weights = [1,2,3,1,1], days = 4
Output: 3
Explanation:
1st day: 1
2nd day: 2
3rd day: 3
4th day: 1, 1
```

 

**Constraints:**

-   `1 <= days <= weights.length <= 5 * 104`
-   `1 <= weights[i] <= 500`





#### `if cur + w <= cap`  

每天你面对一个包裹 `w` 时只有两种选择：

1.  **装进今天**（如果没超载）
    1.  **装不下 → 明天再装**（开新的一天），但是当前已经在这个包裹了，你需要`cur = w` 去重设`cur`，这样`cur`才不会丢失今天的包裹，错误想法是：`cur = 0`

体现的是 **贪心：能装就装满，装不下就开新的一天**。因为把包裹留到明天只会让天数更多，不会更少。



#### 什么时候不能减1？

`l = max(weights) - 1`
此时可能出现 cap < 某个包裹重量，会导致：包裹永远装不上船（实际上必 False）但你的 check 可能“装不下也硬装”（最好一个包裹没有上船，就直接返回need的天数）→ 算出来天数是不正确的



#### 为什么<= days, 要收缩右边界

因为我们要找的是the least capability, 所以当当前的capability所对应的天数满足题目要求的days要求，说明capability是合法的，但可能大了，所以收缩右边界（简少capability）让其继续二分，当当前的capability所对应的天数大于days，说明capability小了，所以收缩左边界，让capability变大一点。



```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def check(cap):
            need = 1
            cur = 0
            for w in weights:
                if cur + w <= cap:
                    cur += w
                else:
                    need += 1
                    cur = w # 要注意重设cur
            return need
             

        l, r = max(weights), sum(weights)
        while l <= r:
            m = (l + r) // 2
            if check(m) <= days: # 想一想为什么<= days, 要收缩右边界
                r = m - 1
            else:
                l = m + 1
        return l
```





### [475. Heaters](https://leetcode.com/problems/heaters/)

Winter is coming! During the contest, your first job is to design a standard heater with a fixed warm radius to warm all the houses.

Every house can be warmed, as long as the house is within the heater's warm radius range. 

Given the positions of `houses` and `heaters` on a horizontal line, return *the minimum radius standard of heaters so that those heaters could cover all houses.*

**Notice** that all the `heaters` follow your radius standard, and the warm radius will be the same.

 

**Example 1:**

```
Input: houses = [1,2,3], heaters = [2]
Output: 1
Explanation: The only heater was placed in the position 2, and if we use the radius 1 standard, then all the houses can be warmed.
```

**Example 2:**

```
Input: houses = [1,2,3,4], heaters = [1,4]
Output: 1
Explanation: The two heaters were placed at positions 1 and 4. We need to use a radius 1 standard, then all the houses can be warmed.
```

**Example 3:**

```
Input: houses = [1,5], heaters = [2]
Output: 3
```

 

**Constraints:**

-   `1 <= houses.length, heaters.length <= 3 * 104`
-   `1 <= houses[i], heaters[i] <= 109`





If you compute the distance from **each heater to every house**, the time complexity is **O(m·n)**. But if you use **house** as a query point and “insert” it into the sorted `heaters` array using binary search, the time complexity becomes **O(m log n)**.

#### Nearest Neighbor Search 

In a **1D array / number line**:

-   You want to find the point **closest to `x`**
-   After sorting, you don’t need to check all elements
-   You only need to check the **two neighbors around the insertion position** given by `lower_bound`

| Category                                       | Problem / Pattern                    | Key Idea                                                     | Common Technique                        | Typical Time Complexity |
| ---------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ | --------------------------------------- | ----------------------- |
| ⭐ A Nearest / Min Difference                   | **Minimum Absolute Difference**      | After sorting, the closest pair must be adjacent             | `sort + compare neighbors`              | `O(n log n)`            |
| ⭐ A Nearest / Min Difference                   | **Find K Closest Elements**          | Find insertion point, then expand outward                    | `sort + bisect + two pointers`          | `O(log n + k)` (often)  |
| ⭐ B Min Distance Between Two Sets              | **Heaters (this problem)**           | For each house, nearest heater is either left neighbor or right neighbor | `sort + bisect (lower_bound)`           | `O(m log n)`            |
| ⭐ B Min Distance Between Two Sets              | **Shortest Distance to a Character** | For each index, nearest target position matters              | `two passes` or `bisect over positions` | `O(n)` or `O(n log n)`  |
| ⭐ C Coverage / Radius / “Answer Binary Search” | **Magnetic Force Between Two Balls** | Maximize the minimum distance → binary search the answer     | `binary search answer + greedy check`   | `O(n log range)`        |
| ⭐ C Coverage / Radius / “Answer Binary Search” | **Koko Eating Bananas**              | Minimize the maximum speed → binary search the answer        | `binary search answer + check function` | `O(n log range)`        |

```python
# find the closest position on the left and on the right
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        ans = 0
        houses.sort()
        heaters.sort()
        n = len(heaters)
        for x in houses:
            i = bisect.bisect_left(heaters, x)
            # edge case
            # if i == 0, it means x is before the first heater, so there is no heater on the left.
            ld = x - heaters[i - 1] if i > 0 else float('inf')
            rd = heaters[i] - x if i < n else float('inf')
            ans = max(ans, min(ld, rd))
        return ans
```







### [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

Koko loves to eat bananas. There are `n` piles of bananas, the `ith` pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses some pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return *the minimum integer* `k` *such that she can eat all the bananas within* `h` *hours*.

**Example 1:**

```
Input: piles = [3,6,7,11], h = 8
Output: 4
```

**Example 2:**

```
Input: piles = [30,11,23,4,20], h = 5
Output: 30
```

**Example 3:**

```
Input: piles = [30,11,23,4,20], h = 6
Output: 23 
```

**Constraints:**

-   `1 <= piles.length <= 104`
-   `piles.length <= h <= 109`
-   `1 <= piles[i] <= 109`

一定要记住m符合<=h, 我们一直缩小右边界，让其不符合，m>h，我们缩小左边界让其符合，最终的while loop是在l > r结束，最终l是落在符合条件的区间，而r是在不符合条件的区间。

```python
# Because we always move l right when m is invalid, and move r left when m is valid, the search ends with r at the last invalid value and l at the first valid value — therefore l is the smallest feasible answer.
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l = 1
        r = sum(piles)
        while l <= r:
            m = (l + r) // 2
            if sum((x + m - 1) // m for x in piles) <= h:
                r = m - 1
            else:
                l = m + 1
        return l       
```







## Find the most

### [275. H-Index II](https://leetcode.com/problems/h-index-ii/)

Given an array of integers `citations` where `citations[i]` is the number of citations a researcher received for their `ith` paper and `citations` is sorted in **non-descending order**, return *the researcher's h-index*.

According to the [definition of h-index on Wikipedia](https://en.wikipedia.org/wiki/H-index): The h-index is defined as the maximum value of `h` such that the given researcher has published at least `h` papers that have each been cited at least `h` times.

You must write an algorithm that runs in logarithmic time.

**Example 1:**

```
Input: citations = [0,1,3,5,6]
Output: 3
Explanation: [0,1,3,5,6] means the researcher has 5 papers in total and each of them had received 0, 1, 3, 5, 6 citations respectively.
Since the researcher has 3 papers with at least 3 citations each and the remaining two with no more than 3 citations each, their h-index is 3.
```

**Example 2:**

```
Input: citations = [1,2,100]
Output: 2
```

**Constraints:**

-   `n == citations.length`
-   `1 <= n <= 105`
-   `0 <= citations[i] <= 1000`
-   `citations` is sorted in **ascending order**.

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        l, r = 0, n

        # T T T...F F F
        while l <= r:
            h = (l + r) // 2
            if h == 0 or citations[n - h] >= h:
                l = h + 1      # True，尝试更大
            else:
                r = h - 1      # False，变小

        return r
```









### [[LeetCode\] 644. Maximum Average Subarray II 子数组的最大平均值之二](https://www.cnblogs.com/grandyang/p/8021421.html)

Given an array consisting of `n` integers, find the contiguous subarray whose length is greater than or equal to `k` that has the maximum average value. And you need to output the maximum average value.

Example 1:

```
Input: [1,12,-5,-6,50,3], k = 4
Output: 12.75
Explanation:
when length is 5, maximum average value is 10.8,
when length is 6, maximum average value is 9.16667.
Thus return 12.75.
```

Note:

1.  1 <= `k` <= `n` <= 10,000.
2.  Elements of the given array will be in range [-10,000, 10,000].
3.  The answer with the calculation error less than 10-5 will be accepted.



#### 为什么这个条件能二分？

因为它对 `mid` 有单调性：

-   `mid` 越小 → 越容易满足 → `check(mid)` 更可能 True
-   `mid` 越大 → 越难满足 → `check(mid)` 更可能 False

所以形状是：

```
True True True ... False False
```

我们要找的是：

✅ **最大的 mid 让 check(mid)=True**
 也就是最大平均值。

```python
from typing import List

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)

        # check(mid) 用来判断：
        # 是否存在一个长度 >= k 的子数组，使得子数组平均值 >= mid
        def check(mid: float) -> bool:
            # 思路：把每个数都减去 mid，得到新数组 b
            # b[i] = nums[i] - mid
            # 若某个子数组平均值 >= mid
            # <=> 子数组的 sum(nums) / len >= mid
            # <=> sum(nums) - mid * len >= 0
            # <=> sum(nums[i] - mid) >= 0
            #
            # 所以 check(mid) 就变成：
            # 是否存在长度 >= k 的子数组，使得 sum(b) >= 0

            pre = 0.0      # pre 表示当前的前缀和：sum(b[0..i])
            pre_k = 0.0    # pre_k 表示前缀和走到 i-k 的位置：sum(b[0..i-k])
            min_pre = 0.0  # min_pre 表示所有合法起点的最小前缀和，用来最大化 pre - pre[j]

            # 先算长度“刚好为 k”的第一个子数组 b[0..k-1] 的和
            for i in range(k):
                pre += nums[i] - mid

            # 如果前 k 个的和已经 >= 0，说明存在长度 >= k 的子数组平均值 >= mid
            if pre >= 0:
                return True

            # 从下标 k 开始继续往右扩展右端点
            for i in range(k, n):
                # 更新当前前缀和：把 b[i] 加进去
                pre += nums[i] - mid

                # 更新 pre_k：
                # 让“合法起点范围”向右扩一格（保证子数组长度 >= k）
                # pre_k 变成 sum(b[0..i-k])
                pre_k += nums[i - k] - mid

                # min_pre 记录截至目前，所有 pre_k 的最小值
                # 代表我们能选择的起点 j（j <= i-k）里最小的 pre[j]
                min_pre = min(min_pre, pre_k)

                # 如果 pre - min_pre >= 0
                # 说明存在某个起点 j，使得 sum(b[j..i]) = pre - pre[j] >= 0
                # => 存在长度 >= k 的子数组平均值 >= mid
                if pre - min_pre >= 0:
                    return True

            # 扫完整个数组都没找到，则不存在平均值 >= mid 的子数组
            return False

        # 最大平均值一定在 [min(nums), max(nums)] 内
        l, r = min(nums), max(nums)

        # eps 是二分的精度，区间小于这个值就可以停
        eps = 1e-5

        # 二分“最大可行的平均值”
        while r - l > eps:
            mid = (l + r) / 2

            # 如果存在长度>=k子数组平均值 >= mid，说明 mid 可行 -> 往大找
            if check(mid):
                l = mid
            # 否则 mid 不可行 -> 往小找
            else:
                r = mid

        # l 就是逼近的最大平均值
        return l

```





### [2226. Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/)

You are given a **0-indexed** integer array `candies`. Each element in the array denotes a pile of candies of size `candies[i]`. You can divide each pile into any number of **sub piles**, but you **cannot** merge two piles together.

You are also given an integer `k`. You should allocate piles of candies to `k` children such that each child gets the **same** number of candies. Each child can be allocated candies from **only one** pile of candies and some piles of candies may go unused.

Return *the **maximum number of candies** each child can get.*

**Example 1:**

```
Input: candies = [5,8,6], k = 3
Output: 5
Explanation: We can divide candies[1] into 2 piles of size 5 and 3, and candies[2] into 2 piles of size 5 and 1. We now have five piles of candies of sizes 5, 5, 3, 5, and 1. We can allocate the 3 piles of size 5 to 3 children. It can be proven that each child cannot receive more than 5 candies.
```

**Example 2:**

```
Input: candies = [2,5], k = 11
Output: 0
Explanation: There are 11 children but only 7 candies in total, so it is impossible to ensure each child receives at least one candy. Thus, each child gets no candy and the answer is 0.
```

 

**Constraints:**

-   `1 <= candies.length <= 105`
-   `1 <= candies[i] <= 107`
-   `1 <= k <= 1012`

```python
# 求最大二分的条件是TTTTFFF, 每个人最少是1个，最多可能是最大值，所以二分时的m符合条件，需要l = m + 1，缩小左边界，不符合缩小右边界
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        def ok(m):
            cnt = 0
            for x in candies:
                cnt += x // m
                if cnt >= k:
                    return True
            return False
        l = 1
        r = max(candies)
        while l <= r:
            m = (l + r) // 2
            if ok(m):
                l = m + 1
            else:
                r = m - 1
        return r

```



