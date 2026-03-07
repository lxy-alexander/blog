---
title: Binary Search
published: 2026-01-24
description: "Binary Search"
image: ""
tags: ["algorithm","Binary Search"]
category: algorithm
draft: false

---

# **I.** Binary Search Master Guide: From Logic to Universal Templates

## 1. The Core Essence(底层/核心本质): Monotonicity

The core of Binary Search isn't "sorting," but **"Binary Properties" (Two-Segment Property(二段性))**. As long as a function `check(x)` exists such that the search range presents one of the following two patterns, Binary Search is applicable(合适的，恰当的):

-   **Find Minimum (First True):** `[False, False, ..., True, True]`
-   **Find Maximum (Last True):** `[True, True, ..., False, False]`

------

## 2. Determining the Search Range `[left, right]`

Set the boundaries based on the **physical meaning of x**:

| **Type**     | **left (Min Valid Value)** | **right (Definitely Feasible)** | **Classic Problem**                                          |
| ------------ | -------------------------- | ------------------------------- | ------------------------------------------------------------ |
| **Index**    | `0`                        | `n - 1`                         | Basic Search                                                 |
| **Time**     | `1`                        | `min(time) * totalTrips`        | [2187. Min Time](https://leetcode.com/problems/minimum-time-to-complete-trips/) |
| **Speed**    | `1`                        | `max(piles)`                    | [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) |
| **Capacity** | `max(weights)`             | `sum(weights)`                  | [1011. Ship Packages](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) |
| **Divisor**  | `1`                        | `max(nums)`                     | [1283. Smallest Divisor](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/) |
| **Distance** | `0` or `1`                 | `max(pos) - min(pos)`           | [1552. Magnetic Force](https://leetcode.com/problems/magnetic-force-between-two-balls/) |

------

### 1) Time-based: [2187. Minimum Time to Complete Trips](https://leetcode.com/problems/minimum-time-to-complete-trips/)

-   **Summary**: Given the time each car takes to complete one trip, find the **minimum total time** required for all cars to complete at least `totalTrips`.
-   **Why these boundaries**:
    -   **`left = 1`**: Time cannot be zero.
    -   **`right = min(time) * totalTrips`**: This is a conservative upper bound. Even if only the **fastest car** were running, the time it takes to finish all trips alone would certainly be enough.

### 2) Speed-based: [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

-   **Summary**: There are $n$ piles of bananas. You must finish all of them within $h$ hours. Find the **minimum eating speed** $K$ (bananas per hour). Note: Koko can only eat from one pile per hour.
-   **Why these boundaries**:
    -   **`left = 1`**: Speed must be at least 1, or she will never finish.
    -   **`right = max(piles)`**: If your speed equals the **largest pile**, you are guaranteed to finish one pile per hour. Since you can't eat more than one pile an hour anyway, any speed higher than this is redundant.

### 3) Capacity-based: [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

-   **Summary**: Packages must be shipped in the order given within $D$ days. Find the **minimum weight capacity** of the conveyor belt.
-   **Why these boundaries**:
    -   **`left = max(weights)`**: The belt must be able to carry the **heaviest** single package; otherwise, that package can never be shipped.
    -   **`right = sum(weights)`**: The extreme case—shipping every single package on the very first day. The total sum of weights is the absolute maximum capacity needed.

### 4) Divisor-based: [1283. Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/)

-   **Summary**: Each number in an array is divided by $d$ (rounded up) and summed. The sum must be $\le$ a given threshold. Find the **minimum** $d$.
-   **Why these boundaries**:
    -   **`left = 1`**: The divisor cannot be zero.
    -   **`right = max(nums)`**: When the divisor equals the maximum value in the array, every result becomes $1$ (except the max itself which also becomes 1). This is the effective boundary that reduces the "sum" to its minimum possible value (the length of the array $n$).

### 5) Distance-based: [1552. Magnetic Force Between Two Balls](https://leetcode.com/problems/magnetic-force-between-two-balls/)

-   **Summary**: Place $M$ balls in baskets such that the **minimum distance** between any two balls is as large as possible. Find this **maximum minimum distance**.
-   **Why these boundaries**:
    -   **`left = 1`**: The balls must be separated by at least 1 unit of distance (assuming distinct basket positions).
    -   **`right = max(pos) - min(pos)`**: The theoretical maximum distance occurs when you place only two balls: one at the very first basket and one at the very last.

------

**Would you like me to combine all these English sections into one single, clean Markdown file for you to save?**



------

## 3. Core Templates: Closed Interval `while l <= r`

This is the most robust implementation. It is recommended to use this consistently.

### 1) Find Minimum (First True)

**Goal:** Find the smallest $x$ such that `check(x)` is True.

Python

```
l, r = min_valid, max_feasible
while l <= r:
    mid = l + (r - l) // 2
    if check(mid): # Feasible, but look for smaller ones to the left
        r = mid - 1
    else:          # Not feasible, must increase x
        l = mid + 1
return l  # When loop ends, l points to the first True
```

### 2) Find Maximum (Last True)

**Goal:** Find the largest $x$ such that `check(x)` is True.

Python

```
while l <= r:
    mid = l + (r - l) // 2
    if check(mid): # Feasible, try to find a larger one to the right
        l = mid + 1
    else:          # Not feasible, must decrease x
        r = mid - 1
return r  # When loop ends, r points to the last True
```

------

## 4. Advanced Tips & Mathematical Details

### 1) Finding Left/Right Boundaries of Elements

-   **Left Boundary (Lower Bound):** First index where `element >= target`.
-   **Right Boundary (Upper Bound):** Last index where `element == target`.
    -   **Trick:** `lowerBound(target + 1) - 1`.
    -   **Principle:** Find the start of the first number `> target`, then move back one spot. If `target + 1` doesn't exist, the search returns `n`, and `n - 1` correctly identifies the last element.

### 2) Avoiding Overflow

In C++/Java, `left + right` can exceed $2^{31} - 1$.

-   **Standard approach:** `mid = left + (right - left) / 2`
-   **Python Note:** Though Python handles arbitrarily large integers, keeping this habit helps in understanding low-level memory constraints.

### 3) Ceiling Division Conversion

When calculating "required days/trips," you often need $\lceil \frac{b}{a} \rceil$:

-   **Universal Formula:** `(b + a - 1) // a`
-   **Logic:** As long as $b$ is not perfectly divisible by $a$, adding $a-1$ will always force the integer division to round up by one.

------

## 5. Post-Loop State Cheat Sheet

When the `while l <= r` loop terminates:

| **Pointer**    | **Physical Meaning**                                         |
| -------------- | ------------------------------------------------------------ |
| **`l (low)`**  | Points to the **first** element that **satisfies condition** (or `>= target`) |
| **`r (high)`** | Points to the **last** element that **fails condition** (or `< target`) |





# **II**.Binary Search Questions

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



```python
# Note: You don't need to sort the input array here because the answer is not required to be an element of the array; we are searching within the range from 1 to max(nums).
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



#### 1) When to Shrink `l` & `r`?

The direction of your search depends on the condition:

-   **`sum(mid) <= threshold`**: `mid` is **feasible**, but a smaller answer might exist → **Search Left** → `r = mid - 1`
-   **`sum(mid) > threshold`**: `mid` is **too small**, you must increase it to satisfy the condition → **Search Right** → `l = mid + 1`



#### 2) Why is `l` the answer after the loop?

The loop terminates when `l > r` (specifically, `l = r + 1`).

-   **`r`** stops at the **last position that fails the condition** (the last `False`).
-   **`l`** stops exactly one position to the right of `r` (the first `True`).

**Logic:** Because we only move `l` right when `mid` is invalid, and move `r` left when `mid` is valid, the search concludes with(结束于) `r` at the last invalid value and `l` at the first valid value. Therefore, `l` is the **smallest feasible answer**.





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



```python
# Ensure that the right boundary is large enough to cover totalTrips. The most reliable way is min(time) * totalTrips.
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
        # After the loop, 'l' is the smallest time that satisfies the condition
        return l
```





### [1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)



A conveyor(传送带，传送装置；传播者，传达者) belt(腰带，皮带；传送带；地带) has packages that must be shipped from one port to another within `days` days.

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

```python
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        # l: must be at least the heaviest package
        # r: the sum of all packages (shipping everything in 1 day)
        l, r = max(weights), sum(weights)
        
        while l <= r:
            mid = (l + r) // 2
            
            # Greedy check: how many days are needed with capacity 'mid'?
            need = 1
            cur = 0
            for w in weights:
                if cur + w <= mid:
                    cur += w
                else:
                    need += 1
                    cur = w # Start new day with the current package
            
            if need <= days:
                # Valid capacity, try to find a smaller one
                r = mid - 1
            else:
                # Capacity too small, need more power
                l = mid + 1
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





```python
# Strategy: For each house, find the nearest heaters on both the left and right sides.
# The minimum radius required for a house is the distance to its closest heater.
# The global answer is the maximum of these minimum distances.
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        ans = 0
        houses.sort()
        heaters.sort()
        n = len(heaters)
        
        for x in houses:
            # Binary Search for the first heater >= house (Lower Bound)
            i = bisect.bisect_left(heaters, x)
            
            # Distance to the nearest heater on the left (largest heater <= x)
            # If i == 0, no heater exists on the left
            ld = x - heaters[i - 1] if i > 0 else float('inf')
            
            # Distance to the nearest heater on the right (smallest heater >= x)
            # If i == n, no heater exists on the right
            rd = heaters[i] - x if i < n else float('inf')
            
            # The house only needs to be covered by the CLOSER of the two
            # Update global max radius to ensure this house (and all others) are covered
            ans = max(ans, min(ld, rd))
            
        return ans
```







### [875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

Koko loves to eat bananas. There are `n` piles(痔疮，堆) of bananas, the `ith` pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

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



```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # l: Smallest possible speed (1 banana/hr)
        # r: A "guaranteed feasible" upper bound (sum of all bananas)
        l = 1
        r = sum(piles) # Note: max(piles) is a tighter, more efficient bound
        
        while l <= r:
            m = (l + r) // 2
            
            # check(m): Calculate total hours needed at speed 'm'
            # (x + m - 1) // m is the integer version of math.ceil(x / m)
            hours_needed = sum((x + m - 1) // m for x in piles)
            
            if hours_needed <= h:
                # current speed 'm' is feasible (True), try a smaller speed
                r = m - 1
            else:
                # current speed 'm' is too slow (False), must increase speed
                l = m + 1
                
        # After the loop, l is the first speed that makes check(m) True
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
        # Search Range: 0 to n (Max possible H-index is the number of papers)
        l, r = 0, n

        # Pattern: T T T...F F F (Looking for the LAST True)
        while l <= r:
            h = (l + r) // 2
            
            # check(h): Are there at least 'h' papers with >= 'h' citations?
            # Since sorted, citations[n-h] is the h-th largest value.
            if h == 0 or citations[n - h] >= h:
                # This 'h' works! Try a larger value to the right.
                l = h + 1      
            else:
                # Too many papers requested or citations too low. Search left.
                r = h - 1      

        # Per the "Last True" template, 'r' is the answer after l > r
        return r
```



#### 1) The Strategy: "Find the Largest Valid H"

The H-Index definition states: "A scientist has index $h$ if $h$ of their $n$ papers have **at least** $h$ citations." Since the `citations` array is sorted, the papers with the most citations are at the end of the array.

-   **The Condition**: If we pick a value `h`, the paper at index `n - h` is the "weakest" paper in our set of $h$ papers. If `citations[n - h] >= h`, then all $h$ papers have at least $h$ citations.
-   **Monotonicity**: If a researcher satisfies the condition for $h=5$, they might satisfy it for $h=6$. If they fail for $h=5$, they will definitely fail for $h=6$.
-   **Pattern**: `[T, T, T, T, F, F]` — We want the **Last True**.





### [[LeetCode\] 644. Maximum Average Subarray II ](https://www.cnblogs.com/grandyang/p/8021421.html)

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



```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)

        def check(mid: float) -> bool:
            # Transform: sum(nums[i] - mid) >= 0
            pre = 0.0      # sum(b[0..i])
            pre_k = 0.0    # sum(b[0..i-k])
            min_pre = 0.0  # min(pre[0...i-k])

            # Initial window of size k
            for i in range(k):
                pre += nums[i] - mid
            if pre >= 0: return True

            # Sliding window with variable start
            for i in range(k, n):
                pre += nums[i] - mid
                pre_k += nums[i - k] - mid
                # Greedy: keep track of the smallest prefix sum seen so far
                # that allows for a subarray length >= k
                min_pre = min(min_pre, pre_k)
                
                if pre - min_pre >= 0:
                    return True
            return False

        # Range: Between the smallest and largest possible numbers
        l, r = min(nums), max(nums)
        eps = 1e-5 # Precision threshold

        # Binary search for the maximum feasible average
        while r - l > eps:
            mid = (l + r) / 2
            if check(mid):
                l = mid  # mid is feasible, try to increase it
            else:
                r = mid  # mid is too high, decrease it
        
        return l
```



#### 1) How the `check` Function Works: Prefix Sums & Greedy Strategy

The `check` function validates the condition in **$O(n)$** time using a combination of prefix sums and a greedy sliding window:

-   **Pre-processing**: Subtract `mid` from every element in the array ($nums[i] - mid$). This transforms the problem from finding an average to finding a **subarray sum $\ge 0$**.
-   **Sliding Window**:
    -   Maintain the current prefix sum **`pre`** (representing the sum from $0$ to $i$).
    -   Maintain **`min_pre`** (representing the minimum prefix sum encountered between index $0$ and $i-k$).
-   **Greedy Validation**:
    -   We are looking for a pair $(i, j)$ such that  pre[i] - pre[j] $\ge 0$(where $i - j \ge k$)  .
    -   To maximize this difference, we simply subtract the **smallest prefix sum (`min_pre`)** that occurred at least $k$ positions ago.
    -   If **pre - min_pre $\ge 0$**, we have successfully found a subarray of length $\ge k$ that satisfies the condition.





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
# In binary search for the **maximum** value (`TTTTFFFF`), we move the left boundary (`l = m + 1`) when the condition is met to seek a larger valid answer, ultimately returning **`r`** as the last "True" position.
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





## Binary Search on an Indirect Value

### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

Given an integer array `nums` and an integer `k`, return *the* `kth` *largest element in the array*.

Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Can you solve it without sorting?

**Example 1:**

```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

**Example 2:**

```
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4
```

**Constraints:**

-   `1 <= k <= nums.length <= 105`
-   `-104 <= nums[i] <= 104`



```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        l = min(nums)
        r = max(nums)
        while l <= r:
            m = (l + r) // 2
            cnt = sum(x >= m for x in nums)
            if cnt >= k:
                l = m + 1
            else:
                r = m - 1
        return r
```

```python
// std::numeric_limits<int>::min()
// std::numeric_limits<int>::max()
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        auto [mn, mx] = minmax_element(nums.begin(), nums.end());
        int l = *mn;
        int r = *mx;
        auto checkKLargest = [&](int m) {
            int cnt = 0;
            for (int x : nums) {
                if (x >= m) cnt++;
            }
            return cnt >= k; 
        };
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (checkKLargest(m)) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return r;
    }
};
```





### [[LeetCode\] Search in a Sorted Array of Unknown Size ](https://www.cnblogs.com/grandyang/p/9937770.html)

 

Given an integer array sorted in ascending order, write a function to search `target` in `nums`. If `target` exists, then return its index, otherwise return `-1`. However, the array size is unknown to you. You may only access the array using an `ArrayReader` interface, where `ArrayReader.get(k)` returns the element of the array at index `k` (0-indexed).

You may assume all integers in the array are less than `10000`, and if you access the array out of bounds, `ArrayReader.get` will return `2147483647`.

 

Example 1:

```php
array
target
nums
```

Example 2:

```php
array
target
nums
```

 

Note:

1.  You may assume that all elements in the array are unique.
2.  The value of each element in the array will be in the range `[-9999, 9999]`.



```python
# Exponential expansion is a method used to find a valid search boundary when the size of a sorted array is unknown or unbounded.
def doubling_search(nums, target):
    l = 0
    r = 1
    while True:
        try:
            if nums[r] < target:
                l = r
                r *= 2
            else:
                break
        except IndexError:
            break

    while l <= r:
        m = (l + r) // 2
        try:
            val = nums[m]
        except IndexError:
            r = m - 1
            continue

        if val > target:
            r = m - 1
        elif val == target:
            return m
        else:
            l = m + 1
    return -1


```







## The Kth Smallest/Biggest

Nearly everyone has used the [Multiplication Table](https://en.wikipedia.org/wiki/Multiplication_table). The multiplication table of size `m x n` is an integer matrix `mat` where `mat[i][j] == i * j` (**1-indexed**).

Given three integers `m`, `n`, and `k`, return *the* `kth` *smallest element in the* `m x n` *multiplication table*.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/multtable1-grid)

```
Input: m = 3, n = 3, k = 5
Output: 3
Explanation: The 5th smallest number is 3.
```

**Example 2:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/multtable2-grid)

```
Input: m = 2, n = 3, k = 6
Output: 6
Explanation: The 6th smallest number is 6.
```

 

**Constraints:**

-   `1 <= m, n <= 3 * 104`
-   `1 <= k <= m * n`



````python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        def count(x):
            cnt = 0
            for i in range(1, m + 1):
                cnt += min(x // i, n)
            return cnt

        l = 0
        r = m * n
        while l <= r:
            mid = (l + r) // 2
            if count(mid) >= k:
                r = mid - 1
            else:
                l = mid + 1
        return l
````

```c++
class Solution {
public:
    int findKthNumber(int m, int n, int k) {
        int l = 0;
        int r = m * n;
        
        // Capture external variables by reference using &.
        auto count = [&](int x) {
            int cnt = 0;
            for (int i = 1; i <= m; i++) {
                cnt += min(x / i, n);
            }
            return cnt;
        }; // <--- IMPORTANT: Do not forget the semicolon after the lambda definition!

        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (count(mid) >= k) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }
};
```

