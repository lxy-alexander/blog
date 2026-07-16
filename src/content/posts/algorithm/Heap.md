---
title: "Heap"
published: 2026-06-17
description: "Heap"
image: ""
tags: ["algorithm","Heap"]
category: algorithm
draft: false
lang: ""
createdAt: "2026-06-17T22:39:15.970.422911032Z"
---

# Heap / Priority Queue

Heap is a data structure used to quickly get the minimum or maximum element.

In LeetCode, heap problems usually use a priority queue to repeatedly take the current smallest or largest element.

Heap is useful when the problem asks for top K elements, kth smallest/largest, dynamic median, scheduling, merging sorted lists, or always choosing the best current option.

| Type                   | Keywords                                                  | Typical Problems                                             |
| ---------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| Basic Heap             | min value, max value, priority queue basics               | 703. Kth Largest Element in a Stream, 1046. Last Stone Weight |
| Top K Elements         | top k, kth largest, kth smallest, most frequent           | 215. Kth Largest Element in an Array, 347. Top K Frequent Elements, 973. K Closest Points to Origin, 692. Top K Frequent Words |
| K-way Merge            | merge sorted lists, merge sorted arrays, k sorted         | 23. Merge k Sorted Lists, 373. Find K Pairs with Smallest Sums, 378. Kth Smallest Element in a Sorted Matrix, 632. Smallest Range Covering Elements from K Lists |
| Two Heaps              | median, dynamic median, lower half and upper half         | 295. Find Median from Data Stream, 480. Sliding Window Median |
| Sliding Window Heap    | window maximum, window median, remove expired elements    | 239. Sliding Window Maximum, 480. Sliding Window Median      |
| Greedy + Heap          | always pick best option, maximize profit, minimize cost   | 502. IPO, 630. Course Schedule III, 871. Minimum Number of Refueling Stops, 1642. Furthest Building You Can Reach |
| Scheduling / Intervals | meeting rooms, CPU tasks, available servers, ending time  | 253. Meeting Rooms II, 621. Task Scheduler, 1834. Single-Threaded CPU, 1882. Process Tasks Using Servers, 2402. Meeting Rooms III |
| Frequency + Heap       | frequency count, rearrange by frequency, most common      | 347. Top K Frequent Elements, 451. Sort Characters By Frequency, 767. Reorganize String, 1054. Distant Barcodes |
| Graph Shortest Path    | shortest path, minimum effort, Dijkstra                   | 743. Network Delay Time, 778. Swim in Rising Water, 1631. Path With Minimum Effort, 1514. Path with Maximum Probability |
| Simulation + Heap      | simulate process, next event, smallest available item     | 355. Design Twitter, 1845. Seat Reservation Manager, 1942. The Number of the Smallest Unoccupied Chair, 2336. Smallest Number in Infinite Set |
| Heap + Binary Search   | kth smallest, sorted matrix, pair distance                | 378. Kth Smallest Element in a Sorted Matrix, 668. Kth Smallest Number in Multiplication Table, 719. Find K-th Smallest Pair Distance, 786. K-th Smallest Prime Fraction |
| Lazy Deletion          | delayed removal, invalid heap elements, outdated elements | 239. Sliding Window Maximum, 480. Sliding Window Median      |

## Basic Heap

Use a heap to quickly get the current minimum or maximum element.

This type is mainly used to practice priority queue operations and custom sorting rules.



### [703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

You are part of a university admissions office and need to keep track of the `kth` highest test score from applicants in real-time. This helps to determine cut-off marks for interviews and admissions dynamically as new applicants submit their scores.

You are tasked to implement a class which, for a given integer `k`, maintains a stream of test scores and continuously returns the `k`th highest test score **after** a new score has been submitted. More specifically, we are looking for the `k`th highest score in the sorted list of all scores.

Implement the `KthLargest` class:

-   `KthLargest(int k, int[] nums)` Initializes the object with the integer `k` and the stream of test scores `nums`.
-   `int add(int val)` Adds a new test score `val` to the stream and returns the element representing the `kth` largest element in the pool of test scores so far.

 

**Example 1:**

**Input:**
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]

**Output:** [null, 4, 5, 5, 8, 8]

**Explanation:**

KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3); // return 4
kthLargest.add(5); // return 5
kthLargest.add(10); // return 5
kthLargest.add(9); // return 8
kthLargest.add(4); // return 8

**Example 2:**

**Input:**
["KthLargest", "add", "add", "add", "add"]
[[4, [7, 7, 7, 7, 8, 3]], [2], [10], [9], [9]]

**Output:** [null, 7, 7, 7, 8]

**Explanation:**

KthLargest kthLargest = new KthLargest(4, [7, 7, 7, 7, 8, 3]);
kthLargest.add(2); // return 7
kthLargest.add(10); // return 7
kthLargest.add(9); // return 7
kthLargest.add(9); // return 8

 

**Constraints:**

-   `0 <= nums.length <= 104`
-   `1 <= k <= nums.length + 1`
-   `-104 <= nums[i] <= 104`
-   `-104 <= val <= 104`
-   At most `104` calls will be made to `add`.

```python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = []
        for num in nums:
            self.add(num)
        
        

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)

        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

        return self.heap[0]

       

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```







### [1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)

You are given an array of integers `stones` where `stones[i]` is the weight of the `ith` stone.

We are playing a game with the stones. On each turn, we choose the **heaviest two stones** and smash them together. Suppose the heaviest two stones have weights `x` and `y` with `x <= y`. The result of this smash is:

-   If `x == y`, both stones are destroyed, and
-   If `x != y`, the stone of weight `x` is destroyed, and the stone of weight `y` has new weight `y - x`.

At the end of the game, there is **at most one** stone left.

Return *the weight of the last remaining stone*. If there are no stones left, return `0`.

 

**Example 1:**

```
Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation: 
We combine 7 and 8 to get 1 so the array converts to [2,4,1,1,1] then,
we combine 2 and 4 to get 2 so the array converts to [2,1,1,1] then,
we combine 2 and 1 to get 1 so the array converts to [1,1,1] then,
we combine 1 and 1 to get 0 so the array converts to [1] then that's the value of the last stone.
```

**Example 2:**

```
Input: stones = [1]
Output: 1
```

 

**Constraints:**

-   `1 <= stones.length <= 30`
-   `1 <= stones[i] <= 1000`

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = []

        for stone in stones:
            heapq.heappush(heap, -stone)

        while len(heap) > 2:
            y = -heapq.heappop(heap)
            x = -heapq.heappop(heap)

            if y != x:
                heapq.heappush(heap, -(y - x))
        
        return -heap[0] if heap else 0
```





## Top K Elements

Use a heap to keep only the best `k` elements.

For top K largest, usually use a min-heap of size `k`.

For top K smallest, usually use a max-heap of size `k`.



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
        heap = []

        for num in nums:
            heapq.heappush(heap,num)
            if len(heap) > k:
                heapq.heappop(heap)
            
        return heap[0]
```





## [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

Given an integer array `nums` and an integer `k`, return *the* `k` *most frequent elements*. You may return the answer in **any order**.

 

**Example 1:**

**Input:** nums = [1,1,1,2,2,3], k = 2

**Output:** [1,2]

**Example 2:**

**Input:** nums = [1], k = 1

**Output:** [1]

**Example 3:**

**Input:** nums = [1,2,1,2,1,2,3,1,3,2], k = 2

**Output:** [1,2]

 

**Constraints:**

-   `1 <= nums.length <= 105`
-   `-104 <= nums[i] <= 104`
-   `k` is in the range `[1, the number of unique elements in the array]`.
-   It is **guaranteed** that the answer is **unique**.

 

**Follow up:** Your algorithm's time complexity must be better than `O(n log n)`, where n is the array's size.

 

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = Counter(nums)

        heap = []
        for num, freq in cnt.items():
            heapq.heappush(heap, (freq, num))

            if len(heap) > k:
                heapq.heappop(heap)

        ans = []
        while heap:
            _, num = heapq.heappop(heap)
            ans.append(num)

        return ans[::-1]

        
```





### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

Given an array of `points` where `points[i] = [xi, yi]` represents a point on the **X-Y** plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the **X-Y** plane is the Euclidean distance (i.e., `√(x1 - x2)2 + (y1 - y2)2`).

You may return the answer in **any order**. The answer is **guaranteed** to be **unique** (except for the order that it is in).

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/closestplane1)

```
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
```

**Example 2:**

```
Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.
```

 

**Constraints:**

-   `1 <= k <= points.length <= 104`
-   `-104 <= xi, yi <= 104`

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []

        for i, point in enumerate(points):
            x, y = point
            dist = x * x + y * y
            heapq.heappush(heap, (-dist, point))
            if len(heap) > k:
                heapq.heappop(heap)
            
        ans = []
        while heap:
            _, point = heapq.heappop(heap)
            ans.append(point)

        return ans
```







## K-way Merge

Use a min-heap to always take the smallest current element from multiple sorted lists or arrays.



### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

*Merge all the linked-lists into one sorted linked-list and return it.*

 

**Example 1:**

```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted linked list:
1->1->2->3->4->4->5->6
```

**Example 2:**

```
Input: lists = []
Output: []
```

**Example 3:**

```
Input: lists = [[]]
Output: []
```

 

**Constraints:**

-   `k == lists.length`
-   `0 <= k <= 104`
-   `0 <= lists[i].length <= 500`
-   `-104 <= lists[i][j] <= 104`
-   `lists[i]` is sorted in **ascending order**.
-   The sum of `lists[i].length` will not exceed `104`.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []

        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))

        dummy = ListNode()
        cur = dummy
        
        while heap:
            val, i, node = heapq.heappop(heap)
            cur.next = node
            cur = node
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
            
        return dummy.next

```





### [373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

You are given two integer arrays `nums1` and `nums2` sorted in **non-decreasing order** and an integer `k`.

Define a pair `(u, v)` which consists of one element from the first array and one element from the second array.

Return *the* `k` *pairs* `(u1, v1), (u2, v2), ..., (uk, vk)` *with the smallest sums*.

 

**Example 1:**

```
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]
```

**Example 2:**

```
Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [[1,1],[1,1]]
Explanation: The first 2 pairs are returned from the sequence: [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
```

 

**Constraints:**

-   `1 <= nums1.length, nums2.length <= 105`
-   `-109 <= nums1[i], nums2[i] <= 109`
-   `nums1` and `nums2` both are sorted in **non-decreasing order**.
-   `1 <= k <= 104`
-   `k <= nums1.length * nums2.length`

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        for i in range(min(k, len(nums1))):
            heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
        
        ans = []
        while heap and len(ans) < k:
            _, i, j = heapq.heappop(heap)
            ans.append([nums1[i], nums2[j]])
            if j + 1 < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            
        return ans

```



### [378. Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)



Given an `n x n` `matrix` where each of the rows and columns is sorted in ascending order, return *the* `kth` *smallest element in the matrix*.

Note that it is the `kth` smallest element **in the sorted order**, not the `kth` **distinct** element.

You must find a solution with a memory complexity better than `O(n2)`.

 

**Example 1:**

```
Input: matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
Output: 13
Explanation: The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13
```

**Example 2:**

```
Input: matrix = [[-5]], k = 1
Output: -5![](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/lock-a6627e2c7fa0ce8bc117c109fb4e567d-20260618175832638)
```

 

**Constraints:**

-   `n == matrix.length == matrix[i].length`
-   `1 <= n <= 300`
-   `-109 <= matrix[i][j] <= 109`
-   All the rows and columns of `matrix` are **guaranteed** to be sorted in **non-decreasing order**.
-   `1 <= k <= n2`



```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        heap = []

        for i in range(min(k, m)):
            heapq.heappush(heap, (matrix[i][0], i, 0))

        for _ in range(k - 1):
            _, i, j = heapq.heappop(heap)
            if j + 1 < n:
                heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
        
        return heap[0][0]
            
```







## Two Heaps

Use one max-heap for the smaller half and one min-heap for the larger half.

This is commonly used for median problems.





### [295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

-   For example, for `arr = [2,3,4]`, the median is `3`.
-   For example, for `arr = [2,3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the MedianFinder class:

-   `MedianFinder()` initializes the `MedianFinder` object.
-   `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
-   `double findMedian()` returns the median of all elements so far. Answers within `10-5` of the actual answer will be accepted.

 

**Example 1:**

```
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

 

**Constraints:**

-   `-105 <= num <= 105`
-   There will be at least one element in the data structure before calling `findMedian`.
-   At most `5 * 104` calls will be made to `addNum` and `findMedian`.

 

**Follow up:**

-   If all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?
-   If `99%` of all integer numbers from the stream are in the range `[0, 100]`, how would you optimize your solution?

```python
class MedianFinder:

    def __init__(self):
        self.small = []
        self.large = []
        
    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
        

    def findMedian(self) -> float:
        if not self.small:
            return -1
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

```python
class MedianFinder:
    def __init__(self):
        self.count = [0] * 101     # count[v] = 值 v 出现的次数
        self.total = 0             # 已加入的数的总个数

    def addNum(self, num: int) -> None:
        self.count[num] += 1
        self.total += 1

    def findMedian(self) -> float:
        # 中位数的位置：奇数取第 (total+1)//2 个，偶数取第 total//2 和 total//2+1 个的平均
        # 用「第 k 个数（1-indexed）」来定位
        def kth(k):
            cum = 0
            for v in range(101):
                cum += self.count[v]
                if cum >= k:
                    return v
            return -1

        if self.total % 2 == 1:
            return kth(self.total // 2 + 1)
        left = kth(self.total // 2)
        right = kth(self.total // 2 + 1)
        return (left + right) / 2
```







## Sliding Window Heap

Use a heap to maintain the maximum, minimum, or median value inside a moving window.

Expired elements usually need lazy deletion.

## Greedy + Heap

Use a heap to repeatedly choose the best current option.

Common examples include choosing the most profitable project, the shortest task, the earliest ending interval, or the largest available resource.

## Scheduling / Intervals

Use a heap to track ending times, available rooms, available servers, or task priority.

## Frequency + Heap

Count frequency with a hashmap, then use a heap to process elements by frequency.

## Graph Shortest Path

Use a min-heap to always expand the node with the current shortest distance.

This is used for shortest path problems in weighted graphs.

## Simulation + Heap

Use a heap to simulate events in chronological order or repeatedly choose the next available item.

## Heap + Binary Search

Combine heap logic with binary search or sorted matrix properties.

This type usually appears in kth-smallest style problems.

## Lazy Deletion

Use delayed removal when heap elements become invalid but are not at the top yet.
