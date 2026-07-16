---
title: "DFS"
published: 2026-06-11
description: "DFS"
image: ""
tags: ["algorithm","DFS"]
category: algorithm
draft: false
lang: ""
createdAt: "2026-06-11T19:22:35.502.913978570Z"
---

# DFS

Depth-First Search is an algorithm used to traverse or search a graph, tree, or grid.

It starts from one node and keeps going as deep as possible along one path.When it cannot continue, it backtracks and tries another path.

DFS is suitable for problems where you need to explore one path deeply, traverse connected areas, check reachability, or try all possible choices.



## [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)

Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return *the number of islands*.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

**Example 1:**

```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**

```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

 

**Constraints:**

-   `m == grid.length`
-   `n == grid[i].length`
-   `1 <= m, n <= 300`
-   `grid[i][j]` is `'0'` or `'1'`.



```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '2'
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    ans += 1
                    dfs(i, j)
        return ans
```





## [695. Max Area of Island](https://leetcode.com/problems/max-area-of-island/)

You are given an `m x n` binary matrix `grid`. An island is a group of `1`'s (representing land) connected **4-directionally** (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The **area** of an island is the number of cells with a value `1` in the island.

Return *the maximum **area** of an island in* `grid`. If there is no island, return `0`.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/maxarea1-grid)

```
Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
```

**Example 2:**

```
Input: grid = [[0,0,0,0,0,0,0,0]]
Output: 0
```



```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        def dfs(i, j):
            if not(0 <= i < m and 0 <= j < n) or grid[i][j] != 1:
                return 0
            grid[i][j] = 2
            area = 1
            for x, y in directions:
                area += dfs(i + x, j + y)
            return area

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    ans = max(ans, dfs(i, j))
        return ans
```







## Water area size

You have an integer matrix land used to represent a piece of land, and the value of each point in this matrix represents the altitude of the corresponding location. If the value is 0, it indicates water area. A pond is a water area connected vertically, horizontally or diagonally. The size of a pond refers to the number of connected water areas. Write a method to calculate the sizes of all the ponds in the matrix, and the return values need to be sorted in ascending order.

Example

Input:

[

,2,1,0 [0],

,1,0,1 [0],

,1,0,1 [1],

,1,0,1 [0]

]

Output: [1,2,4]

Hint

0 < len(land) <= 1000

0 < len(land[i]) <= 1000



```python
class Solution:
    def pondSizes(self, land: List[List[int]]) -> List[int]:
        ans = []
        m, n = len(land), len(land[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (1, 1), (-1, 1), (1, -1)]

        def dfs(i, j):
            if not (0 <= i < m and 0 <= j < n) or land[i][j] != 0:
                return 0
            
            land[i][j] = -1
            area = 1
            for x, y in directions:
                area += dfs(i + x, j + y)
            return area

        for i in range(m):
            for j in range(n):
                if land[i][j] == 0:
                    ans.append(dfs(i, j))
        ans.sort()
        return ans

```









## [463. Island Perimeter](https://leetcode.com/problems/island-perimeter/)

You are given `row x col` `grid` representing a map where `grid[i][j] = 1` represents land and `grid[i][j] = 0` represents water.

Grid cells are connected **horizontally/vertically** (not diagonally). The `grid` is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/island)

```
Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
Output: 16
Explanation: The perimeter is the 16 yellow stripes in the image above.
```

**Example 2:**

```
Input: grid = [[1]]
Output: 4
```

**Example 3:**

```
Input: grid = [[1,0]]
Output: 4
```

 

**Constraints:**

-   `row == grid.length`
-   `col == grid[i].length`
-   `1 <= row, col <= 100`
-   `grid[i][j]` is `0` or `1`.
-   There is exactly one island in `grid`.



 **the four directions: up, down, left, and right**.

```py
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    continue
                if i == 0 or grid[i - 1][j] == 0:
                    ans += 1
                if i == m - 1 or grid[i + 1][j] == 0:
                    ans += 1
                if j == 0 or grid[i][j - 1] == 0:
                    ans += 1
                if j == n - 1 or grid[i][j + 1] == 0:
                    ans += 1
        return ans 
```

