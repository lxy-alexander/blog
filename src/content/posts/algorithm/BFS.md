---
title: "BFS"
published: 2026-06-12
description: "BFS"
image: ""
tags: ["algorithm","BFS"]
category: algorithm
draft: false
lang: ""
createdAt: "2026-06-12T19:25:31.835.002143392Z"
---

# BFS

Breadth-First Search is an algorithm used to traverse or search a graph or grid.

It starts from one or more starting points and explores all nearby nodes first. After finishing the current level, it moves to the next level. 

Because BFS expands level by level, it is useful for finding the shortest path, minimum number of steps, or minimum time in an unweighted graph.



| 类型       | 关键词             | 典型题目                                        |
| ---------- | ------------------ | ----------------------------------------------- |
| 层序遍历   | 一层一层、每层处理 | 102. 二叉树的层序遍历、107. 二叉树的层序遍历 II |
| 最短路径   | 最少步数、最短距离 | 1091. 二进制矩阵中的最短路径、127. 单词接龙     |
| 多源 BFS   | 多个起点同时扩散   | 994. 腐烂的橘子、542. 01 矩阵、1162. 地图分析   |
| 网格扩散   | 向四周扩散、分钟数 | 994. 腐烂的橘子、286. 墙与门                    |
| 状态搜索   | 每个状态当作节点   | 752. 打开转盘锁、773. 滑动谜题                  |
| 拓扑 BFS   | 入度为 0 开始      | 207. 课程表、210. 课程表 II                     |
| 双向 BFS   | 从起点和终点同时搜 | 127. 单词接龙、752. 打开转盘锁                  |
| 图连通遍历 | 图中能到哪些点     | 797. 所有可能的路径、841. 钥匙和房间            |



## Level Order BFS

Process nodes level by level.



### [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

Given the `root` of a binary tree, return *the level order traversal of its nodes' values*. (i.e., from left to right, level by level).

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/tree1)

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
```

**Example 2:**

```
Input: root = [1]
Output: [[1]]
```

**Example 3:**

```
Input: root = []
Output: []
```

 

**Constraints:**

-   The number of nodes in the tree is in the range `[0, 2000]`.
-   `-1000 <= Node.val <= 1000`



```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
            
        q = [root]
        ans = []
        while q:
            tmp = q
            q = []
            level = []
            for node in tmp:
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            ans.append(level)
        return ans

```



### [103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

Given the `root` of a binary tree, return *the zigzag level order traversal of its nodes' values*. (i.e., from left to right, then right to left for the next level and alternate between).

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/tree1-20260613234614109)

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]
```

**Example 2:**

```
Input: root = [1]
Output: [[1]]
```

**Example 3:**

```
Input: root = []
Output: []
```

 

**Constraints:**

-   The number of nodes in the tree is in the range `[0, 2000]`.
-   `-100 <= Node.val <= 100`

 







## Shortest Path BFS

Find the minimum number of steps from start to target.



### [1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

Given an `n x n` binary matrix `grid`, return *the length of the shortest **clear path** in the matrix*. If there is no clear path, return `-1`.

A **clear path** in a binary matrix is a path from the **top-left** cell (i.e., `(0, 0)`) to the **bottom-right** cell (i.e., `(n - 1, n - 1)`) such that:

-   All the visited cells of the path are `0`.
-   All the adjacent cells of the path are **8-directionally** connected (i.e., they are different and they share an edge or a corner).

The **length of a clear path** is the number of visited cells of this path.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/example1_1)

```
Input: grid = [[0,1],[1,0]]
Output: 2
```

**Example 2:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/example2_1)

```
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
```

**Example 3:**

```
Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1
```

 

**Constraints:**

-   `n == grid.length`
-   `n == grid[i].length`
-   `1 <= n <= 100`
-   `grid[i][j] is 0 or 1`

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
            return -1
        q = deque()
        q.append((0, 0, 1))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while q:
            x, y, d = q.popleft()
            if x == n - 1 and y == n - 1:
                return d
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
                    grid[nx][ny] = 1
                    q.append((nx, ny, d + 1)) # Every level has the same distance from the start
        return -1
```



### [127. Word Ladder](https://leetcode.com/problems/word-ladder/)

A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:

-   Every adjacent pair of words differs by a single letter.
-   Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
-   `sk == endWord`

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return *the **number of words** in the **shortest transformation sequence** from* `beginWord` *to* `endWord`*, or* `0` *if no such sequence exists.*

 

**Example 1:**

```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
```

**Example 2:**

```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
```

 

**Constraints:**

-   `1 <= beginWord.length <= 10`
-   `endWord.length == beginWord.length`
-   `1 <= wordList.length <= 5000`
-   `wordList[i].length == beginWord.length`
-   `beginWord`, `endWord`, and `wordList[i]` consist of lowercase English letters.
-   `beginWord != endWord`
-   All the words in `wordList` are **unique**.

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0
        
        q = deque()
        q.append((beginWord, 1))
        while q:
            word, steps = q.popleft()
            if word == endWord:
                return steps
            
            for i in range(len(word)):
                for j in range(26):
                    c = chr(ord('a') + j) # It indicates converting the character 'a' into its corresponding encoded number
                    if c == word[i]:
                        continue
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set:
                        word_set.remove(new_word)
                        q.append((new_word, steps + 1))
        return 0 
```







## Multi-Source BFS

Start from many points at the same time and spread outward.



### [994. Rotting Oranges](https://leetcode.com/problems/rotting-oranges/)

You are given an `m x n` `grid` where each cell can have one of three values:

-   `0` representing an empty cell,
-   `1` representing a fresh orange, or
-   `2` representing a rotten orange.

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return *the minimum number of minutes that must elapse until no cell has a fresh orange*. If *this is impossible, return* `-1`.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/oranges)

```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

**Example 2:**

```
Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
```

**Example 3:**

```
Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
```

 

**Constraints:**

-   `m == grid.length`
-   `n == grid[i].length`
-   `1 <= m, n <= 10`
-   `grid[i][j]` is `0`, `1`, or `2`.

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        q = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    q.append((i, j))
        if fresh == 0:
            return 0
        
        ans = 0
        while q and fresh > 0:
            ans += 1
            tmp = q
            q = []
            for x, y in tmp:
                for i, j in (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1):
                    if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                        fresh -= 1
                        grid[i][j] = 2
                        q.append((i, j))
        
        return -1 if fresh else ans
```







### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/)

Given an `m x n` binary matrix `mat`, return *the distance of the nearest* `0` *for each cell*.

The distance between two cells sharing a common edge is `1`.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/01-1-grid)

```
Input: mat = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[0,0,0],[0,1,0],[0,0,0]]
```

**Example 2:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/01-2-grid)

```
Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
Output: [[0,0,0],[0,1,0],[1,2,1]]
```

 

**Constraints:**

-   `m == mat.length`
-   `n == mat[i].length`
-   `1 <= m, n <= 104`
-   `1 <= m * n <= 104`
-   `mat[i][j]` is either `0` or `1`.
-   There is at least one `0` in `mat`.



```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        ans = [[-1] * n for _ in range(m)]
        q = deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    ans[i][j] = 0
                    q.append((i, j))
        
        while q:
            i, j = q.popleft()

            for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < m and 0 <= y < n and ans[x][y] == -1:
                    ans[x][y] = ans[i][j] + 1
                    q.append((x, y))
        return ans

```







### [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)

Given an `n x n` `grid` containing only values `0` and `1`, where `0` represents water and `1` represents land, find a water cell such that its distance to the nearest land cell is maximized, and return the distance. If no land or water exists in the grid, return `-1`.

The distance used in this problem is the Manhattan distance: the distance between two cells `(x0, y0)` and `(x1, y1)` is `|x0 - x1| + |y0 - y1|`.

 

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/1336_ex1)

```
Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
Explanation: The cell (1, 1) is as far as possible from all the land with distance 2.
```

**Example 2:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/1336_ex2)

```
Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
Output: 4
Explanation: The cell (2, 2) is as far as possible from all the land with distance 4.
```

 

**Constraints:**

-   `n == grid.length`
-   `n == grid[i].length`
-   `1 <= n <= 100`
-   `grid[i][j]` is `0` or `1`

 



```python
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        n = len(grid)
        q = deque()
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 1:
                    q.append((i, j))
        
        if len(q) == n * n:
            return -1

        ans = -1
        while q:
            ans += 1
            for _ in range(len(q)):
                i, j = q.popleft()
                for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                    if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                        grid[x][y] = 1
                        q.append((x, y))

        return ans 
```





## Topological BFS



### [207. Course Schedule](https://leetcode.com/problems/course-schedule/)

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.

-   For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.

 

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
```

**Example 2:**

```
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
```

 

**Constraints:**

-   `1 <= numCourses <= 2000`
-   `0 <= prerequisites.length <= 5000`
-   `prerequisites[i].length == 2`
-   `0 <= ai, bi < numCourses`
-   All the pairs prerequisites[i] are **unique**.



Start with courses that have no prerequisites, take them first, then reduce the prerequisite count of the courses that depend on them, and whenever a course has no prerequisites left, add it to the queue.

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        indgree = [0] * numCourses
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            indgree[course] += 1
        
        q = deque()
        for course in range(numCourses):
            if indgree[course] == 0:
                q.append(course)
        
        finished = 0 
        while q:
            course = q.popleft()

            finished += 1
            for nxt_course in graph[course]:
                indgree[nxt_course] -= 1

                if indgree[nxt_course] == 0:
                    q.append(nxt_course)
        
        return finished == numCourses

```







### [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you **must** take course `bi` first if you want to take course `ai`.

-   For example, the pair `[0, 1]`, indicates that to take course `0` you have to first take course `1`.

Return *the ordering of courses you should take to finish all courses*. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return **an empty array**.

 

**Example 1:**

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
```

**Example 2:**

```
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0.
So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3].
```

**Example 3:**

```
Input: numCourses = 1, prerequisites = []
Output: [0]
```

 

**Constraints:**

-   `1 <= numCourses <= 2000`
-   `0 <= prerequisites.length <= numCourses * (numCourses - 1)`
-   `prerequisites[i].length == 2`
-   `0 <= ai, bi < numCourses`
-   `ai != bi`
-   All the pairs `[ai, bi]` are **distinct**.

Start with courses that have no prerequisites, take them first, then reduce the prerequisite count of the courses that depend on them, and whenever a course has no prerequisites left, add it to the queue.

```python
class Solution:
    def findOrder(self, n: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(n)]
        indgree = [0] * n
        for course, prereq in prerequisites:
            graph[prereq].append(course)
            indgree[course] += 1

        q = deque()
        for course in range(n):
            if indgree[course] == 0:
                q.append(course)
        ans = []
        while q:
            course = q.popleft()
            ans.append(course)
            
            for nxt_course in graph[course]:
                indgree[nxt_course] -= 1

                if indgree[nxt_course] == 0:
                    q.append(nxt_course)
        
        return ans if len(ans) == n else []
```







