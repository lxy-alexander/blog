---
title: "Concurrent Web Crawler"
published: 2026-06-06
description: "Concurrent Web Crawler"
image: ""
tags: ["interview","anthropic","Concurrent Web Crawler"]
category: interview / anthropic
draft: false
lang: ""
createdAt: "2026-06-06T16:08:37.149.211292022Z"
---

## I.[1236. 网络爬虫 🔒](https://leetcode.cn/problems/web-crawler)

给定一个网址 `startUrl` 和一个接口 `HtmlParser` ，请你实现一个网络爬虫，以实现爬取同 `startUrl` 拥有相同 **主机名** 的全部链接。

该爬虫得到的全部网址可以 **任何顺序** 返回结果。

你的网络爬虫应当按照如下模式工作：

-   自页面 `startUrl` 开始爬取
-   调用 `HtmlParser.getUrls(url)` 来获得给定 `url` 网址中的全部链接
-   同一个链接最多只爬取一次
-   只浏览 **域名** 与 `startUrl` **相同** 的链接集合

[![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f75726c686f73746e616d652e706e67)](https://camo.githubusercontent.com/97b6f9e4899f74eed5205b9c1e80b1f613744419f4751af87488f11cfd812b44/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f75726c686f73746e616d652e706e67)

如上所示的一个网址，其域名为 `example.org`。简单起见，你可以假设所有的网址都采用 **http协议** 并没有指定 **端口**。例如，网址 `http://leetcode.com/problems` 和 `http://leetcode.com/contest` 是同一个域名下的，而网址 `http://example.org/test` 和 `http://example.com/abc` 是不在同一域名下的。

`HtmlParser` 接口定义如下： 

```
interface HtmlParser {
  // 返回给定 url 对应的页面中的全部 url 。
  public List<String> getUrls(String url);
}
```

下面是两个实例，用以解释该问题的设计功能，对于自定义测试，你可以使用三个变量 `urls`, `edges` 和 `startUrl`。注意在代码实现中，你只可以访问 `startUrl` ，而 `urls` 和 `edges` 不可以在你的代码中被直接访问。

注意：将尾随斜线“/”的相同网址视为不同的网址。例如，“[http://news.yahoo.com”](http://news.yahoo.xn--com-9o0a/) 和 “http://news.yahoo.com/” 是不同的网址。

 

**示例 1：**

[![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f73616d706c655f325f313439372e706e67)](https://camo.githubusercontent.com/3e699f1fe714c825a2c4d40264a8d08bf882ab4d39dd11a59d5fdc9a072f9ed9/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f73616d706c655f325f313439372e706e67)

```
输入：
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com",
  "http://news.yahoo.com/us"
]
edges = [[2,0],[2,1],[3,2],[3,1],[0,4]]
startUrl = "http://news.yahoo.com/news/topics/"
输出：[
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.yahoo.com/us"
]
```

**示例 2：**

**[![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f73616d706c655f335f313439372e706e67)](https://camo.githubusercontent.com/8cacc45a25eeb77777790f5d2ea2722320ee7cfe30d4a2b2742008ad4b6c3083/68747470733a2f2f666173746c792e6a7364656c6976722e6e65742f67682f646f6f63732f6c656574636f6465406d61696e2f736f6c7574696f6e2f313230302d313239392f313233362e576562253230437261776c65722f696d616765732f73616d706c655f335f313439372e706e67)**

```
输入：
urls = [
  "http://news.yahoo.com",
  "http://news.yahoo.com/news",
  "http://news.yahoo.com/news/topics/",
  "http://news.google.com"
]
edges = [[0,2],[2,1],[3,2],[3,1],[3,0]]
startUrl = "http://news.google.com"
输出：["http://news.google.com"]
解释：startUrl 链接到所有其他不共享相同主机名的页面。
```

 

**提示：**

-   `1 <= urls.length <= 1000`
-   `1 <= urls[i].length <= 300`
-   `startUrl` 为 `urls` 中的一个。
-   主机名的长为1到63个字符（包括点），只能包含从‘a’到‘z’的ASCII字母、‘0’到‘9’的数字以及连字符即减号（‘-’）。
-   主机名不会以连字符即减号（‘-’）开头或结尾。
-   关于域名有效性的约束可参考: https://en.wikipedia.org/wiki/Hostname#Restrictions_on_valid_hostnames
-   你可以假定url库中不包含重复项。

```python
class HtmlParser:
    def __init__(self, urls, edges):
        self.graph = defaultdict(list)
        for i, j in edges:
            self.graph[urls[i]].append(urls[j])
    
    def getUrls(self, url): # instance method
        return self.graph[url]
        
from collections import deque, defaultdict
        
class Solution:
	def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> list[str]:
        def host(url):
            return url.split('/')[2]
        
        start_host = host(startUrl)
        q = deque([startUrl])
        visted = {startUrl} # 非空set可以使用{startUrl}
        while q:
            url = q.popleft()
            for nxt in htmlParser.getUrls(url):
                if nxt not in visted and start_host == host(nxt):
                    q.append(nxt)
                    visted.add(nxt)
        return list(visted)
              
```







## Concurrent Web Crawler

*(This question is a variation of the LeetCode question [1242. Web Crawler Multithreaded](https://leetcode.com/problems/web-crawler-multithreaded/description/). If you haven't completed that question yet, it is recommended to solve it first.)*

Given a URL `startUrl` and an interface `htmlParser`, implement a concurrent web crawler to discover and return all **unique** URLs that have the **exact same hostname** as `startUrl`.

The crawler must follow the rules below:

-   **Starting Point:** Begin crawling from `startUrl`.
-   **Same Hostname:** Only record and further crawl URLs whose hostname is **exactly identical** to the hostname of `startUrl`.

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/tpnnebg42m0vxlwemsx3)

All URLs use the `http` protocol without a port number. The hostname is defined as the substring between `"://"` and the next `'/'` (or the end of the string if no further `'/'` exists).

-   **URL Uniqueness:** Before checking uniqueness, **remove the fragment part** (everything after and including `'#'`). Two URLs are considered the same if they match after this sanitization step.
    -   For example: `"http://example.com/page#section1"` and `"http://example.com/page"` should both be treated as `"http://example.com/page"` for uniqueness checking and for subsequent crawling.
-   **No Duplicate Visits:** A sanitized URL must **not** be crawled or added more than once.
-   **Fragment Sanitization Order:** Fragment removal occurs **before** hostname comparison and before deduplication.
-   **Graph Characteristics:** The hyperlink graph may contain cycles, and pages may reference previously visited pages.

You are provided with the implementation of `HtmlParser`:

```java
/*
 * Provided Html Parser implementation. You should NOT modify it.
 */
class HtmlParser {

  // Returns all raw URLs from the webpage of the given URL.
  List<String> getUrls(String url){...} 
}
```

Each call to `getUrls` is subject to a certain latency to simulate the real-world network conditions, so your solution must use **concurrency** to fetch multiple pages in parallel.

For testing purposes, you will be given three variables `urls`, `edges` and `startUrl` to describe the underlying hyperlink graph. Only `startUrl` is accessible in your code; `urls` and `edges` are **not** directly available.

Return all discovered URLs (after sanitization) in any order.

**Constraints:**

-   All URLs use the `http` protocol and do not contain a port number.
-   1 ≤`urls.length`≤ 1000,
-   1 ≤ `edges.length` ≤ 1000
-   Each `getUrls` call will return in ≤ 15 ms.

**Example 1:**

![img](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/l0vqimxxtu4xbwzknsla)

>   **Input:** urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3#sectionA", "http://example.net/page4#"], edges = [[0, 1], [0, 2], [1, 3], [2, 0]], startUrl = "http://example.com/page1"
>
>   **Output:** ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]
>
>   **Explanation:**
>
>   -   All three reachable pages share the hostname "example.com".
>   -   The URL "http://example.com/page3#sectionA" is sanitized to "http://example.com/page3".
>   -   The page "http://example.net/page4#" is ignored due to a different hostname.

**Example 2:**

>   **Input:** urls = ["http://news.yahoo.com/home", "http://news.google.com/top", "http://news.yahoo.com/news"], edges = [[1, 0], [0, 2]], startUrl = "http://news.google.com/top"
>
>   **Output:** ["http://news.google.com/top"]

**Example 3:**

>   **Input:** urls = ["http://site.com/a", "http://site.com/b#frag1", "http://site.com/b#frag2", "http://site.com/c", "http://other.com/x", "http://site.com/d", "http://site.com/e#", "http://site.com/f"], edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [5, 0], [5, 6], [6, 7], [7, 0]], startUrl = "http://site.com/a"
>
>   **Output:** ["http://site.com/a", "http://site.com/b", "http://site.com/c", "http://site.com/d", "http://site.com/e", "http://site.com/f"]

```python
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from collections import defaultdict

"""
Provided Html Parser implementation. You should NOT modify it.
"""
class HtmlParser:
    def __init__(self, urls, edges):
        self.graph = {}
        for u in urls:
            self.graph[u] = []
        
        for edge in edges:
            from_idx = edge[0]
            to_idx = edge[1]
            self.graph[urls[from_idx]].append(urls[to_idx])

    def getUrls(self, url):
        try:
            time.sleep(0.01)  # Simulate network latency
        except:
            pass
        
        links = self.graph.get(url)
        if links is None:
            return []
        return links

class Solution:
    def crawl(self, startUrl, htmlParser):
        def normlize_url(url): # Hash# The fragment identifier is used to locate a specific section of a page.
            return url.split('#')[0]

        def get_host(url):
            return url.split('/')[2]

        start_url = normlize_url(startUrl)
        start_host = get_host(startUrl)
        q = [start_url]
        visited = {start_url} # use set to check if the url is visited
        ans = [start_url]
        while q:
            tmp = q
            q = []
            for url in tmp:
                for nxt in htmlParser.getUrls(url):
                    normlized_url = normlize_url(nxt)
                    if normlized_url not in visited and start_host == get_host(normlized_url):
                        visited.add(normlized_url)
                        q.append(nxt)
                        ans.append(normlized_url)
        return ans


def main():
    test1()
    test2()
    test3()

def test1():
    print("===== Test 1 =====")

    urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3#sectionA",
            "http://example.net/page4#"]
    edges = [[0, 1], [0, 2], [1, 3], [2, 0]]
    startUrl = "http://example.com/page1"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]

def test2():
    print("===== Test 2 =====")

    urls = ["http://news.yahoo.com/home", "http://news.google.com/top", "http://news.yahoo.com/news"]
    edges = [[1, 0], [0, 2]]
    startUrl = "http://news.google.com/top"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://news.google.com/top"]

def test3():
    print("===== Test 3 =====")

    urls = ["http://site.com/a", "http://site.com/b#frag1", "http://site.com/b#frag2", 
    "http://site.com/c", "http://other.com/x", "http://site.com/d", "http://site.com/e#", 
    "http://site.com/f"]
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [5, 0], [5, 6], [6, 7], [7, 0]]
    startUrl = "http://site.com/a"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://site.com/a", "http://site.com/b", "http://site.com/c",
    # "http://site.com/d", "http://site.com/e", "http://site.com/f"]

if __name__ == "__main__":
    main()
```



```
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from collections import defaultdict

"""
Provided Html Parser implementation. You should NOT modify it.
"""
class HtmlParser:
    def __init__(self, urls, edges):
        self.graph = {}
        for u in urls:
            self.graph[u] = []
        
        for edge in edges:
            from_idx = edge[0]
            to_idx = edge[1]
            self.graph[urls[from_idx]].append(urls[to_idx])

    def getUrls(self, url):
        try:
            time.sleep(0.01)  # Simulate network latency
        except:
            pass
        
        links = self.graph.get(url)
        if links is None:
            return []
        return links

class Solution:
    def crawl(self, startUrl, htmlParser):
        def normlize_url(url): # Hash# The fragment identifier is used to locate a specific section of a page.
            return url.split('#')[0]

        def get_host(url):
            return url.split('/')[2]

        start_url = normlize_url(startUrl)
        start_host = get_host(startUrl)
        q = [start_url]
        visited = {start_url} # use set to check if the url is visited
        ans = [start_url]

        with ThreadPoolExecutor(max_workers=16) as executor:
            while q:
                futures = [executor.submit(htmlParser.getUrls, url) for url in q]
                q = []
                for future in as_completed(futures):
                    for url in future.result():
                        normlized_url = normlize_url(url)
                        if normlized_url not in visited and start_host == get_host(normlized_url):
                            visited.add(normlized_url)
                            q.append(url)
                            ans.append(normlized_url)
        return ans


def main():
    test1()
    test2()
    test3()

def test1():
    print("===== Test 1 =====")

    urls = ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3#sectionA",
            "http://example.net/page4#"]
    edges = [[0, 1], [0, 2], [1, 3], [2, 0]]
    startUrl = "http://example.com/page1"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://example.com/page1", "http://example.com/page2", "http://example.com/page3"]

def test2():
    print("===== Test 2 =====")

    urls = ["http://news.yahoo.com/home", "http://news.google.com/top", "http://news.yahoo.com/news"]
    edges = [[1, 0], [0, 2]]
    startUrl = "http://news.google.com/top"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://news.google.com/top"]

def test3():
    print("===== Test 3 =====")

    urls = ["http://site.com/a", "http://site.com/b#frag1", "http://site.com/b#frag2", 
    "http://site.com/c", "http://other.com/x", "http://site.com/d", "http://site.com/e#", 
    "http://site.com/f"]
    edges = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [5, 0], [5, 6], [6, 7], [7, 0]]
    startUrl = "http://site.com/a"

    parser = HtmlParser(urls, edges)
    solution = Solution()
    result = solution.crawl(startUrl, parser)
    print(result)
    # Expected: ["http://site.com/a", "http://site.com/b", "http://site.com/c",
    # "http://site.com/d", "http://site.com/e", "http://site.com/f"]

if __name__ == "__main__":
    main()
```









# II. Concurrent Web Crawler

## 1. Problem Statement

### 1) Problem Description

You are given a starting URL called `startUrl` and an interface called `HtmlParser`.

The `HtmlParser` provides the following method:

```python
class HtmlParser:
    def getUrls(self, url: str) -> List[str]:
        pass
```

You need to implement:

```python
def crawl(startUrl: str, htmlParser: HtmlParser) -> List[str]:
    pass
```

The crawler should return all URLs that are reachable from `startUrl`.

However, there is one important rule:

```text
Only crawl URLs that have the same hostname as startUrl.
```

The order of the returned URLs does not matter.

------

### 2) Constraints

The crawler must satisfy the following requirements:

1.  Start crawling from `startUrl`.
2.  Use `htmlParser.getUrls(url)` to get all links from the current page.
3.  Do not visit the same URL more than once.
4.  Only crawl URLs with the same hostname as `startUrl`.
5.  Assume all URLs use `http`.
6.  Assume URLs do not contain port numbers.

------

### 3) Example

Input:

```text
startUrl = "http://news.yahoo.com"
```

Example graph:

```text
http://news.yahoo.com
    -> http://news.yahoo.com/news
    -> http://news.yahoo.com/topics
    -> http://google.com
```

Output:

```text
[
    "http://news.yahoo.com",
    "http://news.yahoo.com/news",
    "http://news.yahoo.com/topics"
]
```

Explanation:

```text
http://google.com
```

is ignored because its hostname is different from:

```text
news.yahoo.com
```

------

### 4) Things to Clarify

Before coding, clarify the following with the interviewer:

1.  URL Fragments

For example:

```text
http://example.com/page#section1
http://example.com/page#section2
```

Should these be treated as the same page or different pages?

Usually, they should be treated as the same page because the fragment only points to a section inside the same document.

1.  URL Normalization

Ask whether URL normalization is required.

For example:

```text
http://example.com/a
http://example.com/a/
```

or:

```text
http://example.com/page?b=2&a=1
http://example.com/page?a=1&b=2
```

For the basic coding problem, URL normalization is usually not required unless explicitly stated.

------

## 2. Core Idea

### 1) Graph Traversal

The web can be modeled as a directed graph.

Each page is a node.

Each hyperlink is a directed edge.

Therefore, the problem becomes:

```text
Traverse all reachable nodes under the same hostname.
```

------

### 2) BFS with Queue

For the single-threaded version, we can use BFS.

Core data structures:

```text
Queue
Visited Set
```

The queue stores URLs waiting to be crawled.

The visited set avoid duplicate visits.

------

### 3) Hostname Filtering

Before crawling a new URL, check whether it has the same hostname as `startUrl`.

Example:

```text
startUrl = "http://news.yahoo.com"
```

The hostname is:

```text
news.yahoo.com
```

Only URLs with this hostname should be crawled.

------

### 4) Multithreading Follow-up

The follow-up usually asks how to parallelize the crawler.

Web crawling is mostly I/O-bound because most time is spent waiting for network responses.

Therefore, using multiple threads can improve performance.

The key challenges are:

1.  Avoid visiting the same URL twice.
2.  Protect shared data structures.
3.  Limit the number of concurrent workers.
4.  Wait until all crawling tasks are finished.

------

## 3. Algorithm Steps

### 1) Single-threaded BFS

1.  Extract the hostname from `startUrl`.
2.  Initialize:

```python
queue = deque([startUrl])
visited = set([startUrl])
```

1.  While the queue is not empty:

```text
Pop one URL from the queue.
Call htmlParser.getUrls(url).
For each next URL:
    Check whether it has the same hostname.
    Check whether it has not been visited.
    Add it to visited.
    Add it to the queue.
```

1.  Return all URLs in `visited`.

------

### 2) Multithreaded BFS

1.  Extract the hostname from `startUrl`.
2.  Create a thread-safe visited set.
3.  Create a thread pool.
4.  Submit `startUrl` as the first task.
5.  Each worker crawls one URL:

```text
Call htmlParser.getUrls(url).
For each discovered URL:
    Check hostname.
    Lock visited.
    If the URL has not been visited:
        Add it to visited.
        Submit it as a new task.
```

1.  Continue until all tasks are completed.
2.  Return all URLs in `visited`.

------

## 4. Code Implementation

### 1) Python Single-threaded BFS

```python
from collections import deque
from typing import List
from urllib.parse import urlparse


class Solution:
    def crawl(self, startUrl: str, htmlParser: "HtmlParser") -> List[str]:
        start_host = urlparse(startUrl).hostname

        queue = deque([startUrl])
        visited = set([startUrl])

        while queue:
            current_url = queue.popleft()

            for next_url in htmlParser.getUrls(current_url):
                next_host = urlparse(next_url).hostname

                if next_host != start_host:
                    continue

                if next_url in visited:
                    continue

                visited.add(next_url)
                queue.append(next_url)

        return list(visited)
```

------

### 2) Python Multithreaded Version

```python
from queue import Queue
from threading import Thread, Lock
from typing import List
from urllib.parse import urlparse


class Solution:
    def crawl(self, startUrl: str, htmlParser: "HtmlParser") -> List[str]:
        start_host = urlparse(startUrl).hostname
        visited = {startUrl}
        visited_lock = Lock()
        num_workers = 16

        q = Queue()
        q.put(startUrl)

        def worker():
            while True:
                url = q.get()
                if url is None:
                    break                      # sentinel: added after q.join(), not counted
                for next_url in htmlParser.getUrls(url):   # network I/O outside the lock
                    if urlparse(next_url).hostname != start_host:
                        continue
                    with visited_lock:
                        if next_url not in visited:
                            visited.add(next_url)
                            q.put(next_url)
                q.task_done()

        threads = [Thread(target=worker) for _ in range(num_workers)]
        for t in threads:
            t.start()

        q.join()                               # wait until all real URLs are processed
        for _ in range(num_workers):
            q.put(None)                        # then send sentinels to stop workers
        for t in threads:
            t.join()

        return list(visited)
```





```python
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock
from typing import List
from urllib.parse import urlparse


class Solution:
    def crawl(self, startUrl: str, htmlParser: "HtmlParser") -> List[str]:
        start_host = urlparse(startUrl).hostname

        visited = set([startUrl])
        visited_lock = Lock()
        
        max_workers = 16

        def crawl_url(url: str) -> List[str]:
            new_urls = []

            for next_url in htmlParser.getUrls(url):
                next_host = urlparse(next_url).hostname

                if next_host != start_host:
                    continue

                with visited_lock:
                    if next_url in visited:
                        continue

                    visited.add(next_url)
                    new_urls.append(next_url)

            return new_urls

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            futures.add(executor.submit(crawl_url, startUrl))

            while futures:
                done, futures = wait(
                    futures,
                    return_when=FIRST_COMPLETED
                )

                for future in done:
                    for next_url in future.result():
                        futures.add(
                            executor.submit(crawl_url, next_url)
                        )

        return list(visited)
```

------

### 3) Python Version with URL Fragment Normalization

If the interviewer says that URL fragments should be ignored, we can normalize URLs by removing the `#fragment`.

```python
from collections import deque
from typing import List
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    normalized = parsed._replace(fragment="")

    return urlunparse(normalized)


class Solution:
    def crawl(self, startUrl: str, htmlParser: "HtmlParser") -> List[str]:
        startUrl = normalize_url(startUrl)
        start_host = urlparse(startUrl).hostname

        queue = deque([startUrl])
        visited = set([startUrl])

        while queue:
            current_url = queue.popleft()

            for next_url in htmlParser.getUrls(current_url):
                next_url = normalize_url(next_url)
                next_host = urlparse(next_url).hostname

                if next_host != start_host:
                    continue

                if next_url in visited:
                    continue

                visited.add(next_url)
                queue.append(next_url)

        return list(visited)
```

------

## 5. Complexity Analysis

### 1) Time Complexity

Let:

```text
N = number of reachable same-host URLs
E = number of links among these pages
```

For the single-threaded version:

```text
O(N + E)
```

Each URL is visited once.

Each link is checked once.

------

### 2) Space Complexity

The visited set stores all reachable URLs:

```text
O(N)
```

The queue can also store up to `O(N)` URLs.

Therefore, the total space complexity is:

```text
O(N)
```

------

### 3) Multithreaded Runtime

If there are `T` worker threads, the ideal runtime can be approximated as:

```text
O((N + E) / T)
```

However, real performance depends on:

1.  Network latency
2.  Server response time
3.  Thread pool size
4.  Lock contention
5.  Rate limiting
6.  URL frontier size

------

## 6. System Design Discussion

### 1) Why Use Threads?

A web crawler is usually I/O-bound.

Most time is spent waiting for remote servers to respond.

Threads are a good fit because they allow multiple URLs to be fetched concurrently.

Processes are usually less suitable here because they are heavier and do not share memory easily.

For CPU-heavy tasks, processes may be better.

For web crawling, threads or async I/O are usually preferred.

------

### 2) Why Use a Thread Pool?

A bad design is:

```text
Create one new thread for every URL.
```

This can easily crash the system when the number of URLs becomes large.

A better design is to use a fixed-size thread pool.

Benefits:

1.  Controls concurrency.
2.  Avoids too many threads.
3.  Reduces thread creation overhead.
4.  Provides backpressure through a task queue.

A common interview setup is:

```text
10 to 20 worker threads
```

------

### 3) Thread Safety

In the multithreaded version, multiple workers may discover the same URL at the same time.

This code is not safe:

```python
if next_url not in visited:
    visited.add(next_url)
```

Two threads may both pass the check before either one inserts the URL.

The correct approach is to protect the check-and-insert operation with a lock:

```python
with visited_lock:
    if next_url not in visited:
        visited.add(next_url)
```

This makes the operation atomic.

------

### 4) Async I/O vs Thread Pool

Both can be used for web crawling.

Thread pool is better when:

1.  `htmlParser.getUrls(url)` is blocking.
2.  The interviewer expects a standard multithreading solution.
3.  The implementation should be simple.

Async I/O is better when:

1.  The HTTP client is async-native.
2.  The crawler needs very high concurrency.
3.  The system needs lower memory overhead.

For this problem, because `htmlParser.getUrls(url)` is usually provided as a blocking API, a thread pool is the most practical choice.

------

### 5) Scaling to Multiple Machines

For billions of URLs, one machine is not enough.

A distributed crawler usually contains:

```text
URL Frontier
    |
    v
Distributed Workers
    |
    v
Fetcher
    |
    v
Parser
    |
    v
Storage / Indexer
```

Important components:

1.  URL Frontier

Stores URLs waiting to be crawled.

Possible systems:

```text
Kafka
RabbitMQ
SQS
Redis Stream
```

1.  Distributed Visited Set

Prevents duplicate crawling across machines.

Possible systems:

```text
Redis
Cassandra
DynamoDB
Bloom Filter
```

1.  Worker Pool

Many machines fetch and parse pages in parallel.

1.  Storage System

Stores raw HTML, parsed content, metadata, and content fingerprints.

------

### 6) URL Partitioning

A common strategy is to partition URLs by hostname.

Example:

```python
worker_id = hash(hostname) % number_of_workers
```

Benefits:

1.  Easier per-host rate limiting.
2.  Better cache locality.
3.  Easier `robots.txt` enforcement.
4.  Fewer duplicate checks across workers.

For dynamic scaling, use consistent hashing.

------

### 7) Politeness and Rate Limiting

A production crawler must avoid overwhelming websites.

Important techniques:

1.  Respect `robots.txt`.
2.  Apply per-host rate limiting.

Example:

```text
Only send 1 request per second to the same hostname.
```

1.  Use adaptive throttling.

If the server becomes slow or returns many errors, reduce the crawling speed.

1.  Coordinate rate limits globally in a distributed crawler.

------

### 8) Duplicate Detection

There are two types of duplicates.

#### URL-level duplicates

The same URL is discovered multiple times.

Solution:

```text
Visited Set
```

#### Content-level duplicates

Different URLs may return the same content.

Examples:

```text
http://example.com/page?a=1
http://example.com/page?b=2
```

Solutions:

1.  URL normalization
2.  Content hashing

Examples:

```text
MD5
SHA-256
```

1.  Near-duplicate detection

Examples:

```text
SimHash
Jaccard Similarity
MinHash
```

------

### 9) Fault Tolerance

A production crawler must handle worker failures.

Common solutions:

1.  Store pending URLs in a durable queue.
2.  Use acknowledgements.
3.  Requeue a URL if a worker crashes before acknowledgement.
4.  Store crawl state outside worker memory.
5.  Use retries with exponential backoff.

------

### 10) Common Follow-up Questions

1.  How do you make `visited` thread-safe?
2.  Why is a thread pool better than creating a new thread for each URL?
3.  How do you avoid crawling external domains?
4.  How do you handle duplicate pages?
5.  How do you scale the crawler to billions of URLs?
6.  How do you implement per-domain rate limiting?
7.  How do you handle failed requests?
8.  How do you respect `robots.txt`?
9.  How do you choose between async I/O and threads?
10.  How do you partition URLs across machines?
