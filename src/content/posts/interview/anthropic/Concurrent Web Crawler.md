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

# I. Concurrent Web Crawler

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
