---
title: "WebCrawler"
published: 2026-03-09
description: "WebCrawler"
image: ""
tags: ["interview","anthropic","WebCrawler"]
category: interview / anthropic
draft: false
lang: ""
---

# I.Problem Description

Design a multi-threaded web crawler application that starts from a **Seed URL** and collects all reachable URLs within the same website.

## 1. Requirements

Provide a function

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">crawl(url: str) -> List[str]</code>

that:

-   Fetches page contents from the seed URL
-   Extracts all hyperlinks found in the page
-   Recursively crawls discovered links

The implementation must ensure **URL De-duplication**, meaning the same URL should never be crawled more than once.

For improved performance, the crawler should utilize **Multithreading (多线程)** or **Asynchronous IO (异步IO)**.

## 2. Considerations

-   URLs may contain fragments such as `#section`
-   These fragments should be ignored during **URL De-duplication**
-   Only unique normalized URLs should be returned

------

## 3. Test Case

### 1) Input

```
http://example.com
```



### 2) Output

A list containing all **unique URLs** reachable from the seed URL.

Example output (order not guaranteed):

```
[
  "http://example.com",
  "http://example.com/about",
  "http://example.com/contact"
]
```



# **II. Implementation**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Multi-threaded Web Crawler (多线程网页爬虫)</span> starts from a <span style="color:#E8600A;font-weight:700">Seed URL (种子URL)</span> and collects all reachable links within the same website. It combines <span style="color:#2980B9">HTTP Requests (HTTP请求)</span>, <span style="color:#2980B9">HTML Parsing (HTML解析)</span>, and <span style="color:#2980B9">Thread-safe Data Structures (线程安全数据结构)</span> to crawl pages concurrently while guaranteeing <span style="color:#E8600A;font-weight:700">URL De-duplication (URL去重)</span>. The required function signature is <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">crawl(url: str) -> List[str]</code>. </div>

The crawler must:

<span style="color:#E8600A">1.</span> Fetch page contents from a URL using **HTTP Requests (HTTP请求)** 

<span style="color:#E8600A">2.</span> Extract links from HTML using an **HTML Parser (HTML解析器)** 

<span style="color:#E8600A">3.</span> Avoid crawling the same URL more than once using **URL De-duplication (URL去重)** 

<span style="color:#E8600A">4.</span> Use **Multithreading (多线程)** or **Asynchronous IO (异步IO)** for better performance 

<span style="color:#E8600A">5.</span> Handle URL fragments such as `#section` using **URL Normalization (URL标准化)**

------

## 1. Prerequisite Knowledge

### 1) HTTP Request (HTTP请求)

A <span style="color:#E8600A;font-weight:700">Web Request (网页请求)</span> retrieves webpage content from a server. The most popular Python library is <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">requests</code> — a high-level <span style="color:#2980B9">HTTP Client Library (HTTP客户端库)</span>.

```python
import requests

response = requests.get("http://example.com")
print(response.text)
```

Key concepts: <span style="color:#E8600A;font-weight:700">HTTP Response (HTTP响应)</span> · <span style="color:#E8600A;font-weight:700">Status Code (状态码)</span> · <span style="color:#E8600A;font-weight:700">HTML Content (HTML内容)</span>

------

### 2) HTML Parsing (HTML解析)

Web pages are written in <span style="color:#E8600A;font-weight:700">HTML (超文本标记语言)</span>. To extract links we must parse the <span style="color:#E8600A;font-weight:700">HTML Document (HTML文档)</span> into a <span style="color:#2980B9">DOM Tree (文档对象模型树)</span>. The standard tool is <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BeautifulSoup</code>.

```python
from bs4 import BeautifulSoup

html = "<a href='http://example.com'>Example</a>"
soup = BeautifulSoup(html, "html.parser")

for link in soup.find_all("a"):
    print(link.get("href"))
    
Example:

# <a href="http://example.com">Visit Example</a>
# <a> → HTML Tag (锚点标签) used to create links, standing for Anchor
# href → Hypertext Reference (超文本引用) specifying the target URL
# Visit Example → Anchor Text (锚文本) shown to the user
```

------

### 3) Multithreading (多线程)

A <span style="color:#E8600A;font-weight:700">Thread (线程)</span> is a smallest unit of execution inside a Process. Multiple threads within the same process can <span style="color:#E8600A;font-weight:700">run tasks **concurrently (并发执行)**</span>.

Using <span style="color:#E8600A;font-weight:700">Multithreading (多线程)</span> lets the crawler download multiple pages <span style="color:#2980B9">concurrently (并发地)</span>, greatly reducing total wait time. Python module: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading</code>.

```python
import threading

def task():
    print("Running task")

t = threading.Thread(target=task) # creates a Thread Object
t.start() # begins thread execution
t.join() # waits for the thread to finish
```

Key concepts: <span style="color:#E8600A;font-weight:700">Concurrency (并发)</span> · <span style="color:#E8600A;font-weight:700">Thread Synchronization (线程同步)</span>

------

### 4) Thread-safe Data Structures (线程安全数据结构)

<span style="color:#C0392B;font-weight:600">Warning: when multiple threads share data without synchronization, Race Conditions (竞态条件) cause data corruption.</span>The `queue` module internally uses **Locks (线程锁)** and **Condition Variables (条件变量)** to guarantee **Thread Safety (线程安全)**.

Python's built-in <span style="color:#E8600A;font-weight:700">Thread-safe Queue (线程安全队列)</span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">queue.Queue</code> handles all locking internally:

```python
from queue import Queue

q = Queue()
q.put("task1")
print(q.get())   # → "task1"
```

------

### 5) URL Normalization (URL标准化)

Some URLs contain <span style="color:#E8600A;font-weight:700">Fragment Identifiers (片段标识符)</span> such as:

```
http://example.com/page#section1
http://example.com/page#section2
```

<span style="color:#C0392B;font-weight:600">Pitfall: fragments point to different positions on the SAME page — treating them as distinct URLs causes duplicate crawling.</span>

<span style="color:#2980B9">Solution:</span> strip the fragment with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">urllib.parse.urldefrag()</code> before de-duplication.

------

## 2. System Design

### 1) URL Queue (URL队列)

A <span style="color:#E8600A;font-weight:700">Task Queue (任务队列)</span> stores URLs waiting to be crawled. We use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">queue.Queue</code> for <span style="color:#2980B9">thread-safe</span> producer-consumer coordination.

------

### 2) Visited URL Set (已访问URL集合)

A <span style="color:#E8600A;font-weight:700">Hash Set (哈希集合)</span> — Python's <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set()</code> — stores visited URLs for O(1) lookup. <span style="color:#C0392B;font-weight:600">Must be protected by a Lock (互斥锁) since `set` is NOT thread-safe.</span>

------

### 3) Worker Threads (工作线程)

Each <span style="color:#E8600A;font-weight:700">Worker Thread (工作线程)</span> runs an infinite loop:

<span style="color:#E8600A">1.</span> Fetch a URL from the queue <span style="color:#E8600A">2.</span> Download HTML and extract all `<a href>` links <span style="color:#E8600A">3.</span> Normalize each URL (strip fragment) <span style="color:#E8600A">4.</span> Add unseen, same-domain URLs back into the queue

------

### 4) Domain Restriction (域名限制)

To prevent leaving the target website, only URLs whose <span style="color:#E8600A;font-weight:700">netloc (网络位置)</span> matches the original <span style="color:#E8600A;font-weight:700">Domain (域名)</span> are enqueued.

```
http://example.com/about  ✅ same domain
http://other-site.com/    ❌ different domain — skip
```

------

## 3. Algorithm Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Initialize                                             │
│  ┌──────────────┐    put seed URL                      │
│  │  URL Queue   │◀────────────────── Seed URL          │
│  └──────┬───────┘                                      │
│         │  get()                                       │
│         ▼                                              │
│  ┌──────────────┐   HTTP GET    ┌──────────────────┐   │
│  │ Worker Thread│──────────────▶│  Web Server      │   │
│  └──────┬───────┘   HTML resp  └──────────────────┘   │
│         │                                              │
│         │ BeautifulSoup → extract <a href>             │
│         │ urldefrag()   → normalize URL                │
│         │                                              │
│         ▼                                              │
│  ┌──────────────┐  not in set?   ┌────────────────┐   │
│  │ Visited Set  │───────────────▶│  URL Queue     │   │
│  │  (set + Lock)│                └────────────────┘   │
│  └──────────────┘                                      │
│                                                        │
│  Repeat until queue.join() unblocks                    │
└─────────────────────────────────────────────────────────┘
```

------

## 4. Python Implementation

### 1) Complete Code

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse
from queue import Queue
import threading


class WebCrawler:
    """
    Multi-threaded Web Crawler (多线程网页爬虫)
    """

    def __init__(self, seed_url: str, max_threads: int = 5):
        """
        seed_url    (种子URL)   : starting webpage
        max_threads (最大线程数): number of worker threads
        """
        self.seed_url = seed_url

        # Extract Domain Name (域名) for same-domain filtering
        self.base_domain = urlparse(seed_url).netloc

        # Thread-safe URL Queue (线程安全URL队列)
        self.url_queue = Queue()

        # Visited Set for URL De-duplication (URL去重集合)
        self.visited = set()

        # Lock for Thread Synchronization (线程同步锁)
        self.lock = threading.Lock()

        self.max_threads = max_threads

        # Seed initialization
        self.url_queue.put(seed_url)
        self.visited.add(seed_url)

    # ──────────────────────────────────────────────
    def fetch_links(self, url: str) -> list:
        """
        Send an HTTP Request (HTTP请求) to `url`,
        parse HTML, and return all normalized absolute links.
        """
        links = []
        try:
            # ① HTTP GET with timeout
            response = requests.get(url, timeout=5)

            # ② Check Status Code (状态码)
            if response.status_code != 200:
                return links

            # ③ HTML Parsing (HTML解析) via BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup.find_all("a", href=True):
                href = tag.get("href")

                # Convert Relative URL (相对URL) → Absolute URL (绝对URL)
                absolute_url = urljoin(url, href)

                # URL Normalization (URL标准化): strip Fragment Identifier (片段标识符)
                normalized_url, _ = urldefrag(absolute_url)

                links.append(normalized_url)

        except Exception:
            pass   # silently ignore network/parse errors

        return links

    # ──────────────────────────────────────────────
    def worker(self) -> None:
        """
        Worker Thread (工作线程): consume URLs from queue until timeout.
        """
        while True:
            try:
                url = self.url_queue.get(timeout=3)
            except Exception:
                return   # queue empty → exit thread

            links = self.fetch_links(url)

            for link in links:
                # Domain Restriction (域名限制): skip external links
                if urlparse(link).netloc != self.base_domain:
                    continue

                # Thread-safe URL De-duplication (URL去重)
                with self.lock:
                    if link not in self.visited:
                        self.visited.add(link)
                        self.url_queue.put(link)

            self.url_queue.task_done()

    # ──────────────────────────────────────────────
    def crawl(self) -> list:
        """
        Launch worker threads and block until all URLs are processed.
        Returns a list of all discovered URLs.
        """
        threads = []

        for _ in range(self.max_threads):
            t = threading.Thread(target=self.worker, daemon=True)
            t.start()
            threads.append(t)

        # Block until every task_done() is called
        self.url_queue.join()

        return list(self.visited)


# ──── Required API (题目要求接口) ─────────────────
def crawl(url: str) -> list:
    return WebCrawler(url).crawl()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">daemon=True</code> on threads ensures they are killed automatically if the main thread exits, preventing the program from hanging. Without this, a stuck network request could block the entire process.</div>

------

## 5. Complexity Analysis

### 1) Time Complexity (时间复杂度)

Let <span style="color:#E8600A;font-weight:700">N</span> = number of pages, <span style="color:#E8600A;font-weight:700">E</span> = total number of links.

$$ \text{Time Complexity} = O(N + E) $$

<span style="color:#2980B9">Reasoning:</span> each page is visited **exactly once** (de-duplication guarantee), and each link is extracted and checked **exactly once**.

------

### 2) Space Complexity (空间复杂度)

Memory usage comes from the <span style="color:#E8600A;font-weight:700">Visited URL Set (已访问URL集合)</span> and the <span style="color:#E8600A;font-weight:700">URL Queue (URL队列)</span>, both bounded by N.

$$ \text{Space Complexity} = O(N) $$

------

### 3) Comparison Table (方案对比)

| Dimension         | Single-threaded (单线程) | Multi-threaded (多线程)                                      | Async IO (异步IO)                                            |
| ----------------- | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Concurrency model | Sequential               | Thread pool                                                  | Event loop                                                   |
| CPU overhead      | Low                      | Medium (context switch)                                      | <span style="color:#E8600A;font-weight:700">Very low</span>  |
| Complexity        | Simple                   | Medium                                                       | High                                                         |
| Best for          | Small sites              | Medium sites                                                 | <span style="color:#E8600A;font-weight:700">Large-scale crawls</span> |
| Python GIL impact | N/A                      | <span style="color:#C0392B;font-weight:600">Limits true parallelism</span> | Not affected                                                 |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Python's <span style="color:#C0392B;font-weight:600">GIL (Global Interpreter Lock, 全局解释器锁)</span> prevents true CPU parallelism in threads. For a web crawler this is fine — threads spend most time <em>waiting for network IO</em>, not executing Python bytecode, so the GIL is released during IO and multithreading still delivers significant speedup.</div>

------

## 6. Example Execution

### 1) Example Input

```python
urls = crawl("http://example.com")
print(urls)
```

### 2) Possible Output

```python
[
    "http://example.com",
    "http://example.com/about",
    "http://example.com/contact",
    "http://example.com/blog"
]
```

------

## 7. Possible Improvements

### 1) Async IO (异步IO)

Replace <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading</code> with <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">asyncio</code> + <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">aiohttp</code> for lower memory usage and better scalability at high concurrency.

```python
import asyncio, aiohttp

async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()
```

------

### 2) Rate Limiting (请求速率限制)

<span style="color:#C0392B;font-weight:600">Warning: sending too many requests too fast can trigger IP bans or cause Server Overload (服务器过载).</span>

```python
import time
time.sleep(0.1)   # polite crawl delay between requests
```

------

### 3) Distributed Crawling (分布式爬虫)

For large-scale production crawlers, replace the in-process queue with distributed components:

| Component                                                    | Role                                  |
| ------------------------------------------------------------ | ------------------------------------- |
| <span style="color:#E8600A;font-weight:700">Kafka (消息队列系统)</span> | Distributed URL queue                 |
| <span style="color:#E8600A;font-weight:700">Redis (内存数据库)</span> | Shared visited-URL set across workers |
| <span style="color:#E8600A;font-weight:700">Celery (分布式任务队列)</span> | Task scheduling & worker management   |

------

## 8. Key Takeaways

| #                                    | Concept                                   | Role in Crawler                     |
| ------------------------------------ | ----------------------------------------- | ----------------------------------- |
| <span style="color:#E8600A">1</span> | **HTTP Requests (HTTP请求)**              | Fetch raw HTML                      |
| <span style="color:#E8600A">2</span> | **HTML Parsing (HTML解析)**               | Extract `<a href>` links            |
| <span style="color:#E8600A">3</span> | **Multithreading (多线程)**               | Concurrent page downloads           |
| <span style="color:#E8600A">4</span> | **Thread-safe Structures (线程安全结构)** | `Queue` + `Lock` prevent data races |
| <span style="color:#E8600A">5</span> | **URL Normalization (URL标准化)**         | Strip fragments, avoid duplicates   |
| <span style="color:#E8600A">6</span> | **Domain Restriction (域名限制)**         | Keep crawler within target site     |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> A multi-threaded web crawler is a producer-consumer pipeline: worker threads <span style="color:#E8600A;font-weight:700">consume</span> URLs from a thread-safe queue, fetch and parse HTML, then <span style="color:#E8600A;font-weight:700">produce</span> new URLs back — with a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set + Lock</code> guaranteeing every URL is visited <strong>exactly once</strong>.</div>
