---
title: "time"
published: 2026-05-08
description: "time"
image: ""
tags: ["python","time"]
category: python
draft: false
lang: ""
createdAt: "2026-05-08T21:00:04.357.515585100Z"
---

# Python `time` Module 
*Learning notes for the `time` module (时间模块) including performance measurement, delays, and timestamp handling.*

## 1. Measuring Execution Time (测量执行时间)
Use `time.time()` or `time.perf_counter()` before and after a code block to calculate elapsed time in seconds (秒). `perf_counter()` offers the highest precision (精度) for short intervals.

```python
import time

# Method 1: time.time() — absolute timestamps
start = time.time()
time.sleep(0.5)  # simulate work
end = time.time()
print(f"Elapsed: {end - start:.3f} seconds")  # Output: Elapsed: 0.500 seconds

# Method 2: time.perf_counter() — better for short durations
start = time.perf_counter()
sum(range(1000000))  # actual work
end = time.perf_counter()
print(f"Perf counter: {end - start:.5f} sec")  # Output: Perf counter: 0.01234 sec (varies)
```
<br>

## 2. Adding Delays (添加延迟)
Use `time.sleep(seconds)` to pause execution, which is essential for rate-limiting (速率限制) or simulating real-time behavior (实时行为).

```python
import time

print("Start")        # Output: Start
time.sleep(1.5)       # pause 1.5 seconds
print("After 1.5s")   # Output: After 1.5s

# Example: countdown with delay
for i in range(3, 0, -1):
    print(i)
    time.sleep(0.5)
print("Go!")          
# Output: 
# 3
# 2
# 1
# Go!
```
<br>

## 3. Getting Current Timestamps (获取当前时间戳)
`time.time()` returns seconds since the epoch (Unix纪元，1970-01-01), useful for unique IDs (唯一标识符) or time math.

```python
import time

now = time.time()
print(f"Timestamp: {now}")           # Output: Timestamp: 1734567890.123456
print(f"Seconds since epoch: {int(now)}")  # Output: Seconds since epoch: 1734567890

# Convert to readable format
print(time.ctime(now))                # Output: Wed Dec 18 10:11:30 2024
```
<br>

## 4. Formatting Readable Time (格式化可读时间)
Use `time.strftime(format, time.localtime())` to convert timestamps into human-readable strings (人类可读字符串) with custom formatting directives.

```python
import time

# Get current local time as a formatted string
current = time.localtime()
print(time.strftime("%Y-%m-%d %H:%M:%S", current))  # Output: 2024-12-18 10:11:30

# Common format directives:
# %Y = 4-digit year, %m = month (01-12), %d = day (01-31)
# %H = hour (00-23), %M = minute, %S = second
print(time.strftime("%I:%M %p on %A, %B %d", current))  # Output: 10:11 AM on Wednesday, December 18

# Parse a string back to struct_time (结构化时间)
parsed = time.strptime("2024-12-25 09:30:00", "%Y-%m-%d %H:%M:%S")
print(parsed.tm_year)  # Output: 2024
print(parsed.tm_mon)   # Output: 12
```
<br>

## 5. High-Resolution Timing (高精度计时)
Use `time.perf_counter()` for benchmarking (基准测试) small code snippets, as it includes sleep time and has the highest available resolution.

```python
import time

def benchmark(func, *args, **kwargs):
    """Simple benchmark decorator/function (简单基准测试函数)"""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = (end - start) * 1000  # convert to milliseconds
    print(f"Function '{func.__name__}' took {elapsed:.3f} ms")
    return result

# Example usage
def slow_square(n):
    time.sleep(0.01)
    return n * n

benchmark(slow_square, 5)  # Output: Function 'slow_square' took 10.234 ms
```
<br>

## 6. `time.monotonic()` — Unaffected by System Clock Changes
`time.monotonic()` is a clock that never goes backward (单调时钟), safe for measuring intervals even if the system time is adjusted (e.g., NTP updates).

```python
import time

start = time.monotonic()
time.sleep(0.5)
end = time.monotonic()
print(f"Monotonic elapsed: {end - start:.3f} sec")  # Output: Monotonic elapsed: 0.500 sec

# Unlike time.time(), monotonic() is immune to system clock changes
# (e.g., leap seconds, manual time adjustments)
```
<br>

## 7. Common Time Directives Reference Table (常用时间指令参考表)

| Directive (指令) | Meaning (含义)                        | Example (示例) |
| ---------------- | ------------------------------------- | -------------- |
| `%Y`             | Year with century (四位年份)          | 2024           |
| `%y`             | Year without century (两位年份)       | 24             |
| `%m`             | Month as zero-padded (月份，带前导零) | 12             |
| `%d`             | Day of month (日期)                   | 18             |
| `%H`             | Hour (24-hour, 24小时制)              | 14             |
| `%I`             | Hour (12-hour, 12小时制)              | 02             |
| `%M`             | Minute (分钟)                         | 05             |
| `%S`             | Second (秒)                           | 30             |
| `%p`             | AM or PM (上午/下午)                  | PM             |
| `%A`             | Full weekday name (完整星期名)        | Wednesday      |
| `%a`             | Abbreviated weekday (缩写星期名)      | Wed            |
| `%B`             | Full month name (完整月份名)          | December       |
| `%b`             | Abbreviated month (缩写月份名)        | Dec            |

<br>
<br>
