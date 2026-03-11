---
title: "Python Decorator"
published: 2026-03-09
description: "Python Decorator"
image: ""
tags: ["python","Python Decorator"]
category: python
draft: false
lang: ""
---

# **I. Decorator Pattern In Python**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Decorator (装饰器)</span> is a design pattern in Python that allows you to <span style="color:#E8600A;font-weight:700">wrap (包裹)</span> a function or class with additional behavior — without modifying its source code. Decorators are built on the concept that <span style="color:#2980B9">functions are first-class objects (函数是一等对象)</span>, meaning they can be passed as arguments, returned from other functions, and assigned to variables. </div>

## 1. Why Do We Need Decorators?

<span style="color:#E8600A">1.</span> **Code Reuse (代码复用):** Apply the same cross-cutting logic (logging, timing, auth) to many functions without copy-pasting.

<span style="color:#E8600A">2.</span> **Separation of Concerns (关注点分离):** Keep business logic clean; push auxiliary logic into decorators.

<span style="color:#E8600A">3.</span> **Readability (可读性):** The `@decorator` syntax makes intent explicit at a glance.

## 2. Prerequisites — Functions as First-Class Objects

Before understanding decorators, you must understand three building blocks.

### 1) Functions Can Be Assigned to Variables

```python
def greet(name):
    return f"Hello, {name}!"

say_hello = greet          # Assign function to a variable (赋值给变量)
print(say_hello("Alice"))  # Output: Hello, Alice!
```

### 2) Functions Can Be Passed as Arguments

```python
def apply(func, value):
    return func(value)     # Call the passed-in function (调用传入的函数)

result = apply(greet, "Bob")
print(result)  # Output: Hello, Bob!
```

### 3) Functions Can Be Returned from Functions (Closures 闭包)

```python
def make_multiplier(factor):
    def multiplier(x):
        return x * factor  # Captures `factor` from enclosing scope (捕获外部变量)
    return multiplier      # Return the inner function (返回内部函数)

double = make_multiplier(2)
print(double(5))   # Output: 10
print(double(9))   # Output: 18
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> A <span style="color:#E8600A;font-weight:700">Closure (闭包)</span> is an inner function that "remembers" variables from its enclosing scope even after the outer function has returned. This is the foundation of every decorator.</div>



<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A decorator is essentially a <span style="color:#E8600A;font-weight:700">Higher-Order Function (高阶函数)</span> — it takes a function as input, wraps it in a new function that adds behavior, and returns the new function. The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">@syntax</code> is just syntactic sugar (语法糖) for <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">func = decorator(func)</code>. </div>

## 1. The Simplest Decorator

### 1) Manual Style (Without @ Syntax)

```python
def my_decorator(func):
    def wrapper():
        print("--- Before function call ---")  # Pre-logic (前置逻辑)
        func()                                  # Call original function (调用原函数)
        print("--- After function call ---")   # Post-logic (后置逻辑)
    return wrapper

def say_hi():
    print("Hi!")

# Manually wrap (手动包裹)
say_hi = my_decorator(say_hi)
say_hi()
```

**Output:**

```
--- Before function call ---
Hi!
--- After function call ---
```

### 2) Using @ Syntax (语法糖)

```python
def my_decorator(func):
    def wrapper():
        print("--- Before function call ---")
        func()
        print("--- After function call ---")
    return wrapper

@my_decorator          # Equivalent to: say_hi = my_decorator(say_hi)
def say_hi():
    print("Hi!")

say_hi()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">@my_decorator</code> placed above a function definition is 100% equivalent to writing <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">say_hi = my_decorator(say_hi)</code> right after the definition.</div>

------

## 2. Decorating Functions with Arguments

The <span style="color:#E8600A;font-weight:700">wrapper</span> must accept and forward any arguments the original function takes, using <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">*args</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">**kwargs</code>.

### 1) Problem Without *args / **kwargs

```python
def my_decorator(func):
    def wrapper():         # ← No parameters! Will crash if func takes arguments.
        func()
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(1, 2)  # ❌ TypeError: wrapper() takes 0 positional arguments but 2 were given
```

### 2) Correct: Use *args and **kwargs

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):           # Accept any arguments (接受任意参数)
        print(f"Calling: {func.__name__}")
        result = func(*args, **kwargs)      # Forward to original function (转发给原函数)
        print(f"Result: {result}")
        return result                       # Don't forget to return! (记得返回结果)
    return wrapper

@my_decorator
def add(a, b):
    return a + b

@my_decorator
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

add(3, 4)
greet("Alice", greeting="Hi")
```

**Output:**

```
Calling: add
Result: 7
Calling: greet
Result: Hi, Alice!
```

------

## 3. Preserving Function Metadata with functools.wraps

### 1) The Problem — Metadata Loss (元数据丢失)

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def add(a, b):
    """Adds two numbers."""  # Docstring (文档字符串)
    return a + b

print(add.__name__)   # Output: wrapper  ← WRONG! Should be "add"
print(add.__doc__)    # Output: None     ← WRONG! Lost the docstring
```

### 2) The Fix — @functools.wraps

```python
import functools

def my_decorator(func):
    @functools.wraps(func)           # Copies metadata from func to wrapper (复制元数据)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def add(a, b):
    """Adds two numbers."""
    return a + b

print(add.__name__)   # Output: add      ✅
print(add.__doc__)    # Output: Adds two numbers.  ✅
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Always use @functools.wraps(func) inside every decorator you write.</span> Without it, debugging tools, documentation generators, and frameworks that inspect <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__name__</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__doc__</code> will silently receive wrong information.</div>

------

## 4. Practical Decorator Examples

### 1) Timing Decorator (计时装饰器)

```python
import functools
import time

def timer(func):
    """Measures execution time (测量执行时间) of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()          # High-resolution timer (高精度计时器)
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    """Sum of 0..n using a loop."""
    return sum(range(n))

slow_sum(10_000_000)
# Output: [TIMER] slow_sum took 0.412381s
```

### 2) Logging Decorator (日志装饰器)

```python
import functools

def logger(func):
    """Logs function calls and their arguments (记录函数调用和参数)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"[LOG] Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} returned {result!r}")
        return result
    return wrapper

@logger
def divide(a, b):
    return a / b

divide(10, 2)
divide(7, b=3)
```

**Output:**

```
[LOG] Calling divide(10, 2)
[LOG] divide returned 5.0
[LOG] Calling divide(7, b=3)
[LOG] divide returned 2.3333333333333335
```

### 3) Retry Decorator (重试装饰器)

```python
import functools
import time

def retry(max_attempts=3, delay=1.0):
    """Retries a function on exception (异常时重试)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[RETRY] Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts.")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_api_call(url):
    import random
    if random.random() < 0.7:   # 70% chance of failure (模拟不稳定网络)
        raise ConnectionError("Network timeout")
    return f"200 OK from {url}"

try:
    print(unstable_api_call("https://example.com"))
except RuntimeError as e:
    print(e)
```

### 4) Cache / Memoization Decorator (缓存装饰器)

```python
import functools

def memoize(func):
    """Caches results of expensive function calls (缓存昂贵函数调用的结果)."""
    cache = {}   # Cache storage (缓存存储)
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)   # Compute and store (计算并存储)
        else:
            print(f"[CACHE] Hit for args={args}")
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))   # Computes fresh
print(fibonacci(10))   # [CACHE] Hit for args=(10,)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Python's standard library provides <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">functools.lru_cache</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">functools.cache</code> (Python 3.9+) as production-ready memoization decorators. Prefer those over hand-rolling your own.</div>

### 5) Access Control Decorator (访问控制装饰器)

```python
import functools

def requires_auth(func):
    """Blocks calls if user is not authenticated (未认证则阻止调用)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        user = kwargs.get("user") or (args[0] if args else None)
        if not getattr(user, "is_authenticated", False):
            raise PermissionError(f"Access denied: authentication required.")
        return func(*args, **kwargs)
    return wrapper

class User:
    def __init__(self, name, authenticated):
        self.name = name
        self.is_authenticated = authenticated

@requires_auth
def view_dashboard(user):
    return f"Welcome to dashboard, {user.name}!"

alice = User("Alice", authenticated=True)
bob   = User("Bob",   authenticated=False)

print(view_dashboard(alice))   # ✅ Welcome to dashboard, Alice!
try:
    print(view_dashboard(bob)) # ❌ PermissionError
except PermissionError as e:
    print(e)
```

------

## 5. Decorators with Parameters (带参数的装饰器)

A decorator with parameters requires an **extra layer of nesting (额外一层嵌套)**: an outer function receives the parameters and returns the actual decorator.

```
call site:     @repeat(3)
what happens:  repeat(3)  →  returns decorator
               decorator(func)  →  returns wrapper
```

### 1) repeat(n) — Run a Function n Times

```python
import functools

def repeat(n):
    """Runs the decorated function n times (运行被装饰函数 n 次)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                result = func(*args, **kwargs)
            return result          # Returns last result (返回最后一次结果)
        return wrapper
    return decorator

@repeat(3)
def say(message):
    print(message)

say("Hello!")
# Output:
# Hello!
# Hello!
# Hello!
```

### 2) validate_types — Runtime Type Checking (运行时类型检查)

```python
import functools

def validate_types(**type_map):
    """Validates argument types at runtime (运行时验证参数类型)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for param_name, expected_type in type_map.items():
                value = bound.arguments.get(param_name)
                if value is not None and not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{param_name}' expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(a=int, b=int)
def add(a, b):
    return a + b

print(add(2, 3))       # ✅ 5
print(add(2.0, 3))     # ❌ TypeError: Argument 'a' expected int, got float
```

------

## 6. Stacking Multiple Decorators (叠加多个装饰器)

You can apply several decorators to one function. They are applied **bottom-up (从下到上)** at definition time, but execute **top-down (从上到下)** at call time.

```python
import functools

def decorator_A(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("A: before")
        result = func(*args, **kwargs)
        print("A: after")
        return result
    return wrapper

def decorator_B(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("B: before")
        result = func(*args, **kwargs)
        print("B: after")
        return result
    return wrapper

@decorator_A         # Applied second (第二个应用) → outermost wrapper
@decorator_B         # Applied first  (第一个应用) → innermost wrapper
def my_func():
    print("  → Running my_func")

my_func()
```

**Output:**

```
A: before
B: before
  → Running my_func
B: after
A: after
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">@A @B def f</code> is equivalent to <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">f = A(B(f))</code>. Think of it as wrapping layers of an onion — <span style="color:#2980B9">B wraps f first</span>, then <span style="color:#2980B9">A wraps the result</span>.</div>

------

## 7. Class-Based Decorators (基于类的装饰器)

A class can act as a decorator by implementing <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">**init**</code> (to receive the function) and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">**call**</code> (to act as the wrapper).

### 1) Call Counter Decorator

```python
import functools

class CountCalls:
    """Counts how many times a function has been called (统计调用次数)."""

    def __init__(self, func):
        functools.update_wrapper(self, func)  # Equivalent to @functools.wraps
        self.func = func
        self.count = 0                        # Instance variable (实例变量)

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"[COUNT] {self.func.__name__} has been called {self.count} time(s)")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()
say_hello()
say_hello()
print(f"Total calls: {say_hello.count}")
```

**Output:**

```
[COUNT] say_hello has been called 1 time(s)
Hello!
[COUNT] say_hello has been called 2 time(s)
Hello!
[COUNT] say_hello has been called 3 time(s)
Hello!
Total calls: 3
```

------

## 8. Decorating Classes (装饰类)

Decorators can also be applied to **entire classes (整个类)**, typically to add or modify class-level behavior.

### 1) Singleton Decorator (单例装饰器)

```python
def singleton(cls):
    """Ensures only one instance of a class is created (确保类只有一个实例)."""
    instances = {}
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self, host):
        self.host = host
        print(f"Creating connection to {host}")

db1 = DatabaseConnection("localhost")   # Creating connection to localhost
db2 = DatabaseConnection("localhost")   # (No output — returns existing instance)
print(db1 is db2)                       # True ✅
```

------

## 9. Built-in Decorators in Python (Python内置装饰器)

| Decorator (装饰器)                                           | Location         | Purpose                                                      |
| ------------------------------------------------------------ | ---------------- | ------------------------------------------------------------ |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@staticmethod</code> | Built-in         | Method that doesn't receive `self` or `cls` (不接收self/cls的方法) |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@classmethod</code> | Built-in         | Method that receives the class as first arg `cls` (接收类作为第一参数) |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@property</code> | Built-in         | Makes a method accessible like an attribute (方法变属性访问) |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@functools.wraps</code> | functools        | Preserves metadata of wrapped function (保留被包裹函数的元数据) |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@functools.lru_cache</code> | functools        | LRU memoization cache (LRU缓存)                              |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@functools.cache</code> | functools (3.9+) | Unbounded memoization cache (无界缓存)                       |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@dataclasses.dataclass</code> | dataclasses      | Auto-generates `__init__`, `__repr__`, etc. (自动生成初始化方法等) |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">@abstractmethod</code> | abc              | Marks a method as abstract (标记方法为抽象方法)              |

### 1) @property Example

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """Getter (读取器)."""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Setter with validation (带验证的写入器)."""
        if value < 0:
            raise ValueError("Radius cannot be negative (半径不能为负)")
        self._radius = value

    @property
    def area(self):
        """Computed property (计算属性) — no setter needed."""
        import math
        return math.pi * self._radius ** 2

c = Circle(5)
print(c.radius)   # 5      — accessed like an attribute, not c.radius()
print(c.area)     # 78.53...
c.radius = 10
print(c.area)     # 314.15...
c.radius = -1     # ❌ ValueError
```

### 2) @functools.lru_cache Example

```python
import functools

@functools.lru_cache(maxsize=128)   # Cache up to 128 results (缓存最多128个结果)
def expensive_query(user_id: int) -> str:
    print(f"  [DB] Querying user {user_id}...")   # Only prints on cache miss (仅缓存未命中时打印)
    return f"User #{user_id} data"

print(expensive_query(1))   # [DB] Querying...   → cache miss
print(expensive_query(1))   # (no DB print)       → cache hit ✅
print(expensive_query(2))   # [DB] Querying...   → cache miss
print(expensive_query.cache_info())
# CacheInfo(hits=1, misses=2, maxsize=128, currsize=2)
```

------

## 10. Common Pitfalls (常见陷阱)

### 1) Forgetting to Return the Result

```python
# ❌ WRONG — silently returns None
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)   # Missing return!
    return wrapper

# ✅ CORRECT
def good_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)  # Always return!
    return wrapper
```

### 2) Mutable Default in Cached Decorator

```python
# ❌ lru_cache requires hashable arguments (需要可哈希参数)
@functools.lru_cache(maxsize=128)
def process(data: list):   # list is unhashable (列表不可哈希)!
    pass
# TypeError: unhashable type: 'list'

# ✅ Use tuple instead (使用元组代替)
@functools.lru_cache(maxsize=128)
def process(data: tuple):
    pass
```

### 3) Decorator Evaluated at Import Time (装饰器在导入时执行)

```python
def register(func):
    print(f"Registering {func.__name__}...")  # Runs at import, not at call time!
    return func

@register
def my_task():
    pass

# Output when module is imported: "Registering my_task..."
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">The decorator body runs once at definition time (import time)</span>, while the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">wrapper</code> body runs every time the decorated function is called. Keep heavy setup logic inside wrapper, not the decorator factory.</div>

------

## 11. Summary Comparison Table

| Feature (特性)        | Function Decorator                   | Class Decorator                       |
| --------------------- | ------------------------------------ | ------------------------------------- |
| Syntax (语法)         | `def deco(func)`                     | `class Deco: __init__ + __call__`     |
| State (状态)          | Via closure variables (通过闭包变量) | Via instance variables (通过实例变量) |
| Readability (可读性)  | ✅ More concise                       | 🟡 More explicit for complex state     |
| Works with methods    | ✅ Yes                                | ⚠️ Needs extra care with `self`        |
| Metadata preservation | `@functools.wraps`                   | `functools.update_wrapper`            |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>A <span style="color:#E8600A;font-weight:700">Decorator (装饰器)</span> is a closure that wraps a function to inject reusable behavior — always use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">@functools.wraps</code>, always <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">return</code> the result, and add a third nesting layer when your decorator needs its own parameters.</div>
