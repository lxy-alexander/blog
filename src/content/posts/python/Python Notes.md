---
title: Python Notes
published: 2026-01-16
description: "Python Notes"
image: ""
tags: ["Python Notes"]
category: python
draft: false
---









## **列表推导式（List Comprehension）**

-   过滤型（filter）

```python
[表达式 for 变量 in 可迭代对象 if 条件]
```

-   替换/映射型（map）

```python
[表达式1 if 条件 else 表达式2 for 变量 in 可迭代对象]
```

1）`... for ... if ...`（过滤）

只保留满足条件的元素：

```python
nums = [1, 2, 3, 4, 5]
res = [x for x in nums if x % 2 == 0]
print(res)  # [2, 4]
```

2）`... if ... else ... for ...`（替换/映射）

每个元素都保留，但满足条件用一个值，否则用另一个值：

```python
nums = [1, 2, 3, 4, 5]
res = [x if x % 2 == 0 else 0 for x in nums]
print(res)  # [0, 2, 0, 4, 0]
```

3）`... for ... if ... else ...`（语法错误）

==这种写法 Python 不允许==：

```python
nums = [1, 2, 3]
res = [x for x in nums if x % 2 == 0 else 0]
# SyntaxError
```





## `__pycache__`

it stores compiled bytecode (`.pyc`) to speed up future imports and program startup.



## Global

`global` 用来声明函数里使用/赋值的是模块级全局变量，而不是创建新的局部变量。
`global` declares that a variable inside a function refers to the module-level global variable, so assignments update the global one instead of creating a local variable.



## Union & Sequence

`Union[A, B]`: **A or B**

`Sequence[T]`: an ordered `T` (like `list[T]`, ↳`tuple[T, ...]`)

“Ordered” means the elements have a **fixed, meaningful position** (first, second, third…), and you can usually access them by **index**.



## list & tuple

`(params,)` is a tuple (immutable)

`[params]` is a list (mutable, also fine)

```python
for sp in params if isinstance(params, list) else (params,):
```





## list & Sequence

`list` is a specific mutable list type; 

`Sequence` is a more general “sequence interface” that guarantees readable/indexable behavior.

```
from typing import Sequence

def f(x: Sequence[int]):
    print(x[0])

f([1, 2, 3])     # OK
f((1, 2, 3))     # OK

```



## Type Checking

\# 判断 prompts 这个变量，是不是 字符串(str) 或者 字典(dict) 之一。

\# 允许参数既可以传字符串，也可以传字典（两种格式都支持）。

prompts = "hello"

print(isinstance(prompts, (str, dict)))  # True



prompts = {"a": 1}

print(isinstance(prompts, (str, dict)))  # True



prompts = ["hello"]

print(isinstance(prompts, (str, dict)))  # False



## Python bisect_left

>   Finding the insertion position In a sorted array and return the leftmost position

```python
from bisect import bisect_left

arr = [1, 3, 3, 5]
print(bisect_left(arr, 3))  # 1  find the first postion that correspoding value greater than or euqal to 3
print(bisect_left(arr, 4))  # 3  
print(bisect_left(arr, 0))  # 0
print(bisect_left(arr, 6))  # 4  insert into the lastmost position
```



## return 0 if ans == float('inf') else ans

```python
if ans == float('inf'):
    return 0
else:
    return ans
```





## CPython

Python is a **language specification**, while CPython is a **concrete implementation** of that specification.

<u>Python defines *what the language is*; CPython defines *how the language runs*.</u>



### What Does CPython Do?

- Compiles source code into bytecode (`.pyc`)
- Executes bytecode using the Python Virtual Machine (PVM)



## GIL

The **GIL (Global Interpreter Lock)** is a mechanism for thread safty. It <u>ensures **only one thread executes Python bytecode at a time**.</u>



## <u>GIL is periodically switched</u>

Periodic GIL switching means that **CPython forces the currently running thread to release the GIL at regular time intervals** Without periodic switching, a CPU-bound thread could **monopolize the GIL**,
causing other threads (including I/O threads) to starve.

<u>if the execution time exceeds `switchinterval`</u>

<u>The current thread releases the GIL at the **next safe point**</u>

<u>Other threads then get a chance to acquire the GIL</u>

```python
import sys
sys.getswitchinterval()
```



## thread.join

`join()` blocks the **calling thread** until the target thread finishes.让**调用它的线程阻塞**，直到目标线程执行完毕。



## Convoy effect

In CPython, the convoy effect <u>may occur when a CPU-bound thread holds the GIL</u>, delaying I/O-bound threads that are ready to run.



## Descriptor

A descriptor is <u>an object that is stored as a **class attribute** and implements one or more of `__get__`, `__set__`, or `__delete__`.</u>When the attribute is accessed, Python invokes these methods instead of returning a value directly. 描述符（Descriptor）是一个对象，只要它作为**类属性**存在，并且实现了 `__get__`、`__set__` 或 `__delete__` 中的任意一个，Python 在访问该属性时就会调用这些方法，而不是直接取值。

```python
class MyDescriptor:
    def __get__(self, instance, owner):
        return "hello from descriptor"

class A:
    x = MyDescriptor()

a = A()
print(a.x)
```

 when Python evaluates `obj.attr`, it first checks whether the class attribute is a descriptor. A descriptor is just an **instance of a normal class**, Descriptors are not based on inheritance, but on **implementing a specific protocol**.

| 协议 / Protocol                | 方法 / Methods                     |
| ------------------------------ | ---------------------------------- |
| 描述符 / Descriptor            | `__get__`, `__set__`, `__delete__` |
| 迭代器 / Iterator              | `__iter__`, `__next__`             |
| 上下文管理器 / Context manager | `__enter__`, `__exit__`            |



## Intance

obj.attr
↓
在类 __dict__ 里找到 attr
↓
检查这个对象有没有 __get__ 方法？
↓
有 → 当描述符用
没有 → 当普通属性



## **`{}`, `[]`, and `()` in Python**

| Symbols | Name            | Common uses                       |
| ------- | --------------- | --------------------------------- |
| `{}`    | curly braces    | `dict`, `set`                     |
| `[]`    | square brackets | `list`, indexing                  |
| `()`    | parentheses     | `tuple`, function calls, grouping |

| Type  | Mutable | Ordered | Literal |
| ----- | ------- | ------- | ------- |
| list  | ✅       | ✅       | `[]`    |
| tuple | ❌       | ✅       | `()`    |
| set   | ✅       | ❌       | `{}`    |
| dict  | ✅       | ✅       | `{}`    |





## Set

`{'a','i','o','e','u'}` is a Python `set`; the semicolon is harmless but unnecessary.

Typically this question means:

```python
# set
vowels = {'a', 'e', 'i', 'o', 'u'}
c in vowels
```

vs

```python
# list
vowels = ['a', 'e', 'i', 'o', 'u']
c in vowels
```

| Type   | Operation `c in container` | Time complexity  |
| ------ | -------------------------- | ---------------- |
| `set`  | hash lookup                | **O(1)** average |
| `list` | linear scan                | **O(n)**         |

`"aeiou"` is fast because it’s tiny, runs in optimized C code, and avoids hashing overhead 





## Defaultdict

defaultdict ` provides default values automatically; `dict` does not.

`defaultdict(int)` comes from Python’s `collections` module. It’s a dictionary that **automatically creates missing keys** with a default value.

In this case, `int()` is the default factory, and `int()` → `0`.

`dict` (normal dictionary)

```python
d = {}
d['a'] += 1   # ❌ KeyError
```

You must initialize first:

```python
if 'a' not in d:
    d['a'] = 0
d['a'] += 1
```

`defaultdict`

```python
from collections import defaultdict

d = defaultdict(int)
d['a'] += 1   # ✅ starts from 0 automatically
```

Key differences

| Feature              | dict     | defaultdict  |
| -------------------- | -------- | ------------ |
| Missing key          | KeyError | Auto-created |
| Needs initialization | Yes      | No           |
| Code length          | Longer   | Shorter      |



## Range & slice

Both **`range`** and **slicing** use **left-closed, right-open intervals**:





## Counter & dict

**Use `Counter` for pure frequency counting.**

**Use `dict` when counts are part of larger logic or need control.**

| Scenario              | Use       |
| --------------------- | --------- |
| Counting frequencies  | `Counter` |
| Sliding window counts | `Counter` |
| Top-K frequent items  | `Counter` |
| Key → last index      | `dict`    |
| Config / strict data  | `dict`    |
| Missing key = bug     | `dict`    |
| Fixed small domain    | array     |





## Set

`vowels = set('aeiou') `

is equevalent to 

`vowels = {'a', 'e', 'i', 'o', 'u'}`

if c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u':  ❌  not good





## common regex patterns



`\d+`  本质上，反斜杠的作用就是“转义”，它会改变后面字符的意义,  反斜杠就是一个切换器，让普通字符变高级，或者让特殊字符变普通

-   **`\d` (Digit):** Matches any single character that is a digit (0 to 9). It is a shorthand for `[0-9]`.
-   **`+` (Plus sign):** A quantifier meaning **"one or more."** It tells the engine to keep matching the preceding element as long as it repeats.

| **Lowercase** | **Matches...**                     | **Uppercase** | **Matches NOT...**                          |
| ------------- | ---------------------------------- | ------------- | ------------------------------------------- |
| **`\d`**      | **D**igits (0-9)                   | **`\D`**      | **NOT** a digit (letters, symbols, etc.)    |
| **`\w`**      | **W**ord chars (a-z, 0-9, `_`)     | **`\W`**      | **NOT** a word char (spaces, `!`, `@`, `#`) |
| **`\s`**      | **S**pace (tabs, newlines, spaces) | **`\S`**      | **NOT** a space (any visible character)     |

Multi-Separator Pattern  

To split a string by `/`, `-`, `,`, `;`, and spaces, use: `[/\-,;\s]+`

-   **`[ ]`**: The "Container." Matches any **one** character inside. (匹配括号内任意**一个**字符)
-   **`\s`**: Matches any **space** (space, tab, newline). (匹配所有**空白符**)
-   **`\-`**: Matches a **dash**. We use a backslash `\` because `-` usually means a range (like a-z). (匹配**横杠**，加反斜杠是为了防止被识别为范围符号)
-   **`+`**: Matches **one or more**. This treats `,,,` or `; ` as a single separator. (匹配**一个或多个**，把连续的多个符号当成一个分隔符)

`r''` 代表 **Raw String（原始字符串）**

```python
result = re.split(r'[/\-,;\s]+', text)
```

1. Common Separators (分隔符)

-   **`-`** : **Hyphen** (or Dash)
-   **`/`** : **Forward Slash** (or just Slash)
-   **`,`** : **Comma**
-   **`;`** : **Semicolon**
-   **` `** : **Space** (or Whitespace)
-   **`_`** : **Underscore**

2. Regex Structural Symbols (正则结构符号)

-   **`r''`** : **Raw String**
-   **`[ ]`** : **Square Brackets** (used for Character Classes)
-   **`[^ ]`** : **Caret** (when inside brackets, it means **Negation**)
-   **`\`** : **Backslash** (used for Escaping转义)
-   **`+`** : **Plus Sign** (a Quantifier meaning "one or more")
-   **`\*`** : **Asterisk** (or Star, meaning "zero or more")
-   **`.`** : **Dot** (or Period)

3. Shorthand Character Classes (速记字符集)

-   **`\d`** : **Digit** (matches numbers)
-   **`\D`** : **Non-digit** (anything but a number)
-   **`\w`** : **Word character** (letters, numbers, underscores)
-   **`\W`** : **Non-word character** (symbols, spaces)
-   **`\s`** : **Whitespace** (spaces, tabs, newlines)
-   **`\S`** : **Non-whitespace** (anything visible)
-   **`\n`** : **New Line**
-   **`\t`** : Tab
-   **`\r`** : Carriage Return
-   **`\'`** : Single quote
-   **`\'`** : Double quote
-    **`\\'`** : Backslash `\`



Bilingual Cheat Sheet

| **符号 (Symbol)** | **英文名 (English Name)** | **正则功能 (Regex Function)** | **是否需要转义? (Need Escape?)** |
| ----------------- | ------------------------- | ----------------------------- | -------------------------------- |
| **`.`**           | **Dot / Period**          | 匹配任意字符 (Wildcard)       | **Yes** (写成 `\.`)              |
| **`+`**           | **Plus**                  | 1个或多个 (1 or more)         | **Yes** (写成 `\+`)              |
| **`*`**           | **Asterisk / Star**       | 0个或多个 (0 or more)         | **Yes** (写成 `\*`)              |
| **`?`**           | **Question Mark**         | 0个或1个 (Optional)           | **Yes** (写成 `\?`)              |
| **`^`**           | **Caret**                 | 匹配开头 (Start of line)      | **Yes** (写成 `\^`)              |
| **`$`**           | **Dollar**                | 匹配结尾 (End of line)        | **Yes** (写成 `\$`)              |
| **`\`**           | **Backslash**             | 转义符 (The Escaper)          | **Yes** (写成 `\\`)              |
| **`[ ]`**         | **Square Brackets**       | 字符集 (Character Class)      | **Yes** (写成 `\[ \]`)           |
| **`{ }`**         | **Curly Braces**          | 计数区间 (Quantifier)         | **Yes** (写成 `\{ \}`)           |
| **`-`**           | **Hyphen / Dash**         | 范围 (Range)                  | **仅在中括号内中间位置**         |



## re.split & re.findall

Examples

```python
# 匹配 小写字母 a–z、数字 0–9、以及字符 -
re.findall(r"[a-z\-0-9]", text) 
re.findall(r"[a-z0-9-]", text)

# ^ as start-of-string anchor (most common)
re.findall(r"^abc", "abc123") # match, ^ means “must start at the beginning of the string”

# ^ inside [] → NEGATION (very important)
print(re.findall(r'[^0-9]', 'a1b2c3')) # means “anything except …” ['a', 'b', 'c']


```

| 项目     | `re.split`     | `re.findall`     |
| -------- | -------------- | ---------------- |
| 核心动作 | **切分字符串** | **提取匹配内容** |
| 关注点   | 分隔符         | 匹配项           |
| 返回内容 | 被切开的部分   | 匹配到的内容     |
|          |                |                  |



## Comparison: `10**18` vs `1e18`

| Expression | Value                     | Type    | Exact?               |
| ---------- | ------------------------- | ------- | -------------------- |
| `10**18`   | 1,000,000,000,000,000,000 | `int`   | ✅ exact              |
| `1e18`     | 1,000,000,000,000,000,000 | `float` | ❌ may lose precision |

-   **C++**

| Type        | Very Large Value | Very Small Value | Recommended                             |
| ----------- | ---------------- | ---------------- | --------------------------------------- |
| `int`       | `INT_MAX`        | `INT_MIN`        | `std::numeric_limits<int>::max()`       |
| `long long` | `LLONG_MAX`      | `LLONG_MIN`      | `std::numeric_limits<long long>::max()` |

| Type     | +Infinity                                 | −Infinity                                  | Max Finite                           | Min Finite                              |
| -------- | ----------------------------------------- | ------------------------------------------ | ------------------------------------ | --------------------------------------- |
| `double` | `std::numeric_limits<double>::infinity()` | `-std::numeric_limits<double>::infinity()` | `std::numeric_limits<double>::max()` | `std::numeric_limits<double>::lowest()` |
| `float`  | `std::numeric_limits<float>::infinity()`  | `-std::numeric_limits<float>::infinity()`  | `std::numeric_limits<float>::max()`  | `std::numeric_limits<float>::lowest()`  |

-   **Python**

| Purpose           | Value                       |
| ----------------- | --------------------------- |
| Very large number | `10**18`, `10**100`, etc.   |
| Very small number | `-10**18`, `-10**100`, etc. |

>   Python `int` has **arbitrary precision**, no overflow.

| Purpose    | Value                     |
| ---------- | ------------------------- |
| +Infinity  | `float('inf')`            |
| −Infinity  | `float('-inf')`           |
| Max finite | `1.7976931348623157e308`  |
| Min finite | `-1.7976931348623157e308` |

Practical Recommendation (Algorithms / Contests)

| Language | Best Practice                        |
| -------- | ------------------------------------ |
| C++      | `std::numeric_limits<T>::infinity()` |
| Python   | `float('inf')`                       |







## int(not x)

is equalvalent to `1 if x == 0 else 0`
