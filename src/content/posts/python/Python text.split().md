---
title: "Python text.split()"
published: 2026-03-09
description: "Python text.split()"
image: ""
tags: ["python","Python text.split()"]
category: python
draft: false
lang: ""
---

# **I. `text.split()` in Python**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">split()</code> is a Python <span style="color:#2980B9">string method (字符串方法)</span> that divides a string into a list of substrings. It's one of the most commonly used tools for text processing.
</div>

## 1. Basic Usage

```python
text = "Python is awesome"
result = text.split()
print(result)  # Output: ['Python', 'is', 'awesome']
```

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
By default, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">split()</code> uses <span style="color:#2980B9">whitespace characters (空白字符)</span> as delimiters: spaces, newlines <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">\n</code>, tabs <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">\t</code>, etc.
</div>

## 2. Parameter Details

### 1) `split()` vs `split(' ')` Difference

```python
text = "Python   is   awesome"  # Multiple spaces

# Default split() - handles any amount of whitespace
print(text.split())   # Output: ['Python', 'is', 'awesome']

# split(' ') - strictly splits on single space
print(text.split(' '))  # Output: ['Python', '', '', 'is', '', '', 'awesome']
```

### 2) Specifying Separator `sep`

```python
data = "apple,banana,orange"
print(data.split(','))  # Output: ['apple', 'banana', 'orange']

path = "user/local/bin" # slash/backslash
print(path.split('/'))  # Output: ['user', 'local', 'bin']

sentence = "Hello-World-Python"
print(sentence.split('-'))  # Output: ['Hello', 'World', 'Python']
```

### 3) Limiting Splits with `maxsplit`

The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">*message</code> syntax is used for **Extended Unpacking (扩展解包)**.

-   **`date`, `time`, `level`**: These take the first 3 elements of the list respectively.
-   **`*message`**: This collects **all remaining elements** into a **List (列表)**. `message = parts[3:]  # ['Connection', 'failed'] ← all remaining as list`

```python
text = "one two three four five"

# Split only first 2 times
print(text.split(maxsplit=2))  # Output: ['one', 'two', 'three four five']

# Equivalent syntax
print(text.split(' ', 2))  # Output: ['one', 'two', 'three four five']

# Practical example: parsing simple logs
log = "2024-01-15 10:30:45 ERROR Connection failed"
date, time, level, *message = log.split(maxsplit=3)
print(f"Date: {date}, Time: {time}, Level: {level}, Message: {message}")
# Output: Date: 2024-01-15, Time: 10:30:45, Level: ERROR, Message: ['Connection', 'failed']
```

## 3. Common Use Cases

### 1) Word Frequency Counting

```python
from collections import Counter

text = "Python is awesome. Python is powerful!"
words = text.lower().split()  # Convert to lowercase then split
# Note: Punctuation remains! Output: ['python', 'is', 'awesome.', 'python', 'is', 'powerful!']

# Better approach: clean punctuation
import re
words = re.findall(r'\w+', text.lower())
print(Counter(words))  # Output: Counter({'python': 2, 'is': 2, 'awesome': 1, 'powerful': 1})
```

| Part           | Meaning          | Explanation                                                  |
| :------------- | :--------------- | :----------------------------------------------------------- |
| `re`           | regex module     | Python's <span style="color:#2980B9">regular expression library (正则表达式库)</span> |
| `.findall()`   | find all matches | Returns <span style="color:#E8600A;font-weight:700">all non-overlapping matches</span> as a list |
| `r''`          | raw string       | <span style="color:#2980B9">Raw string (原始字符串)</span> - backslashes are treated literally |
| `\w`           | word character   | Matches <span style="color:#2980B9">letters, digits, and underscore (字母、数字、下划线)</span> |
| `+`            | one or more      | <span style="color:#2980B9">Quantifier (量词)</span> - match 1 or more occurrences |
| `text.lower()` | lowercase        | Converts everything to <span style="color:#2980B9">lowercase (小写)</span> for case-insensitive counting |

### 2) Parsing CSV (Simple Cases)

```python
csv_line = "John,25,Engineer,New York"
name, age, job, city = csv_line.split(',')
print(f"{name} is {age} years old, works as {job} in {city}")
# Output: John is 25 years old, works as Engineer in New York
```

### 3) Handling User Input

```python
# Parsing commands
command = "save document.txt"
action, filename = command.split(maxsplit=1)
print(f"Action: {action}, File: {filename}")  # Output: Action: save, File: document.txt

# Processing multiple inputs
user_input = "5 10 15"
numbers = [int(x) for x in user_input.split()]
print(sum(numbers))  # Output: 30
```

## 4. Important Notes

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">⚠️ Common Pitfalls: </span>

1. **Empty string**:
```python
text = ""
print(text.split())  # Output: [] (empty list)
print(text.split(','))  # Output: [''] (list with one element)
```

2. **Separator not found**:
```python
text = "hello world"
print(text.split(','))  # Output: ['hello world']
```

3. **Consecutive separators**:
```python
text = "a,,b,c"
print(text.split(','))  # Output: ['a', '', 'b', 'c']
```

4. **Return value is always a list**:
```python
text = "python"
result = text.split()
print(type(result))  # Output: <class 'list'>
print(result)  # Output: ['python']
```
</div>

## 5. Method Comparison

| Method         | Purpose               | Example                  | Result              |
| -------------- | --------------------- | ------------------------ | ------------------- |
| `split()`      | Split by whitespace   | `"a b c".split()`        | `['a', 'b', 'c']`   |
| `split(' ')`   | Split by single space | `"a  b".split(' ')`      | `['a', '', 'b']`    |
| `rsplit()`     | Split from right      | `"a-b-c".rsplit('-',1)`  | `['a-b', 'c']`      |
| `splitlines()` | Split by line breaks  | `"a\nb".splitlines()`    | `['a', 'b']`        |
| `partition()`  | Split into 3 parts    | `"a-b-c".partition('-')` | `('a', '-', 'b-c')` |

## 6. Practical Example: Parsing Configuration Files

```python
config = """
host=localhost
port=8080
debug=true
"""

settings = {}
for line in config.strip().split('\n'):
    if '=' in line:
        key, value = line.split('=', 1)
        settings[key] = value

print(settings)
# Output: {'host': 'localhost', 'port': '8080', 'debug': 'true'}
```

## 7. Advanced Techniques

### 1) Using `split()` with List Comprehension

```python
# Extract numbers from mixed string
data = "age:25,score:95,weight:70"
values = [item.split(':')[1] for item in data.split(',')]
print(values)  # Output: ['25', '95', '70']

# Convert to appropriate types
numeric_values = [int(item.split(':')[1]) for item in data.split(',')]
print(numeric_values)  # Output: [25, 95, 70]
```

### 2) Handling Multiple Delimiters

```python
import re

text = "apple;banana,orange|grape"
# Split on ; , or |
fruits = re.split('[;,]', text)  # Simple case
fruits = re.split('[;,\|]', text)  # With escape for |
print(fruits)  # Output: ['apple', 'banana', 'orange', 'grape']
```

### 3) Preserving Delimiters

```python
# Using re.split() with capturing group keeps delimiters
text = "hello-world-python"
parts = re.split('(-)', text)
print(parts)  # Output: ['hello', '-', 'world', '-', 'python']
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">text.split()</code> splits strings into word lists using whitespace; <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">text.split(sep)</code> splits by a specified delimiter; and the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">maxsplit</code> parameter controls how many splits to perform.</div>
