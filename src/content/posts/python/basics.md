---
title: "basics"
published: 2026-04-27
description: "basics"
image: ""
tags: ["python","basics"]
category: python
draft: false
lang: ""
---

# Syntax

Syntax (语法) defines the valid structure of Python Code (代码), and interviewers expect you to write readable code with correct Indentation (缩进).

## 1. Indentation

Indentation (缩进) defines a Code Block (代码块), so Python uses whitespace instead of braces.

```python
if True:
    print("Inside block")

print("Outside block")

# Output:
# Inside block
# Outside block
```

## 2. Comments

Comment (注释) explains Code (代码) for humans and is ignored by the Interpreter (解释器).

```python
# This is a comment
print("Hello, Python")

# Output:
# Hello, Python
```

## 3. Statement

Statement (语句) is a complete instruction that the Interpreter (解释器) can execute.

```python
x = 10
y = 20
print(x + y)

# Output:
# 30
```

## 4. Expression

Expression (表达式) is code that evaluates to a Value (值).

```python
result = 10 + 5 * 2
print(result)

# Output:
# 20
```

## 5. Case Sensitivity

Case Sensitivity (大小写敏感) means `name` and `Name` are different Identifiers (标识符).

```python
name = "Alice"
Name = "Bob"

print(name)
print(Name)

# Output:
# Alice
# Bob
```

<br><br>

# Variables

Variable (变量) is a name that references an Object (对象), not a fixed memory container.

## 1. Assignment

Assignment (赋值) binds a Variable Name (变量名) to a Value (值).

```python
language = "Python"
version = 3

print(language)
print(version)

# Output:
# Python
# 3
```

## 2. Dynamic Typing

Dynamic Typing (动态类型) means a Variable (变量) can reference different Types (类型) at runtime.

```python
value = 100
print(value, type(value))

value = "Python"
print(value, type(value))

# Output:
# 100 <class 'int'>
# Python <class 'str'>
```

## 3. Multiple Assignment

Multiple Assignment (多重赋值) assigns multiple Values (值) to multiple Variables (变量) in one statement.

```python
x, y, z = 1, 2, 3

print(x)
print(y)
print(z)

# Output:
# 1
# 2
# 3
```

## 4. Naming Convention

Naming Convention (命名规范) improves readability, and Python commonly uses Snake Case (蛇形命名法).

```python
user_name = "Alice"
total_score = 95

print(user_name)
print(total_score)

# Output:
# Alice
# 95
```

## 5. Constant

Constant (常量) is usually written in uppercase by convention, but Python does not enforce immutability.

```python
PI = 3.14159
MAX_RETRY = 3

print(PI)
print(MAX_RETRY)

# Output:
# 3.14159
# 3
```

# Operators

Operator (运算符) performs an operation on Operand (操作数), such as calculation, comparison, or logic.

## 1. Arithmetic Operators

Arithmetic Operator (算术运算符) performs mathematical operations on Numeric Values (数值).

```python
a = 10
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b)
print(a % b)
print(a ** b)

# Output:
# 13
# 7
# 30
# 3.3333333333333335
# 3
# 1
# 1000
```

## 2. Comparison Operators

Comparison Operator (比较运算符) compares two Values (值) and returns a Boolean (布尔值).

```python
a = 10
b = 3

print(a == b)
print(a != b)
print(a > b)
print(a < b)
print(a >= b)
print(a <= b)

# Output:
# False
# True
# True
# False
# True
# False
```

## 3. Logical Operators

Logical Operator (逻辑运算符) combines Boolean Expressions (布尔表达式).

```python
age = 20
has_id = True

print(age >= 18 and has_id)
print(age < 18 or has_id)
print(not has_id)

# Output:
# True
# True
# False
```

## 4. Assignment Operators

Assignment Operator (赋值运算符) updates a Variable (变量) with a concise syntax.

```python
count = 10

count += 5
print(count)

count *= 2
print(count)

# Output:
# 15
# 30
```

## 5. Membership Operators

Membership Operator (成员运算符) checks whether a Value (值) exists in a Container (容器).

```python
numbers = [1, 2, 3]

print(2 in numbers)
print(5 not in numbers)

# Output:
# True
# True
```

## 6. Identity Operators

Identity Operator (身份运算符) checks whether two Variables (变量) reference the same Object (对象).

```python
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a is b)
print(a is c)
print(a == c)

# Output:
# True
# False
# True
```

# Control Flow

Control Flow (控制流) controls the execution order of Statements (语句).

## 1. If Statement

If Statement (条件语句) executes a Code Block (代码块) only when a Condition (条件) is true.

```python
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
else:
    print("C")

# Output:
# B
```

## 2. For Loop

For Loop (for循环) iterates over an Iterable (可迭代对象).

```python
for number in [1, 2, 3]:
    print(number)

# Output:
# 1
# 2
# 3
```

## 3. While Loop

While Loop (while循环) repeats a Code Block (代码块) while a Condition (条件) remains true.

```python
count = 3

while count > 0:
    print(count)
    count -= 1

# Output:
# 3
# 2
# 1
```

## 4. Break

Break Statement (终止语句) exits the nearest Loop (循环) immediately.

```python
for number in range(5):
    if number == 3:
        break
    print(number)

# Output:
# 0
# 1
# 2
```

## 5. Continue

Continue Statement (跳过语句) skips the current Iteration (迭代) and continues the Loop (循环).

```python
for number in range(5):
    if number == 2:
        continue
    print(number)

# Output:
# 0
# 1
# 3
# 4
```

## 6. Match Statement

Match Statement (模式匹配语句) selects a branch based on Pattern Matching (模式匹配).

```python
command = "start"

match command:
    case "start":
        print("Starting")
    case "stop":
        print("Stopping")
    case _:
        print("Unknown command")

# Output:
# Starting
```

# Comprehensions

Comprehension (推导式) is a concise way to create a Collection (集合类型) from an Iterable (可迭代对象).

## 1. List Comprehension

List Comprehension (列表推导式) creates a List (列表) using a compact expression.

```python
squares = [x * x for x in range(5)]
print(squares)

# Output:
# [0, 1, 4, 9, 16]
```

## 2. List Comprehension With Condition

Condition (条件) inside a List Comprehension (列表推导式) filters Elements (元素).

```python
even_numbers = [x for x in range(10) if x % 2 == 0]
print(even_numbers)

# Output:
# [0, 2, 4, 6, 8]
```

## 3. Dictionary Comprehension

Dictionary Comprehension (字典推导式) creates a Dictionary (字典) from Key-Value Pairs (键值对).

```python
squares = {x: x * x for x in range(5)}
print(squares)

# Output:
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

## 4. Set Comprehension

Set Comprehension (集合推导式) creates a Set (集合) with unique Elements (元素).

```python
unique_lengths = {len(word) for word in ["hi", "cat", "dog", "hello"]}
print(unique_lengths)

# Output:
# {2, 3, 5}
```

## 5. Generator Expression

Generator Expression (生成器表达式) creates a lazy Iterator (迭代器) to save Memory (内存).

```python
numbers = (x * x for x in range(5))

for number in numbers:
    print(number)

# Output:
# 0
# 1
# 4
# 9
# 16
```



# Common Builtins

Builtin Function (内置函数) is available without Import (导入), and strong Python interviews expect fluent use of them.

## 1. print

`print()` (打印函数) outputs Values (值) to the Console (控制台).

```python
print("Hello")
print(1, 2, 3)

# Output:
# Hello
# 1 2 3
```

## 2. len

`len()` (长度函数) returns the number of Elements (元素) in a Container (容器).

```python
text = "Python"
numbers = [1, 2, 3]

print(len(text))
print(len(numbers))

# Output:
# 6
# 3
```

## 3. type

`type()` (类型函数) returns the Type (类型) of an Object (对象).

```python
value = 123

print(type(value))
print(type("Python"))

# Output:
# <class 'int'>
# <class 'str'>
```

## 4. range

`range()` (范围函数) creates an Integer Sequence (整数序列), commonly used in a Loop (循环).

```python
for number in range(5):
    print(number)

# Output:
# 0
# 1
# 2
# 3
# 4
```

## 5. enumerate

`enumerate()` (枚举函数) returns Index (索引) and Value (值) while iterating.

```python
names = ["Alice", "Bob", "Charlie"]

for index, name in enumerate(names):
    print(index, name)

# Output:
# 0 Alice
# 1 Bob
# 2 Charlie
```

## 6. zip

`zip()` (打包函数) combines multiple Iterables (可迭代对象) element by element.

```python
names = ["Alice", "Bob"]
scores = [90, 85]

for name, score in zip(names, scores):
    print(name, score)

# Output:
# Alice 90
# Bob 85
```

## 7. sum

`sum()` (求和函数) returns the total of Numeric Values (数值).

```python
numbers = [1, 2, 3, 4]

print(sum(numbers))

# Output:
# 10
```

## 8. min and max

`min()` (最小值函数) and `max()` (最大值函数) return the smallest and largest Values (值).

```python
numbers = [5, 2, 9, 1]

print(min(numbers))
print(max(numbers))

# Output:
# 1
# 9
```

## 9. sorted

`sorted()` (排序函数) returns a new sorted List (列表) without changing the original Iterable (可迭代对象).

```python
numbers = [3, 1, 2]

sorted_numbers = sorted(numbers)

print(numbers)
print(sorted_numbers)

# Output:
# [3, 1, 2]
# [1, 2, 3]
```

## 10. input

`input()` (输入函数) reads Text (文本) from the user as a String (字符串).

```python
# Run this script and type your name when prompted.
name = input("Enter your name: ")
print("Hello,", name)

# Example Output:
# Enter your name: Alice
# Hello, Alice
```
