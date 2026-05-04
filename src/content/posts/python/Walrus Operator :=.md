---
title: "Walrus Operator :="
published: 2026-05-03
description: "Walrus Operator :="
image: ""
tags: ["python","Walrus Operator :="]
category: python
draft: false
lang: ""
createdAt: "2026-05-03T04:00:43.041.758295435Z"
---

# Walrus Operator `:=`

The Walrus Operator (海象运算符) assigns a value to a variable inside an Expression (表达式).

## 1. Definition

The Walrus Operator (海象运算符) lets you assign and use a value in the same Expression (表达式).

```
value = 10

if (n := value) > 5:
    print(n)

# Output:
# 10
```

<br>

## 2. Why It Is Useful

The Walrus Operator (海象运算符) reduces repeated computation by storing an intermediate result inside a condition.

```
def get_data():
    print("Function called")
    return [1, 2, 3]


if (data := get_data()):
    print(data)

# Output:
# Function called
# [1, 2, 3]
```

Without `:=`, you usually write:

```
data = get_data()

if data:
    print(data)

# Output:
# Function called
# [1, 2, 3]
```

<br>

## 3. Common Use in `while`

The Walrus Operator (海象运算符) is commonly used in a While Loop (while 循环) to assign and check a value at the same time.

```
inputs = ["hello", "python", "quit"]
index = 0

while (text := inputs[index]) != "quit":
    print(text)
    index += 1

# Output:
# hello
# python
```

<br>

## 4. Common Use in `if`

The Walrus Operator (海象运算符) is useful when a value is needed both for checking and later use.

```
name = "Alice"

if (length := len(name)) > 3:
    print(length)

# Output:
# 5
```

<br>

## 5. Common Use in List Comprehension

The Walrus Operator (海象运算符) can store a computed value inside a List Comprehension (列表推导式).

```
numbers = [1, 2, 3, 4, 5]

result = [square for x in numbers if (square := x * x) > 10]

print(result)

# Output:
# [16, 25]
```

<br>

## 6. Important Rule

The Walrus Operator (海象运算符) assigns inside an Expression (表达式), but a normal Assignment Statement (赋值语句) uses `=`.

```
x = 10
print(x)

# Output:
# 10
if (x := 10) > 5:
    print(x)

# Output:
# 10
```

<br>

## 7. Invalid Example

The Walrus Operator (海象运算符) usually needs parentheses when used in conditions for readability and syntax safety.

```
# Invalid:
# if x := 10 > 5:
#     print(x)

# Correct:
if (x := 10) > 5:
    print(x)

# Output:
# 10
```

<br>
