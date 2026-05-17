---
title: "lambda function"
published: 2026-05-16
description: "lambda function"
image: ""
tags: ["python","lambda function"]
category: python
draft: false
lang: ""
createdAt: "2026-05-16T21:18:12.914.246529784Z"
---

# Python Lambda Functions
A lambda function is an anonymous function (匿名函数) that can take any number of arguments but contains only a single expression.

## 1. Syntax and Structure
The syntax consists of the `lambda` keyword, followed by parameters, a colon, and a single expression (表达式) that implicitly returns a result.

$$\text{lambda } \text{arguments} : \text{expression}$$

```python
# Breakdown of: add_ten = lambda x: x + 10

# 1. 'lambda'   -> Keyword that defines the anonymous function (匿名函数)
# 2. 'x'        -> Parameter (参数)
# 3. ':'        -> Syntactic separator (语法分隔符)
# 4. 'x + 10'   -> Expression that is evaluated and implicitly returned (隐式返回)

add_ten = lambda x: x + 10
print(add_ten(5))  # Output: 15
```



<br>

## 2. Code Example and Use Cases
Lambda functions are ideally used as inline functions (内联函数) passed into higher-order functions (高阶函数) for short-lived, one-time operations.

```python
# 1. Basic usage: A lambda function that adds 10 to the input
add_ten = lambda x: x + 10
print(add_ten(5))  # Output: 15

# 2. Advanced usage: Sorting a list of tuples by the second element using lambda as a key
pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)  # Output: [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]

# 3. Higher-order function usage: Filtering even numbers from a list
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4, 6]
