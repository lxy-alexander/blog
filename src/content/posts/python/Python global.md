---
title: "Python global"
published: 2026-03-09
description: "Python global"
image: ""
tags: ["python","Python global"]
category: python
draft: false
lang: ""
---

# **I. Understanding `global` in Python**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> keyword lets you modify <span style="color:#2980B9">global variables (全局变量)</span> inside functions. Without it, assignments create <span style="color:#2980B9">local variables (局部变量)</span> instead.
</div>


## 1. Scope Basics in Python

Every variable in Python has a defined <span style="color:#2980B9">scope (作用域)</span> – the region of code where it is accessible.

- **Global scope**: Variables defined outside any function
- **Local scope**: Variables defined inside a function

```python
# Global variable
x = 10

def my_function():
    # Local variable (different from global x)
    x = 5
    print("Inside function:", x)

my_function()  # Output: Inside function: 5
print("Outside function:", x)  # Output: Outside function: 10
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In the example above, the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">x</code> inside the function is <span style="color:#C0392B;font-weight:600">completely separate</span> from the global <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">x</code>. This is Python's way of preventing accidental modification of global variables.</div>

## 2. The Problem: Modifying Global Variables

When you try to <span style="color:#E8600A;font-weight:700">modify</span> a global variable inside a function without declaring it as `global`, Python creates a new local variable instead.

```python
counter = 0

def increment():
    # This creates a NEW local variable 'counter'
    counter += 1  # ❌ ERROR!

# increment()  # Uncommenting this line causes UnboundLocalError
```

<span style="color:#C0392B;font-weight:600">Pitfall: </span>This code raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">UnboundLocalError: local variable 'counter' referenced before assignment</code> because Python sees the assignment and treats <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">counter</code> as a local variable, but it's being referenced before it's defined.

## 3. The Solution: Using `global`

The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> keyword tells Python: <span style="color:#2980B9">"This variable belongs to the global scope"</span>.

```python
counter = 0  # Global variable

def increment():
    global counter  # Declare that we're using the global counter
    counter = counter + 1
    print(f"Counter is now: {counter}")

increment()  # Output: Counter is now: 1
increment()  # Output: Counter is now: 2
print(f"Global counter: {counter}")  # Output: Global counter: 2
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span>The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> statement must come <span style="color:#C0392B;font-weight:600">before</span> any use of the variable in the function.</div>

## 4. Multiple Global Variables

You can declare multiple global variables in a single statement:

```python
name = "Python"
version = 3.9
year = 2023

def update_info():
    global name, version, year
    name = "Python 3"
    version = 3.11
    year = 2024

update_info()
print(f"{name} {version} ({year})")  # Output: Python 3 3.11 (2024)
```

## 5. Global vs Local: A Comparison

| Aspect                                            | Local Variable                 | Global Variable                                              |
| ------------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| <span style="color:#2980B9">Scope (作用域)</span> | Inside function only           | Throughout the module                                        |
| <span style="color:#2980B9">Declaration</span>    | Automatic on assignment        | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> keyword required inside functions |
| <span style="color:#2980B9">Read access</span>    | Direct access                  | Direct access (without <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">global</code> for reading) |
| <span style="color:#2980B9">Write access</span>   | Direct assignment              | Requires <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> declaration |
| <span style="color:#2980B9">Memory (内存)</span>  | Created when function runs     | Created when module loads                                    |
| <span style="color:#2980B9">Best practice</span>  | Preferred for temporary values | Use sparingly, prefer parameters                             |

## 6. Reading Global Variables (Without `global`)

Interesting fact: You can <span style="color:#E8600A;font-weight:700">read</span> global variables without the `global` keyword:

```python
message = "Hello, World!"

def show_message():
    # No 'global' needed for reading
    print(message)  # This works!

show_message()  # Output: Hello, World!
```

<span style="color:#C0392B;font-weight:600">Warning: </span>This only works for <span style="color:#2980B9">reading (读取)</span>. The moment you try to assign a value, Python treats it as a local variable unless you use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code>.

## 7. Nested Functions and `global`

The `global` keyword always refers to the <span style="color:#2980B9">module-level (模块级别)</span> scope, not the enclosing function scope.

```python
x = "global x"

def outer():
    x = "outer x"
    
    def inner():
        global x  # This refers to the module-level 'x', not outer's 'x'
        x = "changed by inner"
    
    inner()
    print("outer x:", x)  # Output: outer x: outer x

outer()
print("global x:", x)  # Output: global x: changed by inner
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span>To modify variables in an <span style="color:#2980B9">enclosing (but non-global) scope (外部嵌套作用域)</span>, use the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">nonlocal</code> keyword instead, which we'll cover in a separate note.</div>

## 8. Mutable Objects: A Special Case

For <span style="color:#2980B9">mutable objects (可变对象)</span> like lists and dictionaries, you can modify their <span style="color:#E8600A;font-weight:700">contents</span> without using `global`:

```python
my_list = [1, 2, 3]
my_dict = {"count": 0}

def modify_mutable():
    # No 'global' needed - we're modifying, not reassigning
    my_list.append(4)
    my_dict["count"] += 1
    print("Inside function:", my_list, my_dict)

modify_mutable()
print("Outside function:", my_list, my_dict)
# Output: Outside function: [1, 2, 3, 4] {'count': 1}
```

<span style="color:#C0392B;font-weight:600">Pitfall: </span>This works because we're <span style="color:#2980B9">modifying (修改)</span> the object, not <span style="color:#2980B9">reassigning (重新赋值)</span> the variable. If we tried `my_list = [4, 5, 6]`, that would require `global`.

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">global</code> only when you must modify a module-level variable from inside a function; otherwise, pass values as parameters and return results for cleaner, more maintainable code.</div>
