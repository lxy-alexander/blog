---
title: "ast"
published: 2026-05-02
description: "ast"
image: ""
tags: ["python","ast"]
category: python
draft: false
lang: ""
createdAt: "2026-05-02T12:10:24.538.977526629Z"
---

# Python AST (抽象语法树)

The **`ast` module (抽象语法树模块)** is Python's standard library for parsing source code (解析源码) into a **tree of node objects (节点对象树)**, enabling **static analysis (静态分析)**, **code transformation (代码变换)**, and **safe evaluation (安全求值)** without executing the code.

## 1. Core Concept (核心概念)

An **AST (Abstract Syntax Tree, 抽象语法树)** represents Python source code (Python 源码) as a hierarchical tree of `ast.AST` nodes (节点), where each node corresponds to a syntactic construct (语法结构) like a function definition, call, or expression.

```python
import ast

source = "x = 1 + 2"
tree = ast.parse(source)
print(ast.dump(tree, indent=2)) # indent 2 spaces
# Output:
# Module(
#   body=[
#     Assign(
#       targets=[Name(id='x', ctx=Store())],
#       value=BinOp(
#         left=Constant(value=1),
#         op=Add(),
#         right=Constant(value=2)))])
```

<br>

## 2. ast.parse() — Source to Tree (源码到树)

`ast.parse(source)` is the entry point (入口), converting a source string into an `ast.Module` node — the root of the AST tree (AST 树的根节点).

```python
import ast

source = """
def greet(name):
    return f"Hello, {name}"
"""

# Parse with optional filename for error messages (可选 filename 用于错误信息)
tree = ast.parse(source, filename="example.py") # filename is a tag for the parser and error report.

print(type(tree))
print(type(tree.body[0]))
# Output:
# <class 'ast.Module'>
# <class 'ast.FunctionDef'>
```

<br>

## 3. Common Node Types (常见节点类型)

`ast.AST` is the **base class (基类)** for all nodes; each Python construct maps to a specific subclass (子类).

| Node (节点)            | Represents (代表)                          |
| ---------------------- | ------------------------------------------ |
| `ast.Module`           | A whole file (整个文件)                    |
| `ast.ClassDef`         | `class Xxx:` definition (类定义)           |
| `ast.FunctionDef`      | `def xxx():` function (普通函数)           |
| `ast.AsyncFunctionDef` | `async def xxx():` (异步函数)              |
| `ast.Call`             | Function call like `foo()` (函数调用)      |
| `ast.Attribute`        | Attribute access like `os.path` (属性访问) |
| `ast.Name`             | Variable name like `x`, `torch` (变量名)   |
| `ast.Constant`         | Literal: string/number/True/None (字面量)  |
| `ast.If`               | `if` statement (if 语句)                   |
| `ast.With`             | `with` context manager (with 语句)         |
| `ast.AsyncWith`        | `async with` (异步 with 语句)              |
| `ast.BoolOp`           | `a and b`, `x or y` (布尔运算)             |
| `ast.And` / `ast.Or`   | Boolean operators (布尔运算符)             |

<br>

## 4. Inspecting Node Structure (检查节点结构)

Each node has **specific fields (特定字段)** accessible as attributes (属性), e.g. `name`, `body`, `func`.

```python
import ast

source = """
class Dog:
    '''A dog class.'''
    def bark(self):
        print("Woof!")
"""

tree = ast.parse(source)
class_node = tree.body[0]            # First top-level node (第一个顶层节点)

print(f"Type:    {type(class_node).__name__}")
print(f"Name:    {class_node.name}")
print(f"Body:    {len(class_node.body)} statement(s)")
print(f"Doc:     {ast.get_docstring(class_node)}")
# Output:
# Type:    ClassDef
# Name:    Dog
# Body:    2 statement(s)
# Doc:     A dog class.
```

<br>

## 5. Key Node Fields (关键节点字段)

Different node types expose different fields for accessing their components (访问其组成部分).

| Field (字段) | Available on (出现在)         | Meaning (含义)                           |
| ------------ | ----------------------------- | ---------------------------------------- |
| `node.name`  | `FunctionDef`, `ClassDef`     | Function/class name (函数名/类名)        |
| `node.body`  | `Module`, `FunctionDef`, `If` | Inner code block (内部代码块)            |
| `node.func`  | `Call`                        | The callable being called (被调用的函数) |
| `node.args`  | `Call`, `FunctionDef`         | Argument list (参数列表)                 |
| `node.value` | `Constant`, `Assign`          | The literal value or RHS (字面量或右值)  |
| `node.attr`  | `Attribute`                   | The attribute name string (属性名字符串) |
| `node.id`    | `Name`                        | The variable name string (变量名字符串)  |

<br>

## 6. ast.get_docstring() — Extract Docstrings (提取 docstring)

Convenience function (便捷函数) to read the docstring of modules, classes, and functions.

```python
import ast

source = '''
"""Module-level docstring (模块级 docstring)."""

def func():
    """Function docstring (函数 docstring)."""
    pass

class Cls:
    """Class docstring (类 docstring)."""
    pass
'''

tree = ast.parse(source)

print(ast.get_docstring(tree))
print(ast.get_docstring(tree.body[1]))    # FunctionDef
print(ast.get_docstring(tree.body[2]))    # ClassDef
# Output:
# Module-level docstring (模块级 docstring).
# Function docstring (函数 docstring).
# Class docstring (类 docstring).
```

<br>

## 7. ast.iter_child_nodes() vs ast.walk() (子节点遍历对比)

These two functions differ in **traversal depth (遍历深度)** — direct children vs full tree.

| Function (函数)              | Behavior (行为)                                    |
| ---------------------------- | -------------------------------------------------- |
| `ast.iter_child_nodes(node)` | Direct children only, non-recursive (仅直接子节点) |
| `ast.walk(tree)`             | Recursive over all descendants (递归遍历所有后代)  |

```python
import ast

source = """
def outer():
    def inner():
        x = 1
"""

tree = ast.parse(source)
outer = tree.body[0]

# Direct children only (仅直接子节点)
print("Direct children of outer:")
for child in ast.iter_child_nodes(outer):
    print(f"  {type(child).__name__}")
# Output:
# Direct children of outer:
#   arguments
#   FunctionDef        ← inner function
#                      (no further descent, 不再深入)

# Full recursive walk (完全递归遍历)
print("\nAll descendants:")
for node in ast.walk(outer):
    print(f"  {type(node).__name__}")
# Output:
# All descendants:
#   FunctionDef        ← outer
#   arguments
#   FunctionDef        ← inner
#   arguments
#   Assign
#   Name
#   Store
#   Constant
```

<br>

## 8. ast.NodeVisitor — Custom Traverser (自定义遍历器)

`ast.NodeVisitor` is the **standard pattern (标准模式)** for traversing an AST: subclass it and define `visit_<NodeType>` methods (定义 `visit_Xxx` 方法) that fire automatically (自动触发) when matching nodes are encountered.

```python
import ast

class FunctionFinder(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
    
    # Auto-called for FunctionDef nodes (遇到函数定义自动调用)
    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)        # Continue into children (继续遍历子节点)
    
    def visit_AsyncFunctionDef(self, node):
        self.functions.append(f"async {node.name}")
        self.generic_visit(node)

source = """
def add(a, b): return a + b
async def fetch(): pass
class C:
    def method(self): pass
"""

tree = ast.parse(source)
finder = FunctionFinder()
finder.visit(tree)

print(finder.functions)
# Output: ['add', 'async fetch', 'method']
```

<br>

## 9. Detecting Function Calls (检测函数调用)

A common task is **finding all calls (查找所有调用)** to a specific function via `visit_Call`.

```python
import ast

class CallFinder(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
    
    def visit_Call(self, node):
        # node.func can be Name (foo) or Attribute (obj.method)
        # node.func 可能是 Name (foo) 或 Attribute (obj.method)
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(f"{ast.unparse(node.func.value)}.{node.func.attr}")
        self.generic_visit(node)

source = """
print("hi")
os.path.join("a", "b")
model.generate(prompt)
foo(1, 2)
"""

tree = ast.parse(source)
finder = CallFinder()
finder.visit(tree)

print(finder.calls)
# Output: ['print', 'os.path.join', 'model.generate', 'foo']
```

<br>

## 10. ast.literal_eval() — Safe Evaluation (安全求值)

`ast.literal_eval()` safely converts a **literal string (字面量字符串)** into a Python object, supporting only **literals (仅字面量)** — strings, numbers, tuples, lists, dicts, sets, booleans, and `None`.

```python
import ast

# Safe — only literals are allowed (安全 — 仅允许字面量)
print(ast.literal_eval("[1, 2, 3]"))         # Output: [1, 2, 3]
print(ast.literal_eval("{'a': 1, 'b': 2}"))  # Output: {'a': 1, 'b': 2}
print(ast.literal_eval("(1, True, None)"))   # Output: (1, True, None)
print(ast.literal_eval("'hello'"))           # Output: hello

# Unsafe code is rejected (不安全的代码会被拒绝)
try:
    ast.literal_eval("__import__('os').system('ls')")
except ValueError as e:
    print(f"Blocked: {e}")
# Output: Blocked: malformed node or string ...
```

This is the **safe alternative to `eval()` (`eval()` 的安全替代品)** — never run untrusted input through `eval()` (绝不要对不可信输入用 `eval()`).

<br>

## 11. isinstance() Check Pattern (类型判断模式)

Use `isinstance(node, ast.Xxx)` to **dispatch on node type (按节点类型分支)** when manually traversing.

```python
import ast

source = """
if x and y:
    with open("f") as f:
        result = compute()
"""

tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, ast.If):
        print(f"Found if statement (找到 if 语句)")
    elif isinstance(node, ast.With):
        print(f"Found with statement (找到 with 语句)")
    elif isinstance(node, ast.BoolOp):
        op_type = "and" if isinstance(node.op, ast.And) else "or"
        print(f"Found boolean op: {op_type} (找到布尔运算)")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            print(f"Found call to: {node.func.id} (找到函数调用)")
# Output:
# Found if statement (找到 if 语句)
# Found boolean op: and (找到布尔运算)
# Found with statement (找到 with 语句)
# Found call to: compute (找到函数调用)
```

<br>

## 12. Real-World Example: Linter (真实示例: 代码检查器)

Combine these tools to build a tiny linter (微型代码检查器) that finds dangerous `eval()` calls.

```python
import ast
from pathlib import Path

class EvalDetector(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.warnings = []
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "eval":
            self.warnings.append(
                f"{self.filename}:{node.lineno} — dangerous eval() call (危险的 eval 调用)"
            )
        self.generic_visit(node)

source = """
x = eval(input("Enter: "))
y = int("42")
result = eval(user_data)
"""

# Pass filename for nice error reporting (传入文件名便于错误报告)
tree = ast.parse(source, filename="user_script.py")
detector = EvalDetector("user_script.py")
detector.visit(tree)

for w in detector.warnings:
    print(w)
# Output:
# user_script.py:2 — dangerous eval() call (危险的 eval 调用)
# user_script.py:4 — dangerous eval() call (危险的 eval 调用)
```

<br>

## 13. Common Use Cases (常见应用场景)

The `ast` module powers many production tools (生产工具) across the Python ecosystem.

| Use Case (用例)                | Examples (示例)             |
| ------------------------------ | --------------------------- |
| Linters (代码检查)             | flake8, pylint, ruff        |
| Formatters (代码格式化)        | black, autopep8             |
| Type checkers (类型检查)       | mypy, pyright (partial use) |
| Code transformation (代码变换) | rope, libcst                |
| Safe evaluation (安全求值)     | `ast.literal_eval`          |
| Doc generation (文档生成)      | sphinx-autodoc, pdoc        |
| Test discovery (测试发现)      | pytest collection           |

<br>

## 14. Workflow Summary (工作流总结)

The typical AST analysis pipeline (典型 AST 分析流程):

```text
Source code (源码)
        │
        ▼ ast.parse(source, filename=...)
   AST Tree (AST 树)
        │
        ▼ ast.walk / ast.NodeVisitor
   Visit nodes (遍历节点)
        │
        ▼ isinstance check + node fields
   Extract info / Detect patterns (提取信息 / 检测模式)
        │
        ▼
   Report / Transform / Compile (报告 / 变换 / 编译)
```

<br>

## 15. Key Points (关键要点)

The `ast` module turns Python source into a **traversable tree of typed nodes (类型化节点的可遍历树)**, with `ast.parse()` as the **entry point (入口)**, `ast.NodeVisitor` and `visit_<Type>` methods as the **standard traversal pattern (标准遍历模式)**, `ast.walk()` for recursive iteration, `ast.iter_child_nodes()` for direct children only, `isinstance(node, ast.Xxx)` for **type-based dispatch (基于类型的分支)**, and `ast.literal_eval()` as the **safe replacement for eval (eval 的安全替代品)** — together making `ast` the foundation of all modern Python static analysis tooling (现代 Python 静态分析工具的基石).

<br><br>
