---
title: "Python __init__.py"
published: 2026-01-28
description: "Python __init__.py"
image: ""
tags: ["python","Python __init__.py"]
category: python
draft: false
lang: ""
---

## The function of `__init__.py`  

In Python, `__init__.py` typically has four common roles:

1.  ==**Turn a directory into a package**==
     With `__init__.py`, you can `import` the package name (in modern Python, it can sometimes be omitted, but it is still widely used).
    
    
    
2.  **Act as the package entry point**
     When you run `import xxx`, Python will execute `xxx/__init__.py` first.
    
3.  **Expose a clean public API (hide internal structure)**
     Users can write `from package import something` without needing to know how files are organized inside the package.
    
4.  **Run initialization logic (lazy init / plugin registration / config loading)**
     For example, vLLM’s `current_platform` is initialized only when it is accessed for the first time.



