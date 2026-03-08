---
title: "Bash Shell Script"
published: 2026-02-07
description: "Bash Shell Script"
image: ""
tags: ["tools","Bash Shell Script"]
category: tools
draft: false
lang: ""
---


# **I. Bash Shell Script Fundamentals**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">

<span style="color:#E8600A;font-weight:700">Bash Shell Script (Bash脚本)</span> is a scripting language executed by the <span style="color:#E8600A;font-weight:700">Unix Shell (Unix命令解释器)</span>. It is widely used for system automation, environment configuration, and workflow management on <span style="color:#E8600A;font-weight:700">Linux Systems (Linux系统)</span> and <span style="color:#E8600A;font-weight:700">High Performance Computing (高性能计算, HPC)</span> environments.

Unlike compiled languages, Bash follows an <span style="color:#E8600A;font-weight:700">Interpreter Model (解释执行模型)</span>, meaning commands are executed line by line by the shell.

</div>

---

## <span style="color:#E8600A">1.</span> **What is a Bash Script**

### 1) Core Definition

A <span style="color:#E8600A;font-weight:700">Bash Script (Bash脚本)</span> is a text file containing a sequence of shell commands that are executed by the <span style="color:#E8600A;font-weight:700">Bash Interpreter (Bash解释器)</span>.

Its main purposes include:

1）Automating repetitive command execution
2）Managing <span style="color:#E8600A;font-weight:700">Environment Variables (环境变量)</span>
3）Controlling program flow using <span style="color:#E8600A;font-weight:700">Conditional Statements (条件语句)</span> and <span style="color:#E8600A;font-weight:700">Loops (循环)</span>
4）Serving as initialization scripts in Linux and HPC environments

<span style="color:#2980B9">Therefore</span>, Bash acts as a bridge between system commands and automated workflows.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
Bash is a <span style="color:#E8600A;font-weight:700">Scripting Language (脚本语言)</span>, not a compiled language. Scripts are interpreted directly by the shell rather than compiled into machine code.
</div>

---

## <span style="color:#E8600A">2.</span> **Script Execution Methods**

There are three common ways to run a Bash script.

### 1) Direct Execution with Bash

```bash
bash script.sh
```

This launches a <span style="color:#E8600A;font-weight:700">Subshell (子Shell)</span> to execute the script.

---

### 2) Executable Script

First give execution permission:

```bash
chmod +x script.sh
```

Then run the script:

```bash
./script.sh
```

This uses the system's <span style="color:#E8600A;font-weight:700">Shebang (解释器声明)</span> if defined.

Example:

```bash
#!/bin/bash
```

---

### 3) Source Execution

```bash
source script.sh
```

or

```bash
. script.sh
```

This executes the script inside the <span style="color:#E8600A;font-weight:700">Current Shell (当前Shell)</span>.

---

### Execution Behavior Comparison

| Method                                                                                                   | New Subshell (子Shell) | Variables Persist (变量保留) |
| -------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------ |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">bash script.sh</code>   | Yes                   | No                       |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">./script.sh</code>      | Yes                   | No                       |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">source script.sh</code> | No                    | Yes                      |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
The <span style="color:#E8600A;font-weight:700">source Command (source命令)</span> is commonly used when configuring shell environments such as <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.bashrc</code>, CUDA paths, or Conda environments.
</div>

---

# **II. Basic Bash Syntax**

## <span style="color:#E8600A">1.</span> **Variables and Environment Variables**

### 1) Normal Variables

Variables are assigned without spaces around the equals sign.

```bash
name="Alice"
echo $name
```

Here:

* <span style="color:#E8600A;font-weight:700">Variable Assignment (变量赋值)</span> defines the variable
* <span style="color:#E8600A;font-weight:700">Variable Expansion (变量展开)</span> occurs using `$`

---

### 2) Environment Variables

Environment variables are exported using the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">export</code> command.

```bash
export PATH=/usr/bin:$PATH
```

Properties:

1）Visible to <span style="color:#E8600A;font-weight:700">Child Processes (子进程)</span>
2）Frequently used for software configuration such as:

* <span style="color:#E8600A;font-weight:700">PATH</span>
* <span style="color:#E8600A;font-weight:700">CUDA</span>
* <span style="color:#E8600A;font-weight:700">Conda</span>
* <span style="color:#E8600A;font-weight:700">MPI</span>

---

## <span style="color:#E8600A">2.</span> **Conditional Statements**

### 1) Basic `if` Structure

```bash
if condition; then
    statement
fi
```

Example:

```bash
if [ -f file.txt ]; then
    echo "file exists"
fi
```

Here the brackets use the <span style="color:#E8600A;font-weight:700">Test Command (测试命令)</span>.

---

### 2) Common File Test Operators

| Operator                                                                                   | Meaning      |
| ------------------------------------------------------------------------------------------ | ------------ |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-f</code> | regular file |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-d</code> | directory    |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-e</code> | exists       |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">-x</code> | executable   |

These belong to the <span style="color:#E8600A;font-weight:700">POSIX Test Syntax (POSIX测试语法)</span>.

---

## <span style="color:#E8600A">3.</span> **Loop Structures**

### 1) `for` Loop

The most common loop used to iterate through lists or files.

```bash
for f in *.txt; do
    echo $f
done
```

Typical use cases:

* File iteration
* Command results
* Parameter lists

---

### 2) `while` Loop

Often used for reading files line by line.

```bash
while read line; do
    echo $line
done < file.txt
```

This uses <span style="color:#E8600A;font-weight:700">Input Redirection (输入重定向)</span>.

---

# **III. Command Execution Model**

## <span style="color:#E8600A">1.</span> **Subshell Execution**

```bash
bash script.sh
```

Characteristics:

1）Creates a new <span style="color:#E8600A;font-weight:700">Process (进程)</span>
2）Variables do not affect the current shell

---

## <span style="color:#E8600A">2.</span> **Source Execution**

```bash
source script.sh
```

or

```bash
. script.sh
```

Purpose:

> Execute script content in the <span style="color:#E8600A;font-weight:700">Current Shell Environment (当前Shell环境)</span>.

Typical usage scenarios:

* <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.bashrc</code>
* environment setup
* CUDA configuration
* HPC module initialization

---

# **IV. File and Path Operations**

## <span style="color:#E8600A">1.</span> **File Testing**

```bash
[ -d dir ]
[ -f file ]
```

These checks are part of the <span style="color:#E8600A;font-weight:700">POSIX Test System (POSIX测试系统)</span>.

---

## <span style="color:#E8600A">2.</span> **Path Expansion and Wildcards**

Example:

```bash
~/.bashrc.d/*
```

Meaning:

| Symbol | Meaning                     |
| ------ | --------------------------- |
| `~`    | user home directory         |
| `*`    | wildcard matching all files |

This mechanism is called <span style="color:#E8600A;font-weight:700">Filename Expansion (文件名扩展)</span>.

---

# **V. Modular Configuration Example**

A typical `.bashrc` uses modular configuration loading.

```bash
if [ -d ~/.bashrc.d ]; then
    for rc in ~/.bashrc.d/*; do
        if [ -f "$rc" ]; then
            . "$rc"
        fi
    done
fi
```

Explanation:

1）Check whether the directory exists
2）Iterate through each script file
3）Execute each script using <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">source</code>

This allows a <span style="color:#E8600A;font-weight:700">Modular Configuration System (模块化配置系统)</span>.

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85">
<span style="color:#E8600A;font-weight:700">Note: </span>
This design pattern is widely used in Linux distributions and <span style="color:#E8600A;font-weight:700">HPC Initialization Scripts (HPC初始化脚本)</span> to organize configuration files into reusable modules.
</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>

<span style="color:#E8600A;font-weight:700">Bash Scripts (Bash脚本)</span> provide a powerful automation mechanism for Linux systems by combining command execution, environment configuration, and control flow using an interpreter-based shell environment.

</div>
