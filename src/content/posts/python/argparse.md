---
title: "argparse"
published: 2026-04-29
description: "argparse"
image: ""
tags: ["python","argparse"]
category: python
draft: false
lang: ""
createdAt: "2026-04-29T15:27:02.006.979998727Z"
---

# Python argparse (命令行参数解析库)

## 1. Overview

`argparse` (参数解析库) is a standard Python library used to parse command-line arguments (命令行参数) and generate user-friendly interfaces for CLI tools.

<br>

## 2. Basic Usage

### 1) Simple Example

```
import argparse

# Create parser
parser = argparse.ArgumentParser(description="Demo program")

# Add argument
parser.add_argument("--name", type=str, help="Your name")

# Parse arguments
args = parser.parse_args()

# Use argument
print(f"Hello, {args.name}!")
# Run in terminal
python script.py --name Alice
# Output
Hello, Alice!
```

The `ArgumentParser` (参数解析器) defines expected inputs and converts CLI inputs into accessible Python attributes.

<br>

## 3. Common Argument Types

### 1) Positional Arguments (位置参数)

```
parser.add_argument("filename", type=str)
```

Positional arguments (位置参数) are required inputs determined by their position in the command.

### 2) Optional Arguments (可选参数)

`action="store_true"` (存储为True行为) means the argument becomes `True` if the flag appears in the command line, otherwise it defaults to `False`.

```
parser.add_argument("--verbose", action="store_true")
```

Optional arguments (可选参数) are prefixed with `-` or `--` and can be omitted.

### 3) Default Values (默认值)

```
parser.add_argument("--age", type=int, default=18)
```

Default values (默认值) provide fallback data when arguments are not supplied.

<br>

## 4. Argument Configuration

### 1) Type Enforcement (类型约束)

```
parser.add_argument("--num", type=int)
```

The `type` (类型) parameter enforces input data types during parsing.

### 2) Choices (可选值限制)

```
parser.add_argument("--mode", choices=["train", "test"])
```

The `choices` (可选值) parameter restricts inputs to predefined options.

### 3) Required Arguments (必填参数)

```
parser.add_argument("--config", required=True)
```

The `required` (必填) flag ensures the argument must be provided.

<br>

## 5. Help and Documentation

### 1) Auto Help Message

```
python script.py --help
```

`argparse` automatically generates help messages (帮助信息) based on argument definitions.

### 2) Custom Description

```
parser = argparse.ArgumentParser(description="My CLI Tool")
```

The `description` (描述信息) provides a summary of the program functionality.

<br>

## 6. Advanced Features

### 1) Subparsers (子命令解析)

```
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

train_parser = subparsers.add_parser("train")
train_parser.add_argument("--epochs", type=int)

args = parser.parse_args()

if args.command == "train":
    print(f"Training for {args.epochs} epochs")
```

Subparsers (子命令) allow building multi-command CLI tools similar to `git`.

### 2) Boolean Flags (布尔标志)

```
parser.add_argument("--debug", action="store_true")
```

The `store_true` (布尔存储) action sets the value to `True` when the flag is present.

<br><br>
