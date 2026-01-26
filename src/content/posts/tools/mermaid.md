---
title: Mermaid
published: 2026-01-25
description: "Mermaid"
image: ""
tags: ["Mermaid"]
category: tools
draft: false
---

## Export using Mermaid CLI

## 1) Install Mermaid CLI

```
npm install -g @mermaid-js/mermaid-cli
```

------

## 2) Save your diagram code into `arch.mmd`

Example content:

```
sequenceDiagram
    A->>B: hi
```

------

## 3) Export as PNG

```
mmdc -i arch.mmd -o arch.png
```

------

## 4Export as SVG (optional)

```
mmdc -i arch.mmd -o arch.svg
```