---
title: "zsh-autosuggestions"
published: 2026-01-29
description: "zsh-autosuggestions"
image: ""
tags: ["tools","zsh-autosuggestions"]
category: tools
draft: false
lang: ""
---

>   auto-complete / parameter hints / history suggestions

# MacOS

-   **Command suggestions** in gray while typing (from history)
-   **Syntax highlighting** (valid commands in color, mistakes highlighted)

## 1) Install (Homebrew)

```bash
brew install zsh-autosuggestions zsh-syntax-highlighting
```



## 2) Enable (add to `~/.zshrc`)

```bash
echo 'source /usr/local/share/zsh-autosuggestions/zsh-autosuggestions.zsh' >> ~/.zshrc
echo 'source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh' >> ~/.zshrc
source ~/.zshrc
```

✅ Works immediately.





# Linux

