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

# **I. zsh Plugins — Autosuggestions & Syntax Highlighting**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> Two essential zsh plugins for a better terminal experience: <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">zsh-autosuggestions</code> provides auto-complete, parameter hints, and history-based suggestions; <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">zsh-syntax-highlighting</code> colorizes valid commands and flags errors in real time as you type.
</div>

---

## 1. macOS

- <span style="color:#E8600A;font-weight:700">Command suggestions</span> appear in gray while typing (pulled from shell history)
- <span style="color:#E8600A;font-weight:700">Syntax highlighting</span> colors valid commands and highlights mistakes inline

### 1) Install (Homebrew)

```bash
brew install zsh-autosuggestions zsh-syntax-highlighting
```

### 2) Enable (add to `~/.zshrc`)

```bash
echo 'source /usr/local/share/zsh-autosuggestions/zsh-autosuggestions.zsh' >> ~/.zshrc
echo 'source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh' >> ~/.zshrc
source ~/.zshrc
```

✅ Works immediately.

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Install both plugins via Homebrew, source them in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">~/.zshrc</code>, and your shell gains history-based grey suggestions and live syntax coloring instantly.</div>