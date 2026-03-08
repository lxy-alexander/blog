---
title: "Note Prompt"
published: 2026-03-07
description: "Note Prompt"
image: ""
tags: ["prompt","Note Prompt"]
category: prompt
draft: false
lang: ""
---



```
Generate a Chinese learning note in Typora-compatible Markdown with the following requirements:

## 1. Language
- Write all explanations in **English**
- All **Technical Terms (专业术语)** must include Chinese annotation in parentheses
- Example: `The Time Complexity (时间复杂度) is O(n).`

## 2. Heading Structure
- Level 1: Roman numerals → `# **I. Title**`
- Level 2: Arabic numbers → `## 1. **Keyword**`
- Level 3: Parentheses → `### 1) Sub-title`

## 3. Color System (via inline HTML)
- Key terms / important values: `<span style="color:#E8600A;font-weight:700">text</span>`
- Warnings / pitfalls: `<span style="color:#C0392B;font-weight:600">text</span>`
- Structural / linking words: `<span style="color:#2980B9">text</span>`
- Section number prefix: `<span style="color:#E8600A">1.</span>`
- Inline code tags (orange): `<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">text</code>`
- Inline code tags (blue): `<code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">text</code>`

## 4. Block Design (via inline HTML)
- Overview block (blue-purple):
  `<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> ... </div>`
- Note / annotation block (gray + orange line):
  `<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> ... </div>`
- Summary block (gradient):
  `<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> ... </div>`

## 5. Content Structure
1. Overview block (2–3 sentences summarizing the core idea)
2. Numbered sections expanding each concept
3. Code blocks with `cpp` / `python` / relevant language tag
4. Note blocks for background knowledge or edge cases
5. Comparison table where applicable
6. One-line Takeaway block at the end

## 6. Content
[Paste your content here]
```

