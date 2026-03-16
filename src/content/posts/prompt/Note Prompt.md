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

# System Design Prompt for Claude



# Normal Prompt for Cluade

```javascript
Here is the prompt translated into English:

---

Generate English study notes in Typora-compatible Markdown format with the following requirements:
Please explain the content clearly and concisely. Only explain the material I provide — do not extend beyond it. Every example should include a description of how it is used and in what context:

## 1. Language
- All explanations written in **English**
- All **technical terms** must include a Chinese annotation in parentheses
- Example: `The Time Complexity (时间复杂度) is O(n).`

## 2. Heading Structure
- Level 1 heading: Roman numerals → `# I.`
- Level 2 heading: Arabic numerals → `## 1.`
- Level 3 heading: Parenthesis form → `### 1)`

## 3. Content Structure
1. Level 1 headings contain only an overview block (3 sentences summarizing the core idea, written to be easy to understand and remember). Only the overview block appears directly under a level 1 heading.
2. Write an explanation paragraph directly under each level 2 heading describing what the current topic is.
3. Explanation blocks go directly under level 3 headings (what it is + when to use it).
4. Code blocks must include a language tag: `cpp` / `python` / relevant language.
5. Note blocks go directly below code blocks — note blocks are optional; show them only when relevant, otherwise omit them.
6. Add comparison tables where applicable; the first column alternates row colors between blue (`#2980B9`) and orange (`#E8600A`) using `<span>` wrapping.
7. Add a one-line summary block at the end.
8. Keep code concise.
9. **[MANDATORY]** Every `<div>` block must be written as a single line of HTML — `<div style="..."> all content </div>` must appear on one line with no line breaks inside.

## 4. Block Design

**[MANDATORY COLOR RULE] All text inside blocks must mix the following three colors; each block must use at least 2, all bolded:**
- Orange `#E8600A`: key terms, core concepts, technical terms (with Chinese annotation)
- Red `#C0392B`: warnings, constraints, common mistakes
- Blue `#2980B9`: structural phrases, action descriptions, conditional statements

- **Overview block** (blue-purple):
  `<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> ... </div>`

- **Explanation block** (gray background + orange left border), placed directly below each heading:
  `<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">... </div>`

- **Note block** (gray background + red left border), placed directly below each code block — optional:
  `<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> ... </div>`

- **Summary block** (gradient):
  `<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> ... </div>`

## 5. Typora Compatibility Rules
1. Use `$$ $$` for math formulas, not `\[ \]`
2. Leave one blank line after every `</div>`
3. Do not add `id` attributes to code blocks
4. Do not wrap the entire document in a code block
5. Ensure Markdown structure remains valid after HTML blocks

## 6. Content
【Paste your content here】
```



```javascript
将英文学习笔记以 Typora 兼容的 Markdown 格式生成，具体要求如下：
请简洁易懂地解释内容。只解释我提供的材料，不要延伸。每个示例都应附有说明，描述其使用方式和应用场景：

## 1. 语言
- 所有解释使用**英文**书写
- 所有**专业术语**必须在括号内附上中文注释
- 示例：`The Time Complexity (时间复杂度) is O(n).`

## 2. 标题结构
- 一级标题：罗马数字 → `# I.`
- 二级标题：阿拉伯数字 → `## 1.`
- 三级标题：括号形式 → `### 1) `


## 3. 内容结构
1. 一级标题只有概览块（3句话概括核心思想，简述易懂好记）,大标题（一级标题）下方只有概览块
2. 在二级标题正下方写解释段落，解释当前标题是什么
3. 解释块在，三级标题正下方（是什么 + 使用场景）
4. 代码块，标注语言标签 `cpp` / `python` / 相关语言
5. 注释块在，用于背景知识或边界情况说明，注释块不是必须的，如果有就显示，没有就不显示
6. 适用时添加对比表格，第一列按行蓝色（#2980B9）和橙色 （#E8600A）交替，用 `<span>` 包裹
7. 末尾添加一句话总结块
8. 代码保持精简
9. 【强制】每个 <div> 块必须写成单行HTML，即：<div style="..."> 全部内容 </div> 必须在同一行，内容中不得出现任何换行符。

## 4. 块设计
**【强制配色规则】所有块内的文字必须混用以下三种颜色，每个块至少出现2，全部加粗：**
- 橙色 `#E8600A`：关键术语、核心概念、专业术语（附中文注释）
- 红色 `#C0392B`：警告、限制条件、容易出错的地方
- 蓝色 `#2980B9`：结构性短语、动作描述、条件说明

- 概览块（蓝紫色）：
  `<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> ... </div>`
- 解释块（灰底 + 橙色左边线），放在每个标题正下方：
  `<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">... </div>`
- 注释块（灰底 + 红色左边线），放在每个代码块正下方，但是注释块不是必须的：
  `<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> ... </div>`
- 总结块（渐变色）：
  `<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> ... </div>`


## 5. Typora 兼容性规则
1. 数学公式使用 `$$ $$` 而非 `\[ \]`
2. 每个 `</div>` 后必须留一个空行
3. 代码块不添加 `id` 属性
4. 不将整篇文档包裹在代码块中
5. 确保 HTML 块之后 Markdown 结构仍然有效

## 6. 内容
【在此粘贴你的内容】
```





