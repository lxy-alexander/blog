---
title: Prompt
published: 2025-12-18
description: "Prompt"
image: "./cover.jpeg"
tags: ["Blogging", "Prompt"]
category: Guides
draft: false
---

# Prompt for Algorithm

```c++
Please generate a systematic and comprehensive explanation for the algorithm: XXX.

Very important requirements:
1. Do NOT use any icons, emojis, or decorative symbols.
2. Use Markdown heading levels strictly as follows:
   - Use exactly one level-1 title for the algorithm name: "# XXX"
   - Use level-2 headings for the main sections: "## I. ...", "## II. ...", "## III. ...", "## IV. ..."
   - Use level-3 headings for each representative example name.
   - Do NOT write "Example name:" as plain text anywhere.
3. Writing style should resemble lecture notes and interview preparation notes.
4. The explanation should be logically progressive and easy to understand.
5. Avoid overly academic or complex wording; prefer clear and simple language.
6. Focus on intuition, invariants, and problem-solving patterns.
7. The output must strictly follow the structure below, and all sections must be included.
8. all codes use c++ and python to implement

You must generate the actual algorithm explanation content, not a prompt or outline.

Output structure (must match exactly):

# XXX

## I. Algorithm Overview (What)
- One sentence describing what the algorithm is.

## II. Problem Scope (What Problems It Solves)
- Types of problems this algorithm is designed for.
- Signals that indicate this algorithm should be used.

## III. All Representative Examples

### Example Title 1
(code block)
Time complexity explanation.
Space complexity explanation.

### Example Title 2
(code block)
Time complexity explanation.
Space complexity explanation.

(Include as many examples as needed, but each example must follow the same pattern.)

## IV. Practice Problems and References
- Provide exactly one classic interview problem related to this algorithm.
- Include problem name, platform, and direct link.

Make sure every explanation is clear enough for someone learning the algorithm for the first time, while still being useful for interview preparation.

```



# Prompt for vLLM

```c++
I will give a <TOPIC>. Please generate a systematic and comprehensive explanation for <TOPIC>.

Very important requirements:
1. Do NOT use any icons, emojis, or decorative symbols.
2. Use Markdown heading levels strictly as follows:
   - Use exactly one level-1 title: "# <TOPIC>"
   - Use level-2 headings: "## 1. ...", "## 2. ...", "## 3. ...", "## 4. ..."
3. All code examples must use appropriate languages for system usage.
4. For each section (each level-2 heading):
   - First, write the complete explanation of this section in Chinese.
   - Immediately after that section, provide its full English translation.
   - Do NOT interleave English and Chinese within the same paragraph.
   - Do NOT translate sentence by sentence.
   - Translation granularity is per section.
5. STYLE REQUIREMENT (VERY IMPORTANT):
   - Write in an interview-friendly, oral explanation style.
   - Use short, independent sentences.
   - Each sentence should express exactly one key idea.
   - Each sentence should be easy to explain verbally to an interviewer.
   - Avoid long or nested sentences.
6. No missing knowledge points. No skipped explanations.
7. If the topic has code, show how to use it.

```

