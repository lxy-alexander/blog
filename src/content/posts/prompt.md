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
Please generate a systematic and comprehensive explanation for the system, component, or workflow: <TOPIC>.

Very important requirements:
1. Do NOT use any icons, emojis, or decorative symbols.
2. Use Markdown heading levels strictly as follows:
   - Use exactly one level-1 title for the topic name: "# <TOPIC>"
   - Use level-2 headings for the main sections: "## I. ...", "## II. ...", "## III. ...", "## IV. ..."
   - Use level-3 headings for each representative subcomponent, execution path, or usage scenario.
   - Do NOT write subcomponent, path, or scenario names as plain text outside headings.
3. Writing style should resemble lecture notes and interview preparation notes for systems, infrastructure, or software engineering topics.
4. The explanation should be logically progressive and easy to understand.
5. Avoid overly academic, marketing, or narrative language; prefer clear, precise, and engineering-focused explanations.
6. Focus on intuition, design motivations, invariants, execution flow, and system trade-offs.
7. The output must strictly follow the structure below, and all sections must be included.
8. All code examples must use appropriate languages for system usage (e.g., Python, shell commands, configuration snippets, or minimal C++ where relevant).
9. **For every explanatory sentence or bullet point, first write the English version, then immediately follow with its Chinese translation on the next line. Do not merge the two languages into a single sentence.**

Output structure (must match exactly):

# <TOPIC>

## I. Overview (What)
- One sentence describing what this system, component, or workflow is and its core purpose.

## II. Responsibilities and Applicability (Why and When)
- What responsibilities this system or component handles.
- When it is used or involved during execution.
- Signals or scenarios that indicate this topic is relevant to correctness, performance, or scalability.

## III. Representative Execution Paths or Usage Workflows

### Scenario or Workflow Title 1
(code block)
Execution behavior explanation.
Performance characteristics explanation.
Resource usage characteristics explanation.

### Scenario or Workflow Title 2
(code block)
Execution behavior explanation.
Performance characteristics explanation.
Resource usage characteristics explanation.

(Include as many scenarios or workflows as needed, but each must follow the same pattern.)

## IV. Key Invariants and Mental Model
- Core invariants or constraints that must always hold for the system to function correctly.
- A concise mental model for reasoning about this topic during debugging, optimization, or system design.
- Common pitfalls and best practices.

```

