---
title: "beam search"
published: 2026-04-30
description: "beam search"
image: ""
tags: ["llm","beam search"]
category: llm
draft: false
lang: ""
createdAt: "2026-04-30T04:09:27.802.186613903Z"
---

# Beam Search 

Beam Search (束搜索) is a heuristic decoding(启发式解码) algorithm that keeps the top-k candidate sequences at each step.

## 1. Core Idea

Beam Search (束搜索) keeps multiple candidate paths instead of one, balancing between greedy and exhaustive search.

-   Beam Width (束宽) = k → number of sequences kept at each step
-   At each step:
    1.  Expand all candidates
    2.  Compute probabilities
    3.  Keep top-k sequences

$$
Score = \prod_{t=1}^{T} P(w_t | w_1, ..., w_{t-1})
$$

One-sentence summary: Beam Search (束搜索) keeps top-k paths to reduce early mistakes compared to Greedy Search (贪心搜索).

## 2. Example (Step-by-step)

Beam Width (束宽) = 2

### 1) Step 1

| Word | Probability |
| ---- | ----------- |
| I    | 0.5         |
| You  | 0.3         |
| We   | 0.2         |

Keep top 2:

-   I (0.5)
-   You (0.3)

### 2) Step 2

| Sequence | New Word | Score |
| -------- | -------- | ----- |
| I        | like     | 0.30  |
| I        | hate     | 0.10  |
| You      | like     | 0.15  |
| You      | are      | 0.12  |

Keep:

-   I like (0.30)
-   You like (0.15)

### 3) Step 3

| Sequence | New Word | Score |
| -------- | -------- | ----- |
| I like   | cats     | 0.15  |
| I like   | dogs     | 0.12  |
| You like | cats     | 0.09  |
| You like | dogs     | 0.045 |

Final result:

-   I like cats (0.15)

One-sentence summary: Beam Search (束搜索) evaluates multiple sequences and selects the globally better sentence.

## 3. Python Implementation

### 1) Simple Beam Search

```python
import heapq

def beam_search():
    beam_width = 2
    
    # Step 1
    sequences = [("I", 0.5), ("You", 0.3)]
    
    # Step 2 probabilities
    step2 = {
        "I": [("like", 0.6), ("hate", 0.2)],
        "You": [("like", 0.5), ("are", 0.4)]
    }
    
    new_sequences = []
    
    for seq, score in sequences:
        for word, prob in step2[seq]:
            new_seq = seq + " " + word
            new_score = score * prob
            new_sequences.append((new_seq, new_score))
    
    # keep top-k
    sequences = heapq.nlargest(beam_width, new_sequences, key=lambda x: x[1])
    
    print("After Step 2:", sequences)
    # Output: [('I like', 0.3), ('You like', 0.15)]
    
    return sequences

beam_search()
```

One-sentence summary: The implementation uses a heap (堆) to keep top-k sequences efficiently.

## 4. Greedy vs Beam Search

| Method                   | Idea                | Risk                    |
| ------------------------ | ------------------- | ----------------------- |
| Greedy Search (贪心搜索) | pick best each step | may miss global optimum |
| Beam Search (束搜索)     | keep top-k paths    | more computation        |
| BFS (广度优先搜索)       | explore all         | too expensive           |

One-sentence summary: Beam Search (束搜索) is a trade-off between Greedy Search (贪心搜索) and BFS (广度优先搜索).

## 5. Complexity

Time Complexity (时间复杂度):

$$
O(k \cdot V \cdot T)
$$

-   k = beam width
-   V = vocabulary size (词表大小)
-   T = sequence length

One-sentence summary: Beam Search (束搜索) scales linearly with beam width (束宽) and sequence length.
