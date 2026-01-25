---
title: Algorithm Study Plan
published: 2026-01-20
description: "Algorithm"
image: ""
tags: ["algorithm", "Algorithm Study Plan"]
category: algorithm
draft: false
---

# Plan

<img src="https://raw.githubusercontent.com/lxy-alexander/pico/main/typora/image-20260120171400651.png" alt="image-20260120171400651" style="zoom: 33%;" /> 

按照专题刷题，而不是随机刷题。同一个专题，一个套路可以解决多个题目，刷题效率高。此外，这能让你从不同的角度去观察、思考同一个算法，从而深刻地理解算法的本质。
螺旋上升式学习：先完成难度分 ≤1700 的题目。把各个题单、各个知识点的基础题刷一遍，再刷更难的题目。难度分低的题目一般只会考察一个知识点，而难度分高的题目会同时考察多个知识点。

https://leetcode.cn/discuss/post/3141566/ru-he-ke-xue-shua-ti-by-endlesscheng-q3yd/



# Degree of Completion

## Sliding Window: 2026.1.19 - 2026.1.23 

-   At most k distinct elements (by using `len(cnt)` to check if the window is valid)
-   At most k occurrences of each element (by using `cnt[s[r]] > k` to check if the window is valid.  If  adding s[r] makes the window not valid, we need shrink the window to make it valid again)
-   Exactly K Distinct Elements = At most(K) - At most(K-1)  
-   Some variant questions: 
    -   Including exactly 5 vowels(other character are not vowels) need to divide and conquer firstly. and then do the at most
    -   exactly 

