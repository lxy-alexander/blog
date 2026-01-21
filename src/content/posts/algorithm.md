---
title: Algorithm
published: 2025-12-18
description: "Algorithm"
image: "./cover.jpeg"
tags: ["Blogging", "Algorithm"]
category: Guides
draft: false
---

# Algorithm

<img src="https://raw.githubusercontent.com/lxy-alexander/pico/main/typora/image-20260120171400651.png" alt="image-20260120171400651" style="zoom: 33%;" /> 

按照专题刷题，而不是随机刷题。同一个专题，一个套路可以解决多个题目，刷题效率高。此外，这能让你从不同的角度去观察、思考同一个算法，从而深刻地理解算法的本质。
螺旋上升式学习：先完成难度分 ≤1700 的题目。把各个题单、各个知识点的基础题刷一遍，再刷更难的题目。难度分低的题目一般只会考察一个知识点，而难度分高的题目会同时考察多个知识点。

https://leetcode.cn/discuss/post/3141566/ru-he-ke-xue-shua-ti-by-endlesscheng-q3yd/



## Sliding Window

### Fixed-length sliding window







## Bitwise

> Bitwise algorithms solve problems by directly manipulating the binary representation of integers using bit-level operations.

### Check if a Number Is Odd

```cpp
bool isOdd(int n) {
    return (n & 1) != 0;
}

def is_odd(n: int) -> bool:
    return (n & 1) != 0
```

Time complexity is O(1) because it uses a single bit operation.
Space complexity is O(1) because no extra memory is used.

------

### Count Set Bits (Brian Kernighan’s Method)

```cpp
int countSetBits(int n) {
    int count = 0;
    while (n != 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}

def count_set_bits(n: int) -> int:
    count = 0
    while n != 0:
        n &= (n - 1)
        count += 1
    return count
```

Time complexity is O(k), where k is the number of set bits in the integer.
Space complexity is O(1) because only constant extra space is used.

------

### Find the Unique Element Using XOR

```cpp
int singleNumber(const vector<int>& nums) {
    int result = 0;
    for (int x : nums) {
        result ^= x;
    }
    return result;
}
def single_number(nums: list[int]) -> int:
    result = 0
    for x in nums:
        result ^= x
    return result
```

Time complexity is O(n), where n is the number of elements in the array.
Space complexity is O(1) because no additional data structures are required.

------

### Check if a Number Is a Power of Two

```cpp
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

Time complexity is O(1) since it uses a constant number of operations.
Space complexity is O(1) because it uses constant extra space.

