---
title: Bitwise
published: 2026-01-20
description: "Bitwise"
image: ""
tags: ["algorithm", "Bitwise"]
category: Guides
draft: false

---

> Bitwise algorithms solve problems by directly manipulating the binary representation of integers using bit-level operations.

# Common Bitwise Alogrithm

## Check if a Number Is Odd

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

## Count Set Bits (Brian Kernighan’s Method)

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

## Find the Unique Element Using XOR

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

## Check if a Number Is a Power of Two

```cpp
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
```

Time complexity is O(1) since it uses a constant number of operations.
Space complexity is O(1) because it uses constant extra space.