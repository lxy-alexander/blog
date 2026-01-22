---
title: Algorithm
published: 2026-01-20
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

>**Sliding Window** is a way to look at a small part of data and move it forward one step at a time, instead of starting over each time.

### Fixed-length sliding window

#### [1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

Given a string `s` and an integer `k`, return *the maximum number of vowel letters in any substring of* `s` *with length* `k`.

**Vowel letters** in English are `'a'`, `'e'`, `'i'`, `'o'`, and `'u'`.

 

**Example 1:**

```
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.
```

**Example 2:**

```
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.
```

**Example 3:**

```
Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.
```

 

**Constraints:**

- `1 <= s.length <= 105`
- `s` consists of lowercase English letters.
- `1 <= k <= s.length`

```cpp
class Solution {
public:
    bool isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    
    int maxVowels(string s, int k) {
        // const std::set<char> vowelSet{'a', 'e', 'i', 'o', 'u'};
        int n = s.size();
        int vowelCnt = 0;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (isVowel(s[i])) vowelCnt++;
            if (i < k - 1) continue;
            ans = std::max(ans, vowelCnt);
            if (isVowel(s[i - k + 1])) vowelCnt--;
        }
        return ans;
    }
    
};
```

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        ans = 0
        vowel_cnt = 0

        for i, c in enumerate(s):
            if c in "aieou":
                vowel_cnt += 1
            if i < k - 1:
                continue
            ans = max(ans, vowel_cnt)
            if s[i - k + 1] in "aieou":
                vowel_cnt -= 1
        return ans
```



[643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)

You are given an integer array `nums` consisting of `n` elements, and an integer `k`.

Find a contiguous subarray whose **length is equal to** `k` that has the maximum average value and return *this value*. Any answer with a calculation error less than `10-5` will be accepted.

 

**Example 1:**

```
Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
```

**Example 2:**

```
Input: nums = [5], k = 1
Output: 5.00000
```

 

**Constraints:**

- `n == nums.length`
- `1 <= k <= n <= 105`
- `-104 <= nums[i] <= 104`

```c++
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double ans = -1e18;
        double sum = 0;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            if (i < k - 1) continue;
            ans = std::max(ans, sum / k);
            sum -= nums[i - k + 1];
        }
        return ans;
    }
};
```

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        ans = float('-inf')
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            if i < k - 1:
                continue
            ans = max(ans, sum / k)
            sum -= nums[i - k + 1]
        return ans
```





#### [2461. Maximum Sum of Distinct Subarrays With Length K](https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/)

You are given an integer array `nums` and an integer `k`. Find the maximum subarray sum of all the subarrays of `nums` that meet the following conditions:

- The length of the subarray is `k`, and
- All the elements of the subarray are **distinct**.

Return *the maximum subarray sum of all the subarrays that meet the conditions**.* If no subarray meets the conditions, return `0`.

*A **subarray** is a contiguous non-empty sequence of elements within an array.*

 

**Example 1:**

```
Input: nums = [1,5,4,2,9,9,9], k = 3
Output: 15
Explanation: The subarrays of nums with length 3 are:
- [1,5,4] which meets the requirements and has a sum of 10.
- [5,4,2] which meets the requirements and has a sum of 11.
- [4,2,9] which meets the requirements and has a sum of 15.
- [2,9,9] which does not meet the requirements because the element 9 is repeated.
- [9,9,9] which does not meet the requirements because the element 9 is repeated.
We return 15 because it is the maximum subarray sum of all the subarrays that meet the conditions
```

**Example 2:**

```
Input: nums = [4,4,4], k = 3
Output: 0
Explanation: The subarrays of nums with length 3 are:
- [4,4,4] which does not meet the requirements because the element 4 is repeated.
We return 0 because no subarrays meet the conditions.
```

 

**Constraints:**

- `1 <= k <= nums.length <= 105`
- `1 <= nums[i] <= 105`

```c++
class Solution {
public:
    long long maximumSubarraySum(vector<int>& nums, int k) {
        long long ans = 0;
        long long sum = 0;
        std::unordered_map<int, int> map; // stores the frequency of elements in the window
        int left = 0;
        int n = nums.size();
        for (int right = 0; right < n; right++) {
            int x = nums[right];
            map[x]++;
            sum += x;
            left = right - k + 1;
            if (left < 0) continue;
            if (map.size() == k) ans = std::max(ans, sum); // is used to check whether duplicates exist
            int o = nums[left];
            sum -= o;
            map[o]--;
            if (map[o] == 0) map.erase(o);
        }
        return ans;
    }
};
```

```python
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        ans, sum, left = 0, 0, 0
        d = defaultdict(int)

        for right, x in enumerate(nums):
            d[x] += 1
            sum += x
            left = right - k + 1
            if left < 0:
                continue
            if len(d) == k: # d stores the frequency of elements in the window
                ans = max(ans, sum)
            
            out = nums[left]
            sum -= out
            d[out] -= 1
            if d[out] == 0:
                del d[out] 
        return ans         
```



#### [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

You are given a string `s` and an array of strings `words`. All the strings of `words` are of **the same length**.

A **concatenated string** is a string that exactly contains all the strings of any permutation of `words` concatenated.

- For example, if `words = ["ab","cd","ef"]`, then `"abcdef"`, `"abefcd"`, `"cdabef"`, `"cdefab"`, `"efabcd"`, and `"efcdab"` are all concatenated strings. `"acdbef"` is not a concatenated string because it is not the concatenation of any permutation of `words`.

Return an array of *the starting indices* of all the concatenated substrings in `s`. You can return the answer in **any order**.

 

**Example 1:**

**Input:** s = "barfoothefoobarman", words = ["foo","bar"]

**Output:** [0,9]

**Explanation:**

The substring starting at 0 is `"barfoo"`. It is the concatenation of `["bar","foo"]` which is a permutation of `words`.
The substring starting at 9 is `"foobar"`. It is the concatenation of `["foo","bar"]` which is a permutation of `words`.

**Example 2:**

**Input:** s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]

**Output:** []

**Explanation:**

There is no concatenated substring.

**Example 3:**

**Input:** s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]

**Output:** [6,9,12]

**Explanation:**

The substring starting at 6 is `"foobarthe"`. It is the concatenation of `["foo","bar","the"]`.
The substring starting at 9 is `"barthefoo"`. It is the concatenation of `["bar","the","foo"]`.
The substring starting at 12 is `"thefoobar"`. It is the concatenation of `["the","foo","bar"]`.

 

**Constraints:**

- `1 <= s.length <= 104`
- `1 <= words.length <= 5000`
- `1 <= words[i].length <= 30`
- `s` and `words[i]` consist of lowercase English letters.



// ❌ 不在 → [l, r) → 不加 1     ✅ 在 → [l, r] → 加 1

```c++
class Solution {
public:
    vector<int> findSubstring(string s, vector<string>& words) {
        vector<int> ans;
        int n = s.size();
        int m = words.size();
        int wordLen = words[0].size();
        int winLen = m * wordLen;

        map<string, int> map1;
        for (const string& word : words) {
            map1[word]++;
        }

      
        for (int i = 0; i < wordLen; i++) {
            map<string, int> map2;

            for (int j = i; j + wordLen <= n; j += wordLen) {
                // 1. 进窗口
                string w = s.substr(j, wordLen);
                map2[w]++;

                // 2. 窗口还没满
                if (j + wordLen - i < winLen) continue;

                // 3. 判断
                if (map1 == map2)
                    ans.push_back(j + wordLen - winLen);

                // 4. 出窗口
                string out = s.substr(j + wordLen - winLen, wordLen);
                map2[out]--;
                if (map2[out] == 0) map2.erase(out);
            }
        }

        return ans;
    }
};

```



```python
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        ans = []
        n = len(s)
        word_len = len(words[0])
        win_len = len(words) * word_len
        d1 = defaultdict(int)

        for w in words:
            d1[w] += 1

        for i in range(word_len):
            d2 = defaultdict(int)

            for j in range(i + word_len, n + 1, word_len):
                w = s[j-word_len:j]
                d2[w] += 1
                
                if j - i < win_len:
                    continue

                if d1 == d2:
                    ans.append(j - win_len)

                out = s[j - win_len : j - win_len + word_len]
                d2[out] -= 1
                if d2[out] == 0:
                    del d2[out]
        return ans
```







### Variable-length sliding window

> Variable-length sliding windows are mainly divided into three categories: finding the longest subarray, finding the shortest subarray, and finding the number of subarrays.
>
>  A sliding window is equivalent to maintaining a queue . Moving the right pointer can be seen as enqueuing , and moving the left pointer can be seen as dequeuing .
>



#### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Given a string `s`, find the length of the **longest** **substring** without duplicate characters.

给定一个字符串 s，求没有重复字符的最长子串的长度。

 

**Example 1:**

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3. Note that "bca" and "cab" are also correct answers.
```

**Example 2:**

```
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
```

**Example 3:**

```
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int ans = 0;
        int l = 0; // l is leftbound and r is right bound
        map<char, int> map; // create a map
        int n = s.size();
        for (int r = 0; r < n; r++) {
            // If map contains the current character and this charater is inside the sliding window
            if (map.contains(s[r]) && map[s[r]] >= l) {
                l = map[s[r]] + 1; // satisfy above the condition, it will move the left pointer(that is left boundary) to next postion.
            }
            ans = max(ans, r - l + 1); // you need compare with the newest window size and old value, and then update the answer
            map[s[r]] = r; // update the character's index
        }
        return ans;
    }
};
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        l = 0
        cnt = defaultdict(int)
        for r, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] > 1:
                cnt[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

```



:::important

| 题目条件      | 位置法 | 计数法 |
| ------------- | ------ | ------ |
| 不允许重复    | ✅      | ✅      |
| 最多出现 1 次 | ✅      | ✅      |
| 最多出现 2 次 | ❌      | ✅      |
| 最多出现 k 次 | ❌      | ✅      |
| 至多 k 种元素 | ❌      | ✅      |
| 至少 k 次     | ❌      | ✅      |
| 子数组和 ≤ k  | ❌      | ✅      |
| 单一最近冲突  | ✅      | ❌      |

:::



:::note

 At most 2 occurrences

:::



#### [3090. Maximum Length Substring With Two Occurrences](https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/)

Given a string `s`, return the **maximum** 

length

 of a substring such that it contains *at most two occurrences* of each character.

 

**Example 1:**

**Input:** s = "bcbbbcba"

**Output:** 4

**Explanation:**

The following substring has a 

length

 of 4 and contains at most two 

occurrences

 of each character: `"bcbbbcba"`.

**Example 2:**

**Input:** s = "aaaa"

**Output:** 2

**Explanation:**

The following substring has a 

length

 of 2 and contains at most two 

occurrences

 of each character: `"aaaa"`.

 

**Constraints:**

- `2 <= s.length <= 100`
- `s` consists only of lowercase English letters.

```c++
class Solution {
public:
    int maximumLengthSubstring(string s) {
        int ans = 0;
        map<char, int> cnt;
        int l = 0;
        int n = s.size();
        for (int r = 0; r < n; r++) {
            cnt[s[r]]++;
            while (cnt[s[r]] > 2) { // if the count exceed 2, the window becomes invalid. we will shrink the window from the left
                cnt[s[l]]--; // decreses the count of s[l]
                l++; // move the left boundary to next position
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```

```python
class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        ans = 0
        d = defaultdict(int)
        l = 0
        for r, c in enumerate(s):
            d[c] += 1
            while d[c] > 2:
                d[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

```



:::note

At most k distinct elements

:::

#### [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)

You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array `fruits` where `fruits[i]` is the **type** of fruit the `ith` tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

- You only have **two** baskets, and each basket can only hold a **single type** of fruit. There is no limit on the amount of fruit each basket can hold.
- Starting from any tree of your choice, you must pick **exactly one fruit** from **every** tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
- Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

Given the integer array `fruits`, return *the **maximum** number of fruits you can pick*.

 

**Example 1:**

```
Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.
```

**Example 2:**

```
Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can pick from trees [1,2,2].
If we had started at the first tree, we would only pick from trees [0,1].
```

**Example 3:**

```
Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: We can pick from trees [2,3,2,2].
If we had started at the first tree, we would only pick from trees [1,2].
```

 

**Constraints:**

- `1 <= fruits.length <= 105`
- `0 <= fruits[i] < fruits.length`

```c++
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        int ans = 0;
        map<int, int> cnt; // count each fruit type inside the window
        int l = 0;
        int n = fruits.size();
        for (int r = 0; r < n; r++) {
            cnt[fruits[r]]++; // do increment
            while (cnt.size() > 2) { // if the size of the fruit type exceed 2, we need shrink the window from the left
                cnt[fruits[l]]--;
                if (cnt[fruits[l]] == 0) {
                    cnt.erase(fruits[l]);
                }
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```

```python
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        ans = 0
        cnt = defaultdict(int)
        l = 0
        for r, x in enumerate(fruits):
            cnt[x] += 1
            while len(cnt) > 2: # reflects number of fruit types
                cnt[fruits[l]] -= 1 
                if cnt[fruits[l]] == 0:
                    del cnt[fruits[l]]
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```





:::note

At least k distinct elements, You cannot simply shrink when invalid

:::

### [2062. Count Vowel Substrings of a String](https://leetcode.com/problems/count-vowel-substrings-of-a-string/)

A **substring** is a contiguous (non-empty) sequence of characters within a string.

A **vowel substring** is a substring that **only** consists of vowels (`'a'`, `'e'`, `'i'`, `'o'`, and `'u'`) and has **all five** vowels present in it.

Given a string `word`, return *the number of **vowel substrings** in* `word`.

 

**Example 1:**

```
Input: word = "aeiouu"
Output: 2
Explanation: The vowel substrings of word are as follows (underlined):
- "aeiouu"
- "aeiouu"
```

**Example 2:**

```
Input: word = "unicornarihan"
Output: 0
Explanation: Not all 5 vowels are present, so there are no vowel substrings.
```

**Example 3:**

```
Input: word = "cuaieuouac"
Output: 7
Explanation: The vowel substrings of word are as follows (underlined):
- "cuaieuouac"
- "cuaieuouac"
- "cuaieuouac"
- "cuaieuouac"
- "cuaieuouac"
- "cuaieuouac"
- "cuaieuouac"
```

**Constraints:**

- `1 <= word.length <= 100`
- `word` consists of lowercase English letters only.

```c++
class Solution {
public:
    bool isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i'
            || c == 'o' || c == 'u';
    }

    int countVowelSubstrings(string word) {
        int ans = 0;
        int n = word.size();

        for (int i = 0; i < n; ) {
            if (!isVowel(word[i])) {
                i++;
                continue;
            }

            int j = i;
            while (j < n && isVowel(word[j])) j++;

            if (j - i >= 5) {
                unordered_map<char, int> cnt;
                int l = i;

                for (int r = i; r < j; r++) {
                    cnt[word[r]]++;
                    while (cnt.size() == 5) {
                        ans += j - r;
                        cnt[word[l]]--;
                        if (cnt[word[l]] == 0)
                            cnt.erase(word[l]);
                        l++;
                    }
                }
            }
            i = j;
        }

        return ans;
    }
};

```

```python
class Solution:
    def countVowelSubstrings(self, word: str) -> int:
        ans = 0
        vowels = set('aeiou')
        i = 0
        while i < len(word):
            if word[i] not in vowels:
                i += 1
                continue
            j = i
            while j < len(word) and word[j] in vowels:
                j += 1
            if j - i >= len(vowels):
                cnt = defaultdict(int)
                l = i
                for r in range(i, j):
                    cnt[word[r]] += 1
                    while len(cnt) == len(vowels):
                        ans += j - r
                        cnt[word[l]] -= 1
                        if cnt[word[l]] == 0:
                            del cnt[word[l]]
                        l += 1
            i = j

        return ans 

```

```python

class Solution:
    def countVowelSubstrings(self, word: str) -> int:
        """
        当窗口 [l, r] 已经包含 a e i o u：
        [l, r]
        [l, r+1]
        [l, r+2]
        ……
        全部都是合法子串
        所以是 len(s) - r
        """
        ans = 0
        for s in re.findall(r'[aeiou]+', word):
            if len(s) < 5:
                continue
            cnt = defaultdict(int)
            l = 0
            for r, c in enumerate(s):
                cnt[c] += 1
                while len(cnt) == 5:
                    ans += len(s) - r
                    cnt[s[l]] -= 1
                    if cnt[s[l]] == 0:
                        del cnt[s[l]]
                    l += 1
        return ans
```



### At least k occurrences

- **2516** — Take K of Each Character From Left and Right↳
- **395** — Longest Substring with At Least K Repeating Characters↳

### Subarray sum / cost ≤ k

- **209** — Minimum Size Subarray Sum
- **1208** — Get Equal Substrings Within Budget↳
- **1004** — Max Consecutive Ones III

### “Replace / delete / flip at most k”

- **2024** — Maximize the Confusion of an Exam↳
- **1493** — Longest Subarray of 1’s After Deleting One Element↳
- **487** — Max Consecutive Ones II *(premium)*

### Range / window aggregation

- **2271** — Maximum White Tiles Covered by a Carpet
- **2106** — Maximum Fruits Harvested After at Most K Steps↳
- **2555** — Maximize Win From Two Segments
- **632** — Smallest Range Covering Elements from K Lists









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

