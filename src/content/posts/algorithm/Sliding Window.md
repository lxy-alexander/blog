---
title: Sliding Window
published: 2026-01-20
description: "Sliding Window"
image: ""
tags: ["algorithm","Sliding Window"]
category: algorithm
draft: false
---

>**Sliding Window** is a way to look at a small part of data and move it forward one step at a time, instead of starting over each time.

# Fixed-length sliding window

### [1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

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

-   `1 <= s.length <= 105`
-   `s` consists of lowercase English letters.
-   `1 <= k <= s.length`

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



### [643. Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)

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

-   `n == nums.length`
-   `1 <= k <= n <= 105`
-   `-104 <= nums[i] <= 104`

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





### [2461. Maximum Sum of Distinct Subarrays With Length K](https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/)

You are given an integer array `nums` and an integer `k`. Find the maximum subarray sum of all the subarrays of `nums` that meet the following conditions:

-   The length of the subarray is `k`, and
-   All the elements of the subarray are **distinct**.

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

-   `1 <= k <= nums.length <= 105`
-   `1 <= nums[i] <= 105`

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



### [30. Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

You are given a string `s` and an array of strings `words`. All the strings of `words` are of **the same length**.

A **concatenated string** is a string that exactly contains all the strings of any permutation of `words` concatenated.

-   For example, if `words = ["ab","cd","ef"]`, then `"abcdef"`, `"abefcd"`, `"cdabef"`, `"cdefab"`, `"efabcd"`, and `"efcdab"` are all concatenated strings. `"acdbef"` is not a concatenated string because it is not the concatenation of any permutation of `words`.

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

-   `1 <= s.length <= 104`
-   `1 <= words.length <= 5000`
-   `1 <= words[i].length <= 30`
-   `s` and `words[i]` consist of lowercase English letters.



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



### [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)

Given a binary array `nums` and an integer `k`, return *the maximum number of consecutive* `1`*'s in the array if you can flip at most* `k` `0`'s.↳

**Example 1:**

```
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
```

**Example 2:**

```
Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
```

**Constraints:**

-   `1 <= nums.length <= 105`
-   `nums[i]` is either `0` or `1`.
-   `0 <= k <= nums.length`

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        ans = 0
        l = 0
        cnt = 0
        for r, x in enumerate(nums):
            cnt += int(not x)
            while cnt > k:
                cnt -= int(not nums[l])
                l += 1
            ans = max(ans, r - l + 1)
        return ans
        
```



```python
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        // you can flip at most k 0, it means the window has at most k zero, it can be 0, 1, to k.
        int ans = 0;
        int cnt = 0;
        int l = 0;
        int n = nums.size();
        for (int r = 0; r < n; r++) {
            // accumulate the amount of zero
            cnt += !nums[r];
            while (cnt > k) {
                cnt -= !nums[l];
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```









# Variable-length sliding window

>   Variable-length sliding windows are mainly divided into three categories: finding the longest subarray, finding the shortest subarray, and finding the number of subarrays.
>
>    A sliding window is equivalent to maintaining a queue . Moving the right pointer can be seen as enqueuing , and moving the left pointer can be seen as dequeuing .
>
>   <  ≤  → direct sliding window
>   ==     → at most K − at most (K − 1)
>   ≥      → total − at most (K − 1)





## At Most

:::important

At Most（至多）

-   ≤ K
-   **单调性最好**
-   滑动窗口首选

**“至多”之所以单调性最好，是因为扩张和收缩对合法性的影响方向是完全相反且确定的。**

-   at most K distinct
-   at most K occurrences
-   sum ≤ K

:::

### [1446. Consecutive Characters](https://leetcode.com/problems/consecutive-characters/)

The **power** of the string is the maximum length of a non-empty substring that contains only one unique character.

Given a string `s`, return *the **power** of* `s`.

**Example 1:**

```
Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.
```

**Example 2:**

```
Input: s = "abbcccddddeeeeedcba"
Output: 5
Explanation: The substring "eeeee" is of length 5 with the character 'e' only.
```

**Constraints:**

-   `1 <= s.length <= 500`
-   `s` consists of only lowercase English letters.

```python
class Solution:
    def maxPower(self, s: str) -> int:
        ans = 0
        l = 0
        cnt = defaultdict(int)
        for r, c in enumerate(s):
            cnt[c] += 1
            while len(cnt) > 1:
                cnt[s[l]] -= 1
                if cnt[s[l]] == 0: #因为你的check条件是size，所以当这个map中的某个元素的个数为0，要把它移出，不移除会让这个check条件一直符合，最终会让l一直++，到最后会超过s的长度，导致溢出。
                    del cnt[s[l]]
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```

```c++
class Solution {
public:
    int maxPower(string s) {
        int ans = 0;
        int l = 0;
        int n = s.size();
        map<char, int> cnt;
        for (int r = 0; r < n; r++) {
            cnt[s[r]]++;
            while (cnt.size() > 1) {
                cnt[s[l]]--;
                if (cnt[s[l]] == 0) {
                    cnt.erase(s[l]);
                }
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};
```







### [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)

Given a binary array `nums` and an integer `k`, return *the maximum number of consecutive* `1`*'s in the array if you can flip at most* `k` `0`'s.

**Example 1:**

```
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
```

**Example 2:**

```
Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
```

**Constraints:**

-   `1 <= nums.length <= 105`
-   `nums[i]` is either `0` or `1`.
-   `0 <= k <= nums.length`

```python
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        ans = 0
        l = 0
        cnt = 0
        for r, x in enumerate(nums):
            cnt += int(not x)
            while cnt > k:
                cnt -= int(not nums[l])
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```

```c++
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        // you can flip at most k 0, it means the window has at most k zero, it can be 0, 1, to k.
        int ans = 0;
        int cnt = 0;
        int l = 0;
        int n = nums.size();
        for (int r = 0; r < n; r++) {
            // accumulate the amount of zero
            cnt += !nums[r];
            while (cnt > k) {
                cnt -= !nums[l];
                l++;
            }
            ans = max(ans, r - l + 1);
        }
        return ans;
    }
};

```





### [[LeetCode\] 340. Longest Substring with At Most K Distinct Characters](https://www.cnblogs.com/grandyang/p/5351347.html) (at most k distinct elements )

 Given a string, find the length of the longest substring T that contains at most *k* distinct characters.

Example 1:

```
Input: s = "eceba", k = 2
Output: 3
Explanation: T is "ece" which its length is 3.
```

Example 2:

```
Input: s = "aa", k = 1
Output: 2
Explanation: T is "aa" which its length is 2.
```

```c++
class Solution {
public:
    int lengthOfLongestSubstringKDistinct(string s, int k) {
        int ans = 0;
        map<char, int> cnt;
        int l = 0;
        int n = s.size();
        for (int r = 0; r < n; r++) {
            cnt[s[r]]++;
            while (cnt.size() > k) {
                cnt[s[l]]--;
                if (cnt[s[l]] == 0) {
                    cnt.erase(s[l]);
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
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        ans = 0
        cnt = defaultdict(int)
        l = 0
        for r, c in enumerate(s):
            cnt[c] += 1 # cnt[c] is an integer (the frequency of character c)
            while len(cnt) > k: // we need check how many distinct characters are currently in the sliding window. 
                cnt[s[l]] -= 1
                if cnt[s[l]] == 0:
                    del cnt[s[l]]
                l += 1
            ans = max(ans, r - l + 1)
        return ans
```



### [904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)  (at most 2 distinct elements )

You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array `fruits` where `fruits[i]` is the **type** of fruit the `ith` tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

-   You only have **two** baskets, and each basket can only hold a **single type** of fruit. There is no limit on the amount of fruit each basket can hold.
-   Starting from any tree of your choice, you must pick **exactly one fruit** from **every** tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
-   Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

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

-   `1 <= fruits.length <= 105`
-   `0 <= fruits[i] < fruits.length`

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





### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) （at most 1 occurrence）

Given a string `s`, find the length of the **longest** **substring** without duplicate characters.

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
        map<char, int> cnt;
        int n = s.size();
        for (int r = 0; r < n; r++) {
            cnt[s[r]]++;
            
            // if the occurrences of the current character is greater than one, it is not valid substring, we need shrink the left boundary to make it valid
            while (cnt[s[r]] > 1) {
                cnt[s[l]]--;
                if (cnt[s[l]] == 0) {
                    cnt.erase(s[l]);
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





### [3090. Maximum Length Substring With Two Occurrences](https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/) （at most 2 occurrences）

Given a string `s`, return the **maximum** length of a substring such that it contains *at most two occurrences* of each character.

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

-   `2 <= s.length <= 100`
-   `s` consists only of lowercase English letters.

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





## Exactly K Distinct

:::important

Exactly K Distinct Elements

:::

### [1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/)

Given an array of integers `nums` and an integer `k`. A continuous subarray is called **nice** if there are `k` odd numbers on it.

Return *the number of **nice** sub-arrays*.

**Example 1:**

```
Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].
```

**Example 2:**

```
Input: nums = [2,4,6], k = 1
Output: 0
Explanation: There are no odd numbers in the array.
```

**Example 3:**

```
Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
Output: 16
```

 

**Constraints:**

-   `1 <= nums.length <= 50000`
-   `1 <= nums[i] <= 10^5`
-   `1 <= k <= nums.length`



```c++
class Solution {
public:
    int numberOfSubarrays(vector<int>& nums, int k) {
        return atMostNumberOfSubarrays(nums, k)
            - atMostNumberOfSubarrays(nums, k - 1);

    }

private:
    int atMostNumberOfSubarrays(vector<int>& nums, int k) {
        int ans = 0;
        int oddCnt = 0;
        int left = 0;
        int n = nums.size();
        for (int right = 0; right < n; right++) {
            oddCnt += nums[right] & 1;
            while (oddCnt > k) {
                oddCnt -= nums[left] & 1;
                left++;
            }
            ans += right - left + 1;
        }
        return ans;
    }
};

```

```python
# Why exactly K cannot be handled directly with a sliding window
# Imagine the current window contains exactly k odd numbers.
# When the right pointer moves:
# If the new number is odd → the count becomes k + 1 (invalid)
# If the new number is even → the count stays k (still valid)
# When the left pointer moves:
# If the removed number is odd → the count becomes k − 1 (invalid)
# If the removed number is even → the count stays k (still valid)
class Solution:
    """
    exactly K = at most K - at most (k - 1)
    """

    def numberOfSubarrays(self, nums: List[int], k: int) -> int: 
        # This nested function exists only to help solve this problem.
        # share the context via closures
        def atMost(k):
            ans = l = cnt = 0
            for r, x in enumerate(nums):
                cnt += x & 1
                while cnt > k:
                    cnt -= nums[l] & 1
                    l += 1
                # when the while loop finishes, cnt must be less than or equal to k, it means the window [l, r] satisfies the constraint. so all subarrays ending at r and starting from any index between l and r are valid, so we add r - l + 1.
                ans += r - l + 1
            return ans
        
        # To compute the number of subarrays with exactly k odd numbers by using the numbers of subarrays with at most k odd numbers to substracting the one with at most k - 1 odd numbers.
        return atMost(k) - atMost(k - 1)
                
```







### [2062. Count Vowel Substrings of a String](https://leetcode.com/problems/count-vowel-substrings-of-a-string/) （exactly including 5 all vowels 这个和其他exactly k 不一样）

:::note

For “contains all 5 vowels”, expanding `r` is monotonic (valid stays valid), but shrinking `l` can break validity, so the classic “invalid → shrink” template doesn’t directly apply.

那还能用滑窗吗？

可以，只是换一种形式（合法时缩）：

-   不合法 → 扩 r 去变合法
-   合法 → 尝试缩 l，并统计答案

扩到合法，再缩到刚好不合法，再继续扩

:::

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

-   `1 <= word.length <= 100`
-   `word` consists of lowercase English letters only.

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
        
        int i = 0;
        while (i < n) {
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
                        ans += j - r; // [l, r] is a valid substring, so [l, r], [l, r+1], [l, r+2], ..., [l, j-1] are also valid.

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





### [992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) （Exactly K Different Integers）

Given an integer array `nums` and an integer `k`, return *the number of **good subarrays** of* `nums`.

A **good array** is an array where the number of different integers in that array is exactly `k`.

-   For example, `[1,2,3,1,2]` has `3` different integers: `1`, `2`, and `3`.

A **subarray** is a **contiguous** part of an array.

**Example 1:**

```
Input: nums = [1,2,1,2,3], k = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2]
```

**Example 2:**

```
Input: nums = [1,2,1,3,4], k = 3
Output: 3
Explanation: Subarrays formed with exactly 3 different integers: [1,2,1,3], [2,1,3], [1,3,4].
```

 

**Constraints:**

-   `1 <= nums.length <= 2 * 104`
-   `1 <= nums[i], k <= nums.length`

```c++
class Solution {
public:
    int subarrayAtMostKDistinct(vector<int>& nums, int k) {
        int n = nums.size();
        int ans = 0;
        map<int, int>cnt;
        int l = 0;
        for (int r = 0; r < n; r++) {
            cnt[nums[r]]++;
            while (cnt.size() > k) { // distinc elements is greater than k
                cnt[nums[l]]--;
                if (cnt[nums[l]] == 0) {
                    cnt.erase(nums[l]);
                }
                l++;
            }
            /**
            [l, r] 中最多有 k 个不同元素

            [l, r]
            [l+1, r]
            [l+2, r]
            ...
            [r, r]

            r - l + 1 个
            **/
            ans += r - l + 1;
        }
        return ans;

    }
    int subarraysWithKDistinct(vector<int>& nums, int k) {
        return subarrayAtMostKDistinct(nums, k) - subarrayAtMostKDistinct(nums, k - 1);
    }
};
```

```python
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(K):
            cnt = defaultdict(int)
            l = 0
            res = 0

            for r, x in enumerate(nums):
                cnt[x] += 1

                while len(cnt) > K:
                    cnt[nums[l]] -= 1
                    if cnt[nums[l]] == 0:
                        del cnt[nums[l]]
                    l += 1

                res += (r - l + 1)
            return res
        
        return atMost(k) - atMost(k - 1)

```





### [713. Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/)  (less than k (belong to at most))

Given an array of integers `nums` and an integer `k`, return *the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than* `k`.

**Example 1:**

```
Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
```

**Example 2:**

```
Input: nums = [1,2,3], k = 0
Output: 0
```

 

**Constraints:**

-   `1 <= nums.length <= 3 * 104`
-   `1 <= nums[i] <= 1000`
-   `0 <= k <= 106`

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # 当 k ≤ 1 时，不可能存在 product < k 的子数组，所以答案必为 0。
        if k <= 1:
            return 0
        ans, product, l = 0, 1, 0
        for r, x in enumerate(nums):
            product *= x
            while product >= k:
                product //= nums[l] # /会使变成float
                l += 1
            ans += r - l + 1
        return ans
```









## At Least

:::important

At Least（至少）

-   ≥ K
-   **不稳定**
-   通常转化为 at most

满足 ≥ K 的数量  = 所有可能的数量 − 不满足 ≥ K（也就是 ≤ K−1）的数量

`at least K = total − at most (K − 1) ` 这种还没遇到这样的题目

:::

### [2962. Count Subarrays Where Max Element Appears at Least K Times](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/)

You are given an integer array `nums` and a **positive** integer `k`.

Return *the number of subarrays where the **maximum** element of* `nums` *appears **at least*** `k` *times in that subarray.*

A **subarray** is a contiguous sequence of elements within an array.

 

**Example 1:**

```
Input: nums = [1,3,2,3,3], k = 2
Output: 6
Explanation: The subarrays that contain the element 3 at least 2 times are: [1,3,2,3], [1,3,2,3,3], [3,2,3], [3,2,3,3], [2,3,3] and [3,3].
```

**Example 2:**

```
Input: nums = [1,4,2,1], k = 3
Output: 0
Explanation: No subarray contains the element 4 at least 3 times.
```

 

**Constraints:**

-   `1 <= nums.length <= 105`
-   `1 <= nums[i] <= 106`
-   `1 <= k <= 105`

```python
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = 0
        l = 0
        cnt = defaultdict(int)
        mx = max(nums)
        n = len(nums)
        for r, x in enumerate(nums):
            cnt[x] += 1
            while cnt[mx] >= k:
                ans += n - r
                cnt[nums[l]] -= 1
                l += 1
        return ans
```





## Greedy + Two Pointers + run-based scanning

### [1839. Longest Substring Of All Vowels in Order](https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/)

A string is considered **beautiful** if it satisfies the following conditions:

-   Each of the 5 English vowels (`'a'`, `'e'`, `'i'`, `'o'`, `'u'`) must appear **at least once** in it.
-   The letters must be sorted in **alphabetical order** (i.e. all `'a'`s before `'e'`s, all `'e'`s before `'i'`s, etc.).

For example, strings `"aeiou"` and `"aaaaaaeiiiioou"` are considered **beautiful**, but `"uaeio"`, `"aeoiu"`, and `"aaaeeeooo"` are **not beautiful**.

Given a string `word` consisting of English vowels, return *the **length of the longest beautiful substring** of* `word`*. If no such substring exists, return* `0`.

A **substring** is a contiguous sequence of characters in a string.

**Example 1:**

```
Input: word = "aeiaaioaaaaeiiiiouuuooaauuaeiu"
Output: 13
Explanation: The longest beautiful substring in word is "aaaaeiiiiouuu" of length 13.
```

**Example 2:**

```
Input: word = "aeeeiiiioooauuuaeiou"
Output: 5
Explanation: The longest beautiful substring in word is "aeiou" of length 5.
```

**Example 3:**

```
Input: word = "a"
Output: 0
Explanation: There is no beautiful substring, so return 0.
```



```c++
class Solution {
public:
    int longestBeautifulSubstring(string word) {
        int ans = 0;
        int l = 0, r = 0;
        int n = word.size();
        while (r < n) {
            if (word[r] != 'a') {
                r++;
                continue;
            }
            int l = r;
            r += 1;
            int type = 1;
            while (r < n && word[r] >= word[r - 1]) {
                if (word[r] > word[r - 1]) {
                    type++;
                }
                r++;
            }
            if (type == 5)
                ans = max(ans, r - l);
        }
        return ans;
    }
};
```

```python
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        ans, l, r = 0, 0, 0
        n = len(word)
        while r < n:
            if word[r] != 'a':
                r += 1
                continue
            l = r
            r += 1
            type = 1 # initially we set the type to be 1, because the first char is 'a'
            while r < n and word[r] >= word[r - 1]: # if next char is greater than or equal to previous one, it means it is sorted in alphabetical order
                if word[r] != word[r - 1]:
                    type += 1
                r += 1

            if type == 5:
                ans = max(ans, r - l)
        return ans

```



