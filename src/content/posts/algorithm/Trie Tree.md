---
title: "Trie Tree"
published: 2026-06-24
description: "Trie Tree"
image: ""
tags: ["algorithm","Trie Tree"]
category: algorithm
draft: false
lang: ""
createdAt: "2026-06-24T15:27:03.461.812247710Z"

---

# Trie / Prefix Tree

Trie is a tree data structure used to efficiently store and search strings by their prefixes.

In LeetCode, trie problems usually appear when the problem asks for prefix matching, word search, dictionary lookup, autocomplete, or replacing words by roots.

Trie is useful when the problem involves many string queries, repeated prefix checks, longest or shortest prefix matching, wildcard search, or searching words on a board.

| Type                       | Problem                                             | Why It Matters                                       |
| -------------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| Basic Trie                 | **208. Implement Trie Prefix Tree**                 | Core Trie template: `insert`, `search`, `startsWith` |
| Prefix Matching            | **648. Replace Words**                              | Shortest prefix matching with a dictionary           |
| Trie + DFS / Wildcard      | **211. Design Add and Search Words Data Structure** | Trie with DFS and `.` wildcard search                |
| Autocomplete / Suggestions | **1268. Search Suggestions System**                 | Prefix search with lexicographic suggestions         |
| Counting Trie              | **1804. Implement Trie II Prefix Tree**             | Trie with word count, prefix count, and deletion     |
| Trie + Backtracking        | **212. Word Search II**                             | Trie combined with grid DFS / backtracking           |



# Basic Trie

**Problem:** 208. Implement Trie Prefix Tree

This is the core Trie template problem. You need to implement three operations: `insert`, `search`, and `startsWith`.

Use this problem to understand how a Trie stores words character by character.

```text
Each node represents one character path.
children stores next characters.
end marks whether this node is the end of a complete word.
```

### [208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

A [**trie**](https://en.wikipedia.org/wiki/Trie) (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

-   `Trie()` Initializes the trie object.
-   `void insert(String word)` Inserts the string `word` into the trie.
-   `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
-   `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.

 

**Example 1:**

```
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
```

 

**Constraints:**

-   `1 <= word.length, prefix.length <= 2000`
-   `word` and `prefix` consist only of lowercase English letters.
-   At most `3 * 104` calls **in total** will be made to `insert`, `search`, and `startsWith`.

```python
class Trie:
    # root --'a'--> 节点 --'p'--> 节点 --'p'--> 节点(is_word=True)
    def __init__(self):
        self.children = {}
        self.end = False
        
    def insert(self, word: str) -> None:
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = Trie()
            cur = cur.children[c]
        cur.end = True
        
    def search(self, word: str) -> bool:
        cur = self
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.end

    def startsWith(self, prefix: str) -> bool:
        cur = self
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```





------

# Prefix Matching

**Problem:** 648. Replace Words

This type is about finding a valid prefix from a dictionary. For each word, walk through the Trie from left to right. If you meet a node marked as a complete word, return that prefix immediately.

This is usually used for **shortest prefix matching**.

Key idea:

```text
Insert all dictionary roots into Trie.
For each word, find the shortest root that is its prefix.
If found, replace the word.
Otherwise, keep the original word.
```





### [648. Replace Words](https://leetcode.com/problems/replace-words/)

In English, we have a concept called **root**, which can be followed by some other word to form another longer word - let's call this word **derivative**. For example, when the **root** `"help"` is followed by the word `"ful"`, we can form a derivative `"helpful"`.

Given a `dictionary` consisting of many **roots** and a `sentence` consisting of words separated by spaces, replace all the derivatives in the sentence with the **root** forming it. If a derivative can be replaced by more than one **root**, replace it with the **root** that has **the shortest length**.

Return *the `sentence`* after the replacement.

 

**Example 1:**

```
Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"
```

**Example 2:**

```
Input: dictionary = ["a","b","c"], sentence = "aadsfasf absbs bbab cadsfafs"
Output: "a a b c"
```

 

**Constraints:**

-   `1 <= dictionary.length <= 1000`
-   `1 <= dictionary[i].length <= 100`
-   `dictionary[i]` consists of only lower-case letters.
-   `1 <= sentence.length <= 106`
-   `sentence` consists of only lower-case letters and spaces.
-   The number of words in `sentence` is in the range `[1, 1000]`
-   The length of each word in `sentence` is in the range `[1, 1000]`
-   Every two consecutive words in `sentence` will be separated by exactly one space.
-   `sentence` does not have leading or trailing spaces.

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.end = False
        self.word = None

class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        root = Trie()
        for word in dictionary:
            cur = root
            for c in word:
                if c not in cur.children:
                    cur.children[c] = Trie()
                cur = cur.children[c]
            cur.end = True
            cur.word = word

        def findWord(word):
            cur = root
            for c in word:
                if c not in cur.children:
                    return word
                cur = cur.children[c]
                if cur.end:
                    return cur.word
            return word

        ans = []
        
        for word in sentence.split():
            ans.append(findWord(word))
        
        return ' '.join(ans) 
```



------

# Trie + DFS / Wildcard

**Problem:** 211. Design Add and Search Words Data Structure

This type combines Trie with DFS. The special part is that the search word may contain The dot can match any single character, so when you see `.`, you need to try all children.

Key idea:

```text
Normal character: follow that child only.
Dot character: try every child with DFS.
At the end, check whether current node is a complete word.
```





### [211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:

-   `WordDictionary()` Initializes the object.
-   `void addWord(word)` Adds `word` to the data structure, it can be matched later.
-   `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.

 

**Example:**

```
Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
```

 

**Constraints:**

-   `1 <= word.length <= 25`
-   `word` in `addWord` consists of lowercase English letters.
-   `word` in `search` consist of `'.'` or lowercase English letters.
-   There will be at most `2` dots in `word` for `search` queries.
-   At most `104` calls will be made to `addWord` and `search`.



```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False


class WordDictionary:

    def __init__(self):
        self.root = Trie()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = Trie()
            cur = cur.children[c]
        cur.is_end = True

    def search(self, word: str) -> bool:
        def dfs(i, node):
            # 1）边界：word 已经匹配完
            if i == len(word):
                return node.is_end

            # 2）当前字符
            c = word[i]

            # 3）找选择
            if c == '.':
                choices = node.children.values()
            else:
                if c not in node.children:
                    return False
                choices = [node.children[c]]

            # 4）尝试选择 As long as there is a successful match behind one of the roads, I will succeed here.
            for child in choices:
                if dfs(i + 1, child):
                    return True

            # 5）所有选择都失败
            return False

        return dfs(0, self.root)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```











------

# Autocomplete / Suggestions

**Problem:** 1268. Search Suggestions System

This type is used for search suggestions or autocomplete.

Given a search word, after typing each character, return up to three products with that prefix.

Key idea:

```text
Sort products first.
Insert products into Trie.
At each Trie node, store at most 3 lexicographically smallest words.
Then each prefix query can directly return the stored suggestions.
```

Alternative method:

```text
Sorting + binary search is often simpler than Trie for this problem.
```

------

# Counting Trie

**Problem:** 1804. Implement Trie II Prefix Tree

This type adds counters to Trie nodes.

You need to support:

```text
insert
countWordsEqualTo
countWordsStartingWith
erase
```

Each node usually stores:

```text
prefix_count
word_count
```

Key idea:

```text
When inserting a word, increase prefix_count on every node along the path.
At the final node, increase word_count.

When erasing a word, decrease those counts.
```

------

# Trie + Backtracking

**Problem:** 212. Word Search II

This type combines Trie with grid DFS / backtracking.

Instead of searching each word separately on the board, insert all words into a Trie first.
Then start DFS from every cell and follow Trie paths.

If the current board path is not a prefix in the Trie, stop immediately.

Key idea:

```text
Trie stores all target words.
DFS explores the board.
Trie helps prune invalid paths early.
When a Trie node is a complete word, add it to the answer.
```







------

# Recommended Order

```text
208 -> 648 -> 211 -> 1268 -> 1804 -> 212
```

This order goes from basic operations to prefix matching, wildcard DFS, autocomplete, counting, and finally grid backtrackin
