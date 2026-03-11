---
title: "Python re"
published: 2026-03-09
description: "Python re"
image: ""
tags: ["python","Python re"]
category: python
draft: false
lang: ""
---

# **I. Python `re` — Regular Expressions**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
A <span style="color:#E8600A;font-weight:700">Regular Expression (正则表达式)</span> is a <span style="color:#2980B9">pattern (模式)</span> used to find and manipulate text. Think of it as a <span style="color:#E8600A;font-weight:700">"super-powered search"</span> that can match patterns, not just exact words. Python's <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re</code> module gives you tools to use regex.
</div>

------

# **II. Pattern Syntax — The Complete Reference **

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Every regex pattern is built from three kinds of building blocks: <span style="color:#E8600A;font-weight:700">Literals (字面量)</span> that match themselves, <span style="color:#E8600A;font-weight:700">Metacharacters (元字符)</span> that have special meaning, and <span style="color:#E8600A;font-weight:700">Quantifiers (量词)</span> that control repetition. Learn these 30-odd symbols and you can write any pattern. </div>

## 1. Literals and Metacharacters (字面量与元字符)

### 1) Plain literals (普通字面量)

Most characters match themselves exactly.

```python
import re

# Literal match — 'cat' matches exactly the string "cat"
print(re.search(r'cat', 'I have a cat'))        # match
print(re.search(r'cat', 'I have a CAT'))        # None  (case-sensitive by default)
print(re.search(r'cat', 'concatenate'))         # match (found inside)
```

### 2) The 14 metacharacters (14个元字符)

These characters have special meaning and must be **escaped** with `\` to match literally:

```
. ^ $ * + ? { } [ ] \ | ( )
import re

# Matching a literal dot — must escape it
text = "price: $3.99"

print(re.search(r'3.99',  text))    # matches "3.99" BUT also "3X99" (dot = any char!)
print(re.search(r'3\.99', text))    # matches ONLY "3.99"  ← correct

# Matching a literal backslash
print(re.search(r'C:\\Users', r'C:\Users'))   # matches C:\Users
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Always use raw strings <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">r'pattern'</code> for regex patterns.</span> Without <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">r</code>, Python processes <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\n</code> as newline before the regex engine sees it. With <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">r'\n'</code>, the regex engine receives the literal two characters <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\n</code> and interprets them as "newline character".</div>

------

## 2. The Dot `.` — Any character (任意字符)

`.` matches **any single character** except a newline `\n` (unless `re.DOTALL` flag is set).

```python
import re

# . matches exactly ONE character (any except \n)
print(re.findall(r'c.t', 'cat cut c t c\nt coot'))
# → ['cat', 'cut', 'c t']   ('c\nt' skipped — \n not matched by dot)
# Note: 'coot' not matched — dot matches exactly 1 char

# With re.DOTALL, dot matches newline too
text = "first\nsecond"
print(re.search(r'first.second',  text))             # None
print(re.search(r'first.second',  text, re.DOTALL))  # match
```

------

## 3. Anchors — Position matchers (锚点 — 位置匹配)

Anchors match **positions**, not characters.

### 1) `^` and `$` — Start and end of string/line

```python
import re

text = "hello world"

print(re.search(r'^hello', text))   # match  — 'hello' is at start
print(re.search(r'^world', text))   # None   — 'world' is NOT at start
print(re.search(r'world$', text))   # match  — 'world' is at end
print(re.search(r'hello$', text))   # None

# With re.MULTILINE: ^ and $ match start/end of EACH LINE
multiline = "line1\nline2\nline3"
print(re.findall(r'^\w+', multiline, re.MULTILINE))
# → ['line1', 'line2', 'line3']

print(re.findall(r'\w+$', multiline, re.MULTILINE))
# → ['line1', 'line2', 'line3']
```

### 2) `\b` and `\B` — Word boundaries (单词边界)

`\b` matches the **boundary between a word character and a non-word character**.

```python
import re

# \b matches word boundary — prevents partial matches
print(re.findall(r'\bcat\b', 'cat cats concatenate scatter'))
# → ['cat']   (only the standalone word)

print(re.findall(r'cat',     'cat cats concatenate scatter'))
# → ['cat', 'cat', 'cat', 'cat']  (too many!)

# \B matches NON-word boundary (inside a word)
print(re.findall(r'\Bcat\B', 'cat cats concatenate'))
# → ['cat']   (only the 'cat' inside 'concatenate')
```

### 3) `\A`, `\Z` — Absolute start/end of string (字符串绝对首尾)

```python
import re

# \A and \Z are NOT affected by re.MULTILINE — always match string start/end
text = "line1\nline2"

print(re.search(r'\Aline1', text))   # match — absolute start
print(re.search(r'\Aline2', text))   # None  — line2 is NOT at absolute start
print(re.search(r'line2\Z', text))   # match — absolute end
```

------

## 4. Character Classes `[ ]` (字符类)

### 1) Basic character class (基本字符类)

A character class matches **one character** that is any of the listed characters.

```python
import re

# [aeiou] matches any single vowel
print(re.findall(r'[aeiou]', 'hello world'))
# → ['e', 'o', 'o']

# [a-z] matches any lowercase letter (range syntax)
print(re.findall(r'[a-z]+', 'Hello World 123'))
# → ['ello', 'orld']

# [A-Za-z0-9] matches any alphanumeric
print(re.findall(r'[A-Za-z0-9]+', 'foo_bar-123!'))
# → ['foo', 'bar', '123']

# [0-9] is equivalent to \d
print(re.findall(r'[0-9]+', 'abc 123 def 456'))
# → ['123', '456']
```

### 2) Negated character class `[^ ]` (否定字符类)

`[^...]` matches any character **NOT** in the class.

```python
import re

# [^aeiou] matches any consonant (non-vowel)
print(re.findall(r'[^aeiou\s]+', 'hello world'))
# → ['h', 'll', 'w', 'rld']

# [^0-9] matches any non-digit character
print(re.findall(r'[^0-9]+', 'abc123def456'))
# → ['abc', 'def']

# Strip all non-alphanumeric characters
cleaned = re.sub(r'[^A-Za-z0-9]', '', 'Hello, World! 123')
print(cleaned)   # → HelloWorld123
```

### 3) Special sequences inside `[ ]`

```python
import re

# Inside [], most metacharacters lose special meaning
# - (dash) is literal if first, last, or escaped
print(re.findall(r'[-+*/]', '3+4-2*1/5'))  # → ['+', '-', '*', '/']

# ^ is literal unless it is the FIRST character
print(re.findall(r'[a^b]', 'a^b c'))       # → ['a', '^', 'b']  (literal ^)

# ] must be escaped or placed first
print(re.findall(r'[]a-z]', 'a]b'))        # → ['a', ']', 'b']
```

------

## 5. Predefined Character Classes (预定义字符类)

These are shorthand for common character sets:

| Shorthand                                                    | Equivalent       | Meaning                         |
| ------------------------------------------------------------ | ---------------- | ------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\d</code> | `[0-9]`          | Any digit (数字)                |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\D</code> | `[^0-9]`         | Any non-digit (非数字)          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\w</code> | `[A-Za-z0-9_]`   | Word character (单词字符)       |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\W</code> | `[^A-Za-z0-9_]`  | Non-word character (非单词字符) |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\s</code> | `[ \t\n\r\f\v]`  | Whitespace (空白字符)           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">\S</code> | `[^ \t\n\r\f\v]` | Non-whitespace (非空白字符)     |

```python
import re

text = "Hello, World! 42 items at $3.99 each.\n"

print(re.findall(r'\d+',  text))  # → ['42', '3', '99']
print(re.findall(r'\w+',  text))  # → ['Hello', 'World', '42', 'items', 'at', '3', '99', 'each']
print(re.findall(r'\s+',  text))  # → [' ', ' ', ' ', ' ', ' ', '\n']
print(re.findall(r'\W+',  text))  # → [', ', '! ', ' ', ' $', '.', '\n']

# Combining: \w+ matches whole words
print(re.findall(r'\b\w{5}\b', text))  # words of exactly 5 chars
# → ['Hello', 'World', 'items']
```

------

## 6. Quantifiers — Repetition (量词 — 重复)

### 1) Basic quantifiers (基本量词)

| Quantifier                                                   | Meaning                        |
| ------------------------------------------------------------ | ------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">*</code> | 0 or more (零次或多次)         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">+</code> | 1 or more (一次或多次)         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">?</code> | 0 or 1 (零次或一次，可选)      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">{n}</code> | Exactly n times (恰好n次)      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">{n,}</code> | n or more times (n次或更多)    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">{n,m}</code> | Between n and m times (n到m次) |

```python
import re

s = "colour   color   colouur"

print(re.findall(r'colou?r',    s))   # ? → u is optional
# → ['colour', 'color']

print(re.findall(r'colou*r',    s))   # * → 0 or more u's
# → ['colour', 'color', 'colouur']

print(re.findall(r'colou+r',    s))   # + → 1 or more u's
# → ['colour', 'colouur']

print(re.findall(r'colou{2}r',  s))   # exactly 2 u's
# → ['colouur']

print(re.findall(r'colou{1,2}r',s))   # 1 or 2 u's
# → ['colour', 'colouur']

# Phone number: exactly 10 digits
print(re.findall(r'\d{10}', '1234567890 12345'))
# → ['1234567890']
```

### 2) Greedy vs Non-greedy (贪婪 vs 非贪婪)

By default, quantifiers are <span style="color:#E8600A;font-weight:700">greedy (贪婪)</span> — they match **as much as possible**. Adding `?` makes them <span style="color:#E8600A;font-weight:700">non-greedy (非贪婪/懒惰)</span> — they match **as little as possible**.

```python
import re

html = "<b>bold</b> and <i>italic</i>"

# Greedy: .* expands as far right as possible
print(re.findall(r'<.*>',  html))
# → ['<b>bold</b> and <i>italic</i>']   ← one huge match (too greedy)

# Non-greedy: .*? stops at the FIRST >
print(re.findall(r'<.*?>',  html))
# → ['<b>', '</b>', '<i>', '</i>']        ← each tag separately

# Extracting content between tags
print(re.findall(r'<b>(.*?)</b>', html))
# → ['bold']

# More examples
text = '"first" and "second"'
print(re.findall(r'".*"',  text))   # → ['"first" and "second"']  greedy
print(re.findall(r'".*?"', text))   # → ['"first"', '"second"']   non-greedy
```

| Pattern                                                      | Type       | Matches                   |
| ------------------------------------------------------------ | ---------- | ------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.*</code> | Greedy     | As many chars as possible |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.*?</code> | Non-greedy | As few chars as possible  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.+</code> | Greedy     | 1+ chars, maximum         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.+?</code> | Non-greedy | 1+ chars, minimum         |

------

## 7. Groups — Capturing and Non-capturing (分组 — 捕获与非捕获)

### 1) Capturing group `( )` (捕获组)

Groups serve two purposes: **grouping** for quantifiers, and **capturing** the matched text.

```python
import re

# Grouping: (ab)+ repeats the whole "ab"
print(re.findall(r'(ab)+', 'ab abab ababab'))
# → ['ab', 'ab', 'ab']  (returns last captured group)

# Capturing: extract the content inside ()
dates = "2024-01-15, 2023-12-31, 2025-06-01"
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', dates))
# → [('2024', '01', '15'), ('2023', '12', '31'), ('2025', '06', '01')]
#   ↑ each match returns a tuple of all captured groups

# .group() on a Match object
m = re.search(r'(\d{4})-(\d{2})-(\d{2})', '2024-01-15')
print(m.group(0))   # → 2024-01-15  (entire match)
print(m.group(1))   # → 2024        (group 1)
print(m.group(2))   # → 01          (group 2)
print(m.group(3))   # → 15          (group 3)
```

### 2) Named group `(?P<name>...)` (命名捕获组)

```python
import re

# Named groups — access by name instead of index
pattern = r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
m = re.search(pattern, '2024-01-15')

print(m.group('year'))    # → 2024
print(m.group('month'))   # → 01
print(m.group('day'))     # → 15
print(m.groupdict())      # → {'year': '2024', 'month': '01', 'day': '15'}

# Named groups in re.sub — backreference by name
result = re.sub(
    r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})',
    r'\g<day>/\g<month>/\g<year>',    # reorder: DD/MM/YYYY
    '2024-01-15'
)
print(result)   # → 15/01/2024
```

### 3) Non-capturing group `(?:...)` (非捕获组)

When you need grouping for quantifiers but **don't** want the group in your results:

```python
import re

# Without (?:...) — capturing group pollutes findall results
print(re.findall(r'(\d+)(?:px|em|rem)', '12px 3em 100rem'))
# → ['12', '3', '100']   ← only numbers, units NOT captured ✅

# With capturing group — units would also appear
print(re.findall(r'(\d+)(px|em|rem)', '12px 3em 100rem'))
# → [('12', 'px'), ('3', 'em'), ('100', 'rem')]  ← units captured too

# (?:...) for grouping quantifiers
print(re.findall(r'(?:ha)+', 'hahaha haha ha h'))
# → ['hahaha', 'haha', 'ha']   (group 'ha' as a unit for +)
```

### 4) Backreferences `\1` `\2` (反向引用)

Refer to a previously captured group **within the same pattern**.

```python
import re

# Find repeated words
text = "the the quick brown fox fox jumps"
print(re.findall(r'\b(\w+)\s+\1\b', text))
# → ['the', 'fox']   (\1 refers back to group 1)

# Find doubled characters
print(re.findall(r'(.)\1', 'aabcddee'))
# → ['a', 'd', 'e']

# HTML tag matching: opening and closing tags must match
html = "<h1>Title</h1> <h2>Subtitle</h2>"
print(re.findall(r'<(\w+)>(.*?)</\1>', html))
# → [('h1', 'Title'), ('h2', 'Subtitle')]
#   \1 ensures the closing tag matches the opening tag
```

------

## 8. Lookahead and Lookbehind — Zero-width assertions (零宽断言)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">Lookaround (环视断言)</span> matches a position based on what is around it, <span style="color:#2980B9">without consuming characters (不消耗字符)</span>. They are "zero-width" — the match position doesn't advance. </div>

### 1) Positive lookahead `(?=...)` (正向先行断言)

"Match X only if followed by Y" — Y is NOT included in the match.

```python
import re

# Match a number only if followed by "px"
print(re.findall(r'\d+(?=px)', '12px 3em 100px 5rem'))
# → ['12', '100']   (px NOT included in results)

# Match word only if followed by a colon
text = "name: Alice age: 30 city: NYC"
print(re.findall(r'\w+(?=:)', text))
# → ['name', 'age', 'city']

# Password validation: must contain a digit
import re
def has_digit(pw): return bool(re.search(r'(?=.*\d)', pw))
print(has_digit("abc123"))   # → True
print(has_digit("abcdef"))   # → False
```

### 2) Negative lookahead `(?!...)` (负向先行断言)

"Match X only if NOT followed by Y"

```python
import re

# Match a number only if NOT followed by "px"
print(re.findall(r'\d+(?!px)\b', '12px 3em 100px 5rem'))
# → ['3', '5']

# Match 'foo' not followed by 'bar'
print(re.findall(r'foo(?!bar)', 'foobar foobaz foo'))
# → ['foo', 'foo']   ('foobar' excluded, 'foobaz' and 'foo' included)
```

### 3) Positive lookbehind `(?<=...)` (正向后行断言)

"Match X only if preceded by Y" — Y is NOT included in the match.

```python
import re

# Match digits only if preceded by '$'
prices = "items: $10, €20, £30, $50"
print(re.findall(r'(?<=\$)\d+', prices))
# → ['10', '50']   ($ NOT included in results)

# Match word after a colon and space
text = "name: Alice, city: NYC, age: 30"
print(re.findall(r'(?<=: )\w+', text))
# → ['Alice', 'NYC', '30']
```

### 4) Negative lookbehind `(?<!...)` (负向后行断言)

"Match X only if NOT preceded by Y"

```python
import re

# Match digits NOT preceded by '$'
prices = "items: $10, 20, $50, 100"
print(re.findall(r'(?<!\$)\b\d+\b', prices))
# → ['20', '100']

# Match 'ing' not preceded by 'run'
words = "running swimming singing"
print(re.findall(r'(?<!run)ning\b', words))
# → ['ning', 'ning']   (swim→ming yes, sing→ning yes, runNING excluded)
```

### 5) Lookaround summary table (环视断言总结)

| Syntax                                                       | Name                           | Meaning           |
| ------------------------------------------------------------ | ------------------------------ | ----------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?=Y)</code> | Positive lookahead (正向先行)  | Followed by Y     |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?!Y)</code> | Negative lookahead (负向先行)  | NOT followed by Y |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?<=Y)</code> | Positive lookbehind (正向后行) | Preceded by Y     |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?<!Y)</code> | Negative lookbehind (负向后行) | NOT preceded by Y |

------

## 9. Alternation `|` — OR operator (或运算符)

```python
import re

# | matches either the left or right pattern
print(re.findall(r'cat|dog|fish', 'I have a cat and a dog'))
# → ['cat', 'dog']

# With groups: (cat|dog) scopes the alternation
print(re.findall(r'(cat|dog)s?', 'cats dogs cat dog'))
# → ['cat', 'dog', 'cat', 'dog']

# Alternation of longer patterns
log = "ERROR: disk full  WARNING: low memory  INFO: started"
print(re.findall(r'ERROR|WARNING|INFO', log))
# → ['ERROR', 'WARNING', 'INFO']

# Order matters: first match wins
print(re.search(r'cat|catch', 'I catch cats'))   # matches 'cat' (not 'catch'!)
print(re.search(r'catch|cat', 'I catch cats'))   # matches 'catch' ← correct order
```

------

## 10. Flags — Modifying match behavior (标志位)

### 1) All flags (所有标志位)

| Flag (short)                                                 | Flag (long)                                                  | Effect                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.I</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.IGNORECASE</code> | Case-insensitive matching (忽略大小写)          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.M</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.MULTILINE</code> | `^`/`$` match each line (多行模式)              |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.S</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.DOTALL</code> | `.` matches `\n` too (点号匹配换行)             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.X</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.VERBOSE</code> | Allow whitespace/comments in pattern (详细模式) |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.A</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.ASCII</code> | `\w \d \s` match ASCII only (ASCII模式)         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.L</code> | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.LOCALE</code> | Locale-dependent matching (本地化模式)          |

### 2) `re.IGNORECASE` (re.I)

```python
import re

print(re.findall(r'hello', 'Hello HELLO hello', re.I))
# → ['Hello', 'HELLO', 'hello']

# Case-insensitive word boundary
print(re.findall(r'\bpython\b', 'Python PYTHON python', re.IGNORECASE))
# → ['Python', 'PYTHON', 'python']
```

### 3) `re.MULTILINE` (re.M)

```python
import re

log = """ERROR: disk full
WARNING: low memory
ERROR: timeout
INFO: done"""

# Without re.M: ^ only matches start of entire string
print(re.findall(r'^ERROR.*',  log))
# → ['ERROR: disk full']

# With re.M: ^ matches start of EACH line
print(re.findall(r'^ERROR.*',  log, re.M))
# → ['ERROR: disk full', 'ERROR: timeout']
```

### 4) `re.DOTALL` (re.S)

```python
import re

html = "<div>\n  <p>Hello</p>\n</div>"

# Without re.S: . does not match \n
print(re.search(r'<div>.*</div>',  html))          # None

# With re.S: . matches everything including \n
print(re.search(r'<div>.*</div>',  html, re.S))    # match
print(re.search(r'<div>.*?</div>', html, re.S).group())
# → <div>\n  <p>Hello</p>\n</div>
```

### 5) `re.VERBOSE` (re.X) — Readable complex patterns (可读的复杂模式)

```python
import re

# Without re.X — hard to read
email_pattern_compact = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# With re.X — add whitespace and comments freely
email_pattern_verbose = re.compile(r'''
    ^                       # start of string
    [a-zA-Z0-9._%+-]+       # local part (user name)
    @                       # @ symbol
    [a-zA-Z0-9.-]+          # domain name
    \.                      # literal dot
    [a-zA-Z]{2,}            # top-level domain (2+ letters)
    $                       # end of string
''', re.VERBOSE)

print(email_pattern_verbose.match('user@example.com'))   # match
print(email_pattern_verbose.match('bad@'))               # None
```

### 6) Combining flags (组合标志位)

```python
import re

# Combine with | (bitwise OR)
text = "Hello\nWorld"
print(re.findall(r'^.+$', text, re.M | re.I))
# re.M → ^ and $ per line
# re.I → case-insensitive
# → ['Hello', 'World']

# Inline flags in the pattern (?flags) — scoped to pattern
print(re.findall(r'(?i)hello', 'Hello HELLO hello'))
# → ['Hello', 'HELLO', 'hello']

# Inline flags for part of pattern
print(re.findall(r'(?i:hello) world', 'HELLO world hello World'))
# → ['HELLO world']   (only 'hello' is case-insensitive, 'world' is not)
```

------

# **III. The `re` Module API — Complete Reference**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re</code> module has two usage modes: <span style="color:#E8600A;font-weight:700">① module-level functions</span> like <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.search()</code> (convenient for one-off use) and <span style="color:#E8600A;font-weight:700">② compiled Pattern objects</span> via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.compile()</code> (preferred when the same pattern is used repeatedly — avoids recompilation overhead). </div>

## 1. `re.compile()` — Pre-compile a pattern (预编译模式)

```python
import re

# compile() returns a Pattern object
pattern = re.compile(r'\d{4}-\d{2}-\d{2}', re.IGNORECASE)

# Call methods on the Pattern object (same names as module-level functions)
print(pattern.search('date: 2024-01-15'))
print(pattern.findall('from 2024-01-01 to 2024-12-31'))
# → ['2024-01-01', '2024-12-31']

# Pattern attributes
print(pattern.pattern)    # → \d{4}-\d{2}-\d{2}
print(pattern.flags)      # → 34  (2 = default + 32 = IGNORECASE)
print(pattern.groups)     # → 0   (no capturing groups)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Module-level functions like <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.search(pattern, string)</code> use an internal cache of the last 512 compiled patterns. For hot loops, prefer explicit <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.compile()</code> to guarantee no cache misses and to make intent clear.</div>

------

## 2. `re.search()` — Find first match anywhere (查找第一个匹配)

Returns a <span style="color:#E8600A;font-weight:700">Match object</span> if found anywhere in the string, or `None`.

```python
import re

text = "The price is $42.99 for 3 items"

m = re.search(r'\$(\d+\.\d{2})', text)
if m:
    print(m.group())    # → $42.99   (full match)
    print(m.group(1))   # → 42.99    (group 1 — no $)
    print(m.start())    # → 13       (start index)
    print(m.end())      # → 19       (end index)
    print(m.span())     # → (13, 19) (start, end)
    print(m.string)     # → "The price is $42.99 for 3 items"  (original)
```

------

## 3. `re.match()` — Match at string start (从字符串开头匹配)

<span style="color:#C0392B;font-weight:600">Warning: `re.match()` only matches at the BEGINNING of the string — NOT the same as `re.search()`!</span>

```python
import re

# match() — only succeeds if pattern starts at position 0
print(re.match(r'\d+', '123 abc'))    # match   — starts at position 0
print(re.match(r'\d+', 'abc 123'))    # None    — 'abc' is not \d+
print(re.search(r'\d+', 'abc 123'))   # match   — search finds it anywhere

# match() with ^ is redundant (both restrict to start)
print(re.match(r'hello', 'hello world'))    # match
print(re.match(r'hello', 'say hello'))      # None

# Practical: validate that a string is ENTIRELY a number
def is_integer(s):
    return bool(re.match(r'^\d+$', s))

print(is_integer("12345"))    # → True
print(is_integer("123a5"))    # → False
```

------

## 4. `re.fullmatch()` — Match entire string (匹配整个字符串)

Requires the pattern to match the **complete** string from start to end.

```python
import re

# fullmatch() equivalent to match() with ^ and $ anchors
print(re.fullmatch(r'\d+', '12345'))     # match   — entire string is digits
print(re.fullmatch(r'\d+', '123abc'))    # None    — not ALL digits
print(re.fullmatch(r'\d+', '  123  '))   # None    — spaces don't match \d

# Validate formats completely
ip_pattern  = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
zip_pattern = re.compile(r'\d{5}(-\d{4})?')
email_pat   = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

tests = ['192.168.1.1', '12345', 'user@example.com', 'bad_input']
for t in tests:
    results = {
        'ip':    bool(ip_pattern.fullmatch(t)),
        'zip':   bool(zip_pattern.fullmatch(t)),
        'email': bool(email_pat.fullmatch(t)),
    }
    print(f"{t:<25} → {results}")
```

------

## 5. `re.findall()` — Find all matches (查找所有匹配)

Returns a **list** of all non-overlapping matches.

```python
import re

text = "2024-01-15, 2023-12-31, 2025-06-01"

# No groups → returns list of strings
print(re.findall(r'\d{4}-\d{2}-\d{2}', text))
# → ['2024-01-15', '2023-12-31', '2025-06-01']

# One group → returns list of group contents
print(re.findall(r'(\d{4})-\d{2}-\d{2}', text))
# → ['2024', '2023', '2025']   (only the year group)

# Multiple groups → returns list of tuples
print(re.findall(r'(\d{4})-(\d{2})-(\d{2})', text))
# → [('2024', '01', '15'), ('2023', '12', '31'), ('2025', '06', '01')]
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">The return type of <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">findall()</code> changes based on groups: no groups → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">List[str]</code>, one group → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">List[str]</code>, multiple groups → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">List[tuple]</code>. This is a common source of bugs.</span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">finditer()</code> for consistent Match objects.</div>

------

## 6. `re.finditer()` — Iterator of Match objects (匹配对象迭代器)

Returns an **iterator** of Match objects. More powerful than `findall()` because each Match has `.start()`, `.end()`, `.group()`, etc.

```python
import re

text = "Alice scored 95, Bob scored 87, Carol scored 100"

for m in re.finditer(r'(\w+) scored (\d+)', text):
    name  = m.group(1)
    score = int(m.group(2))
    print(f"{name}: {score} pts  | span={m.span()}")
# → Alice: 95 pts  | span=(0, 16)
# → Bob: 87 pts    | span=(18, 32)
# → Carol: 100 pts | span=(34, 49)

# Collect all spans for highlighting
positions = [(m.start(), m.end()) for m in re.finditer(r'\d+', text)]
print(positions)   # → [(13, 15), (28, 30), (44, 47)]
```

------

## 7. `re.sub()` — Substitute matches (替换匹配)

### 1) Basic substitution (基本替换)

```python
import re

text = "Hello   World   Python"

# Replace multiple spaces with single space
result = re.sub(r'\s+', ' ', text)
print(result)   # → Hello World Python

# count parameter: replace only first N occurrences
result = re.sub(r'\s+', ' ', text, count=1)
print(result)   # → Hello World   Python  (only first replaced)
```

### 2) Backreferences in replacement (替换中的反向引用)

```python
import re

# \1, \2 refer to captured groups in the replacement string
# Reformat date from YYYY-MM-DD to DD/MM/YYYY
dates = "Born: 2024-01-15, Died: 2099-12-31"
result = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1', dates)
print(result)   # → Born: 15/01/2024, Died: 31/12/2099

# Wrap all numbers in <b> tags
result = re.sub(r'(\d+)', r'<b>\1</b>', 'I have 3 cats and 2 dogs')
print(result)   # → I have <b>3</b> cats and <b>2</b> dogs

# Named group backreference \g<name>
result = re.sub(
    r'(?P<last>\w+), (?P<first>\w+)',
    r'\g<first> \g<last>',
    'Smith, John'
)
print(result)   # → John Smith
```

### 3) Replacement function (替换函数)

Pass a **callable** as the replacement — it receives the Match object and returns the replacement string.

```python
import re

# Convert all numbers to their double
def double(m):
    return str(int(m.group()) * 2)

result = re.sub(r'\d+', double, 'I have 3 cats and 10 dogs')
print(result)   # → I have 6 cats and 20 dogs

# Normalize different date formats to ISO 8601
def normalize_date(m):
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    month = month_map.get(m.group('month_name'),
                          int(m.group('month_num') or 0))
    day   = int(m.group('day'))
    year  = int(m.group('year'))
    return f"{year:04d}-{month:02d}-{day:02d}"

pattern = re.compile(r'''
    (?:(?P<month_name>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
       \s+(?P<day>\d{1,2}),\s+(?P<year>\d{4}))
    |
    (?:(?P<month_num>\d{1,2})/(?P<day2>\d{1,2})/(?P<year2>\d{4}))
''', re.VERBOSE)

# Just demonstrate the function approach:
text = "Meeting on Jan 15, 2024"
result = re.sub(
    r'(?P<month_name>Jan|Feb|Mar)\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})',
    normalize_date,
    text
)
print(result)   # → Meeting on 2024-01-15
```

------

## 8. `re.subn()` — Substitute and count (替换并计数)

Like `re.sub()` but returns a tuple `(new_string, count)`.

```python
import re

text = "foo bar foo baz foo"
result, n = re.subn(r'foo', 'qux', text)
print(result)   # → qux bar qux baz qux
print(n)        # → 3  (number of substitutions made)

# Useful for detecting if any replacements occurred
text2 = "no matches here"
_, count = re.subn(r'foo', 'qux', text2)
if count == 0:
    print("No substitutions made")
```

------

## 9. `re.split()` — Split by pattern (按模式分割)

```python
import re

# Split on any non-alphanumeric sequence
text = "one,two;;three   four\tfive"
print(re.split(r'[^a-zA-Z0-9]+', text))
# → ['one', 'two', 'three', 'four', 'five']

# Split on commas with optional surrounding whitespace
csv = "Alice , Bob,Carol ,  Dave"
print(re.split(r'\s*,\s*', csv))
# → ['Alice', 'Bob', 'Carol', 'Dave']

# maxsplit: only split N times
print(re.split(r'\s+', 'a b c d e', maxsplit=2))
# → ['a', 'b', 'c d e']

# Capturing group: delimiters are INCLUDED in the result
text = "one+two-three*four"
print(re.split(r'([+\-*])', text))
# → ['one', '+', 'two', '-', 'three', '*', 'four']  ← operators kept
```

------

## 10. `re.escape()` — Escape special characters (转义特殊字符)

Escapes all non-alphanumeric characters so a raw string can be used as a literal pattern.

```python
import re

# When user input is used as part of a pattern — MUST escape it
user_input = "hello.world (test)"
safe_pattern = re.escape(user_input)
print(safe_pattern)   # → hello\.world\ \(test\)

# Safe search
text = "I said: hello.world (test) today"
m = re.search(re.escape(user_input), text)
print(bool(m))   # → True

# Dangerous without escape:
print(re.search(user_input, text))   # . and () have special meaning!

# Common use: build a pattern from a list of keywords
keywords = ['c++', 'c#', '.net', 'node.js']
pattern  = '|'.join(re.escape(k) for k in keywords)
print(pattern)   # → c\+\+|c\#|\.net|node\.js

found = re.findall(pattern, 'I know c++ and .net and node.js', re.I)
print(found)   # → ['c++', '.net', 'node.js']
```

------

## 11. Match Object — Complete API (匹配对象完整API)

```python
import re

text = "2024-01-15 is a Monday in New York"
m    = re.search(r'(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})', text)

# ── Accessing matched text ─────────────────────────────────
print(m.group())           # → 2024-01-15  (full match, same as group(0))
print(m.group(0))          # → 2024-01-15
print(m.group(1))          # → 2024         (group 1 by index)
print(m.group(2, 3))       # → ('01', '15') (multiple groups)
print(m.group('year'))     # → 2024         (group by name)
print(m.groupdict())       # → {'year': '2024', 'month': '01', 'day': '15'}
print(m.groups())          # → ('2024', '01', '15')  (all groups as tuple)
print(m.groups(default='N/A'))  # groups() with default for non-participating groups

# ── Position information ───────────────────────────────────
print(m.start())           # → 0    (start of full match)
print(m.end())             # → 10   (end of full match)
print(m.span())            # → (0, 10)
print(m.start(1))          # → 0    (start of group 1)
print(m.end('month'))      # → 7    (end of named group)
print(m.span('day'))       # → (8, 10)

# ── Context ────────────────────────────────────────────────
print(m.string)            # → full original string
print(m.re)                # → compiled pattern object
print(m.pos)               # → 0    (start position passed to search)
print(m.endpos)            # → 34   (end position passed to search)
print(m.lastindex)         # → 3    (index of last matched group)
print(m.lastgroup)         # → 'day' (name of last matched group)

# ── Expand — backreferences in a template string ───────────
print(m.expand(r'\g<day>/\g<month>/\g<year>'))
# → 15/01/2024
```

------

# **IV. Practical Patterns — Production-Ready Recipes (生产级常用模式)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> This section provides <span style="color:#E8600A;font-weight:700">ready-to-use, battle-tested patterns</span> for the most common real-world tasks. Each pattern is annotated and tested. </div>

## 1. Validation Patterns (验证模式)

```python
import re

patterns = {

    # Email (simplified RFC 5321 compliant)
    'email': re.compile(r'''
        ^[a-zA-Z0-9._%+\-]+     # local part
        @
        [a-zA-Z0-9.\-]+          # domain
        \.[a-zA-Z]{2,}$          # TLD (2+ chars)
    ''', re.VERBOSE),

    # Phone: +1 (555) 123-4567 / 555-123-4567 / 5551234567
    'phone_us': re.compile(
        r'^(\+1[-.\s]?)?'
        r'(\(?\d{3}\)?[-.\s]?)'
        r'\d{3}[-.\s]?\d{4}$'
    ),

    # IPv4 address
    'ipv4': re.compile(
        r'^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(25[0-5]|2[0-4]\d|[01]?\d\d?)$'
    ),

    # URL (http/https)
    'url': re.compile(
        r'^https?://'
        r'(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})'
        r'(:\d+)?'
        r'(/[^\s]*)?$'
    ),

    # Date: YYYY-MM-DD
    'date_iso': re.compile(
        r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$'
    ),

    # Strong password: 8+ chars, upper, lower, digit, special
    'strong_password': re.compile(
        r'^(?=.*[a-z])'          # at least one lowercase
        r'(?=.*[A-Z])'           # at least one uppercase
        r'(?=.*\d)'              # at least one digit
        r'(?=.*[!@#$%^&*])'     # at least one special char
        r'.{8,}$'                # at least 8 chars total
    ),

    # Credit card (Visa/MC/Amex, with/without spaces)
    'credit_card': re.compile(
        r'^(?:4\d{12}(?:\d{3})?'     # Visa
        r'|5[1-5]\d{14}'             # MasterCard
        r'|3[47]\d{13})$'            # Amex
    ),

    # ZIP code (US)
    'zip_us': re.compile(r'^\d{5}(-\d{4})?$'),

    # Hex color
    'hex_color': re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'),

    # Semantic version: 1.2.3 or 1.2.3-alpha.1
    'semver': re.compile(
        r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
        r'(-[a-zA-Z0-9.\-]+)?(\+[a-zA-Z0-9.\-]+)?$'
    ),
}

# Test them
tests = {
    'email':           ['user@example.com', 'bad@', 'no-at-sign'],
    'ipv4':            ['192.168.1.1', '256.0.0.1', '10.0.0'],
    'date_iso':        ['2024-01-15', '2024-13-01', '24-1-1'],
    'strong_password': ['Abc@1234', 'weakpass', 'NoSpecial1'],
    'hex_color':       ['#FF5733', '#abc', '#GGGGGG'],
    'semver':          ['1.2.3', '1.0.0-alpha.1', '1.2'],
}

for field, values in tests.items():
    pat = patterns[field]
    print(f"\n{field}:")
    for v in values:
        ok = '✅' if pat.fullmatch(v) else '❌'
        print(f"  {ok} {v!r}")
```

------

## 2. Extraction Patterns (提取模式)

```python
import re

# ── Extract all URLs from text ──────────────────────────────
def extract_urls(text):
    pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(pattern, text)

html = 'Visit <a href="https://example.com/path?q=1">site</a> or http://other.org'
print(extract_urls(html))
# → ['https://example.com/path?q=1', 'http://other.org']


# ── Extract all emails ──────────────────────────────────────
def extract_emails(text):
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

text = "Contact alice@example.com or bob.smith@company.co.uk for info"
print(extract_emails(text))
# → ['alice@example.com', 'bob.smith@company.co.uk']


# ── Parse log lines ─────────────────────────────────────────
def parse_log(line):
    pattern = re.compile(r'''
        (?P<ip>[\d.]+)          \s+   # IP address
        \S+                     \s+   # ident
        \S+                     \s+   # auth user
        \[(?P<time>[^\]]+)\]    \s+   # timestamp
        "(?P<method>\w+)        \s+
         (?P<path>[^\s"]+)      \s+
         \S+"                   \s+   # HTTP version
        (?P<status>\d{3})       \s+   # status code
        (?P<size>\d+)                 # bytes
    ''', re.VERBOSE)
    m = pattern.match(line)
    return m.groupdict() if m else None

log_line = '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326'
print(parse_log(log_line))
# → {'ip': '127.0.0.1', 'time': '10/Oct/2000:13:55:36 -0700',
#    'method': 'GET', 'path': '/apache_pb.gif', 'status': '200', 'size': '2326'}


# ── Extract numbers with units ──────────────────────────────
def extract_measurements(text):
    pattern = r'(\d+(?:\.\d+)?)\s*(px|em|rem|%|pt|vh|vw)'
    return [(float(v), u) for v, u in re.findall(pattern, text)]

css = "width: 100px; margin: 1.5em; font-size: 16px; height: 50vh"
print(extract_measurements(css))
# → [(100.0, 'px'), (1.5, 'em'), (16.0, 'px'), (50.0, 'vh')]
```

------

## 3. Cleaning and Normalization Patterns (清理与标准化模式)

```python
import re

# ── Normalize whitespace ────────────────────────────────────
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

print(normalize_whitespace("  Hello   World  \n\t  Python  "))
# → Hello World Python


# ── Remove HTML tags ────────────────────────────────────────
def strip_html(html):
    clean = re.sub(r'<[^>]+>', '', html)
    return re.sub(r'\s+', ' ', clean).strip()

html = "<h1>Title</h1><p>Some <b>bold</b> and <em>italic</em> text.</p>"
print(strip_html(html))
# → Title Some bold and italic text.


# ── Slugify a string ────────────────────────────────────────
def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '',  text)   # remove non-word chars
    text = re.sub(r'[\s_]+',   '-', text)   # spaces/underscores → dash
    text = re.sub(r'-+',       '-', text)   # multiple dashes → one
    return text.strip('-')

print(slugify("Hello, World! This is Python 3.12"))
# → hello-world-this-is-python-312


# ── Camel case to snake case ────────────────────────────────
def camel_to_snake(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)  # ABCDef → ABC_Def
    name = re.sub(r'([a-z\d])([A-Z])',      r'\1_\2', name)  # fooBar → foo_Bar
    return name.lower()

print(camel_to_snake('camelCaseString'))    # → camel_case_string
print(camel_to_snake('parseHTMLContent'))   # → parse_html_content
print(camel_to_snake('MyClassName'))        # → my_class_name


# ── Mask sensitive data ─────────────────────────────────────
def mask_credit_card(text):
    return re.sub(r'\b(\d{4})\d{8}(\d{4})\b', r'\1 **** **** \2', text)

def mask_email(text):
    return re.sub(r'(\w{2})\w+(@[^\s]+)', r'\1***\2', text)

print(mask_credit_card("Card: 4111111111111111"))
# → Card: 4111 **** **** 1111
print(mask_email("Email alice@example.com to bob@test.org"))
# → Email al***@example.com to bo***@test.org
```

------

## 4. Common Pitfalls (常见陷阱)

### 1) Catastrophic backtracking (灾难性回溯)

```python
import re, time

# ⚠️ DANGEROUS pattern: (a+)+ causes exponential backtracking
evil_pattern   = r'^(a+)+$'
safe_pattern   = r'^a+$'

test_string = 'a' * 25 + 'X'  # no match — forces max backtracking

# Safe pattern — fast
t = time.time()
re.search(safe_pattern, test_string)
print(f"Safe:  {time.time()-t:.6f}s")   # → ~0.000001s

# Evil pattern — hangs for long inputs!
# (DO NOT run with 'a' * 30 + 'X')
t = time.time()
re.search(evil_pattern, 'a' * 20 + 'X')
print(f"Evil:  {time.time()-t:.6f}s")   # → much longer

# FIX: use atomic groups or possessive quantifiers, or restructure
# In Python 3.11+: use re.POSSESSIVE or regex module
```

### 2) `re.match()` vs `re.search()` confusion

```python
import re

# COMMON MISTAKE: using match() when search() is needed
data = "  123 some text"

# Incorrect — thinking match() searches anywhere
result = re.match(r'\d+', data)   # → None!  (leading spaces)

# Correct
result = re.search(r'\d+', data)  # → '123'

# Or anchor explicitly
result = re.match(r'\s*(\d+)', data)  # → group(1) = '123'
```

### 3) `findall()` group return type surprise

```python
import re

text = "2024-01 2024-02"

# Bug: adding a group changes return type
print(re.findall(r'\d{4}-\d{2}',       text))  # → ['2024-01', '2024-02']
print(re.findall(r'(\d{4})-\d{2}',     text))  # → ['2024', '2024']  (years only!)
print(re.findall(r'(\d{4})-(\d{2})',   text))  # → [('2024','01'), ('2024','02')]

# Fix: use non-capturing group when you don't need the group value
print(re.findall(r'(?:\d{4})-(?:\d{2})', text))  # → ['2024-01', '2024-02']
```

### 4) Forgetting raw strings

```python
import re

# WRONG: \b interpreted by Python as backspace character (ASCII 8)
print(re.findall('\bword\b', 'word in a sentence'))   # → []  WRONG

# CORRECT: raw string
print(re.findall(r'\bword\b', 'word in a sentence'))  # → ['word']

# WRONG: \d interpreted as literal 'd' in some contexts
print(re.findall('\d+', 'abc 123'))   # may work but is fragile
# CORRECT:
print(re.findall(r'\d+', 'abc 123'))  # → ['123']
```

------

# **V. Complete API Quick Reference (完整API速查表)**

| Function / Method                                            | Returns            | Use when                         |
| ------------------------------------------------------------ | ------------------ | -------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.compile(pat, flags)</code> | Pattern            | Pattern reused multiple times    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.search(pat, s)</code> | Match or None      | Find first match anywhere        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.match(pat, s)</code> | Match or None      | Match only at position 0         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.fullmatch(pat, s)</code> | Match or None      | Pattern must cover entire string |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.findall(pat, s)</code> | List[str or tuple] | All matches as a list            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.finditer(pat, s)</code> | Iterator[Match]    | All matches with position info   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.sub(pat, repl, s)</code> | str                | Replace matches                  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.subn(pat, repl, s)</code> | (str, int)         | Replace + count substitutions    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.split(pat, s)</code> | List[str]          | Split string by pattern          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.escape(s)</code> | str                | Treat literal string as pattern  |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.group(n)</code> | str                | Get captured group text          |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.groups()</code> | tuple              | All groups as tuple              |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.groupdict()</code> | dict               | Named groups as dict             |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.start() / m.end()</code> | int                | Match position                   |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.span()</code> | (int, int)         | (start, end) tuple               |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">m.expand(template)</code> | str                | Backreference expansion          |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Master regex in four steps: <span style="color:#E8600A;font-weight:700">① know the 5 building blocks</span> (literals, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.</code>, classes <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[]</code>, quantifiers <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">*+?{}</code>, anchors <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">^$\b</code>) → <span style="color:#E8600A;font-weight:700">② use groups <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">()</code> to capture, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?:)</code> to group without capturing</span> → <span style="color:#E8600A;font-weight:700">③ add lookaround <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">(?=)(?!)</code> for context-sensitive matching</span> → <span style="color:#E8600A;font-weight:700">④ always use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">r''</code> raw strings, prefer non-greedy <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">.*?</code>, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">re.VERBOSE</code> for complex patterns</span>. </div>
