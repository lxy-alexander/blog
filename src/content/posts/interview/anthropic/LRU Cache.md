---
title: "LRU Cache"
published: 2026-06-06
description: "LRU Cache"
image: ""
tags: ["interview","anthropic","LRU Cache"]
category: interview / anthropic
draft: false
lang: ""
createdAt: "2026-06-06T23:39:48.972.497067625Z"
---

# I. LRU Cache

## 1. Problem Statement

### 1) Problem Description

Design and implement an LRU Cache.

The cache should support two operations:

```python
get(key)
put(key, value)
```

The cache has a fixed capacity.

When the cache exceeds its capacity, it should remove the least recently used key.

The required average time complexity for both operations is:

```text
O(1)
```

------

### 2) Operation Rules

#### get(key)

If the key exists in the cache:

```text
Return the value.
Mark this key as recently used.
```

If the key does not exist:

```text
Return -1.
```

------

#### put(key, value)

If the key already exists:

```text
Update the value.
Mark this key as recently used.
```

If the key does not exist:

```text
Insert the key-value pair.
Mark this key as recently used.
```

If the cache size exceeds capacity:

```text
Remove the least recently used key.
```

------

### 3) Example

Input:

```text
capacity = 2

put(1, 1)
put(2, 2)
get(1)
put(3, 3)
get(2)
put(4, 4)
get(1)
get(3)
get(4)
```

Output:

```text
1
-1
-1
3
4
```

Explanation:

```text
put(1, 1)    cache = [1]
put(2, 2)    cache = [2, 1]
get(1)       return 1, cache = [1, 2]
put(3, 3)    evict 2, cache = [3, 1]
get(2)       return -1
put(4, 4)    evict 1, cache = [4, 3]
get(1)       return -1
get(3)       return 3, cache = [3, 4]
get(4)       return 4, cache = [4, 3]
```

In the cache order above:

```text
Left side = most recently used
Right side = least recently used
```

------

## 2. Core Idea

### 1) Hash Map

We need fast lookup by key.

Use a hash map:

```python
cache = {
    key: node
}
```

This allows us to find the corresponding node in:

```text
O(1)
```

Without a hash map, we would need to scan the linked list, which takes:

```text
O(n)
```

------

### 2) Doubly Linked List

We also need to maintain usage order.

Use a doubly linked list where:

```text
Head side = most recently used
Tail side = least recently used
```

For example:

```text
head <-> 4 <-> 3 <-> 1 <-> tail
```

This means:

```text
4 is the most recently used key.
1 is the least recently used key.
```

When a key is accessed, move it to the front.

When the cache exceeds capacity, remove the node before `tail`.

------

### 3) Why Not Only Use Hash Map?

A hash map can find a key in O(1).

However, it does not maintain usage order.

For example:

```python
{
    1: 1,
    2: 2,
    3: 3
}
```

From this structure alone, we cannot know which key is least recently used.

Therefore, hash map alone is not enough.

------

### 4) Why Not Only Use Linked List?

A linked list can maintain usage order.

However, finding a key requires scanning from head to tail.

That takes:

```text
O(n)
```

Therefore, linked list alone is not enough.

------

### 5) Final Design

Use:

```text
Hash Map + Doubly Linked List
```

The hash map provides:

```text
O(1) lookup
```

The doubly linked list provides:

```text
O(1) remove
O(1) insert at front
O(1) remove least recently used node
```

------

## 3. Algorithm Steps

### 1) Data Structures

Use a node class:

```text
Node:
    key
    value
    prev
    next
```

Use a hash map:

```python
cache = {}
```

The map stores:

```text
key -> node
```

Use two dummy nodes:

```text
head <-> ... <-> tail
```

The real cache nodes are always between `head` and `tail`.

------

### 2) Helper Function: Remove Node

To remove a node from the linked list:

```text
node.prev.next = node.next
node.next.prev = node.prev
```

This operation takes:

```text
O(1)
```

------

### 3) Helper Function: Add Node to Front

To mark a node as most recently used, insert it right after `head`.

```text
head <-> node <-> old_first
```

This operation takes:

```text
O(1)
```

------

### 4) Helper Function: Move Node to Front

When a key is used, move its node to the front.

Steps:

```text
Remove node from its current position.
Add node right after head.
```

------

### 5) get(key)

Algorithm:

```text
If key does not exist:
    Return -1.

If key exists:
    Get the node from hash map.
    Move the node to the front.
    Return node.value.
```

------

### 6) put(key, value)

Algorithm:

```text
If key exists:
    Update node.value.
    Move the node to the front.
    Return.

If key does not exist:
    Create a new node.
    Add it to the hash map.
    Add it to the front of the linked list.

If cache size exceeds capacity:
    Remove the node before tail.
    Delete its key from the hash map.
```

------

## 4. Code Implementation

### 1) Python Implementation with Hash Map and Doubly Linked List

```python
class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

        self.head = Node()
        self.tail = Node()

        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        prev_node = node.prev
        next_node = node.next

        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        first_node = self.head.next

        node.prev = self.head
        node.next = first_node

        self.head.next = node
        first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._move_to_front(node)

        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            return

        new_node = Node(key, value)
        self.cache[key] = new_node
        self._add_to_front(new_node)

        if len(self.cache) > self.capacity:
            lru_node = self.tail.prev

            self._remove(lru_node)
            del self.cache[lru_node.key]
```

------

### 2) Python Implementation with Input Parser

This version is useful for online judges that provide commands through standard input.

```python
import sys


class Node:
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

        self.head = Node()
        self.tail = Node()

        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        prev_node = node.prev
        next_node = node.next

        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node) -> None:
        first_node = self.head.next

        node.prev = self.head
        node.next = first_node

        self.head.next = node
        first_node.prev = node

    def _move_to_front(self, node: Node) -> None:
        self._remove(node)
        self._add_to_front(node)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._move_to_front(node)

        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
            return

        new_node = Node(key, value)
        self.cache[key] = new_node
        self._add_to_front(new_node)

        if len(self.cache) > self.capacity:
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self.cache[lru_node.key]


def main():
    input_data = sys.stdin.read().strip().splitlines()

    if not input_data:
        return

    capacity, n = map(int, input_data[0].split())

    lru = LRUCache(capacity)
    output = []

    for line in input_data[1:]:
        parts = line.split()

        if parts[0] == "get":
            key = int(parts[1])
            output.append(str(lru.get(key)))
        else:
            key = int(parts[1])
            value = int(parts[2])
            lru.put(key, value)

    print("\n".join(output))


if __name__ == "__main__":
    main()
```

------

### 3) Python Implementation with OrderedDict

Python provides `OrderedDict`, which can simplify the implementation.

```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        value = self.cache.pop(key)
        self.cache[key] = value

        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)

        self.cache[key] = value

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

This version is shorter, but for interviews, the manual doubly linked list version is usually preferred because the problem is designed to test the underlying data structure design.

------

## 5. Complexity Analysis

### 1) Time Complexity

#### get(key)

```text
O(1)
```

Because:

```text
Hash map lookup is O(1).
Removing a node from a doubly linked list is O(1).
Adding a node to the front is O(1).
```

------

#### put(key, value)

```text
O(1)
```

Because:

```text
Hash map insertion/update is O(1).
Linked list insertion is O(1).
LRU eviction from the tail is O(1).
```

------

### 2) Space Complexity

The cache stores at most `capacity` nodes.

Therefore:

```text
O(capacity)
```

The hash map stores at most `capacity` key-node mappings.

The linked list also stores at most `capacity` real nodes.

------

## 6. System Design Discussion

### 1) Why Use Dummy Head and Tail?

Using dummy nodes simplifies linked list operations.

The list always looks like this:

```text
head <-> real nodes <-> tail
```

Even when the cache is empty:

```text
head <-> tail
```

This avoids special cases such as:

```text
Removing the first node
Removing the last node
Adding into an empty list
Handling only one node
```

With dummy nodes, every remove and insert operation has the same logic.

------

### 2) Why get() Must Update Usage Order

A common mistake is to write:

```python
def get(self, key):
    if key not in self.cache:
        return -1

    return self.cache[key].value
```

This is incorrect.

Calling `get(key)` means the key has just been used.

Therefore, the node must be moved to the front.

Correct logic:

```python
node = self.cache[key]
self._move_to_front(node)
return node.value
```

------

### 3) Why put() on Existing Key Must Update Usage Order

Another common mistake is to update the value but not move the key to the front.

For example:

```text
put(1, 1)
put(2, 2)
put(1, 100)
```

The second `put(1, 100)` is also a use of key `1`.

Therefore, key `1` should become the most recently used key.

------

### 4) Why Eviction Must Remove from Both Structures

When the cache exceeds capacity, we remove the least recently used node from the linked list.

But we must also remove it from the hash map.

Incorrect:

```python
self._remove(lru_node)
```

Correct:

```python
self._remove(lru_node)
del self.cache[lru_node.key]
```

If we forget to delete it from the hash map, the cache will still think the key exists.

This creates a stale pointer bug.

------

### 5) Manual Linked List vs OrderedDict

Using `OrderedDict` is concise and practical in real Python code.

However, in interviews, the manual implementation is usually preferred.

Reason:

```text
The problem is designed to test whether you understand how hash map and doubly linked list work together.
```

Use `OrderedDict` only if the interviewer allows library-based solutions.

------

### 6) Difference from a Generic Function Cache

This problem is a standard LRU Cache design problem.

It focuses on:

```text
Hash Map
Doubly Linked List
O(1) get
O(1) put
LRU eviction policy
```

A generic function cache with `*args` and `**kwargs` focuses on different topics:

```text
Generic callable caching
Hashable cache keys
Argument normalization
Disk persistence
Crash recovery
Durability
```

Therefore, these are two different problems.

------

### 7) Common Follow-up Questions

1.  Why do we need both a hash map and a doubly linked list?
2.  Why does `get()` need to move the node to the front?
3.  Why does `put()` for an existing key also move the node to the front?
4.  Why do we use dummy head and dummy tail nodes?
5.  What happens if capacity is 0?
6.  How would you make this cache thread-safe?
7.  How would you add TTL expiration?
8.  How would you implement LFU Cache instead of LRU Cache?
9.  How would you shard this cache across multiple machines?
10.  How would you persist this cache to disk?
