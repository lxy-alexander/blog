---
title: "Fireworks AI"
published: 2026-10-02
description: "Fireworks AI interview notes: mutual-friend recommendations and NDCG evaluation."
image: ""
tags: ["interview", "fireworks-ai", "graph", "recommendation", "NDCG"]
category: interview
draft: false
lang: ""
---

# Referral System (共同好友推荐系统)

[题目链接 / Problem Link](https://www.1point3acres.com/interview/thread/1189232)

## 1. 题目描述 / Problem Statement

### 中文

给定一个无向用户关系图，节点表示用户，边表示两个用户互为好友。为目标用户 `u` 生成好友推荐列表。

候选用户 `v` 必须满足以下条件：

- `v != u`，不能推荐用户本人。
- `v` 不是 `u` 的直接好友。
- `u` 和 `v` 至少有一个共同好友。

以共同好友数量作为排序分数，按共同好友数降序排列；数量相同时，按用户 ID 升序排列。没有候选人时返回空列表。

页面整理版题目的约束为：用户 ID 是 `0` 到 `n - 1`，`2 <= n <= 2 * 10^5`，`0 <= m <= 5 * 10^5`，没有自环或重复边。

原帖还提到第二部分：给推荐结果提供 `[0, 1]` 范围内的分数，并提到了 NDCG。面试时需要确认要求的是每个候选人的推荐分数，还是整份列表的排序质量指标；两者分别在代码实现和面试追问中讨论。

### English

Given an undirected friendship graph, where nodes represent users and edges represent mutual friendships, generate friend recommendations for a target user `u`.

A candidate user `v` must satisfy all of the following conditions:

- `v != u`: do not recommend the user to themselves.
- `v` is not already a direct friend of `u`.
- `u` and `v` share at least one mutual friend.

Rank candidates by their number of mutual friends in descending order. Break ties by user ID in ascending order. Return an empty list when no candidates exist.

The structured problem on the page specifies user IDs from `0` to `n - 1`, `2 <= n <= 2 * 10^5`, and `0 <= m <= 5 * 10^5`, with no self-loops or duplicate edges.

The original post also mentions bounded scores in `[0, 1]` and NDCG. Clarify whether the interviewer wants a score for each candidate or an evaluation metric for the entire ranked list. These are discussed separately below.

### 示例 / Example

```text
n = 6
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 5)]
u = 0
```

| 候选人 / Candidate | 共同好友 / Mutual Friends | 数量 / Count | 顺序 / Rank |
| --- | --- | --- | --- |
| 3 | 1, 2 | 2 | 1 |
| 4 | 1 | 1 | 2 |
| 5 | 2 | 1 | 3 |

```text
Recommended user IDs: [3, 4, 5]
```

## 2. 核心思路 / Core Approach

### 中文

使用集合形式的邻接表 `graph[u]` 存储用户 `u` 的直接好友，使用哈希计数器记录候选人的共同好友数。

1. 读取目标用户的好友集合 `friends = graph[u]`。
2. 遍历每个 `friend`，再遍历 `graph[friend]` 中的用户 `candidate`。
3. 排除目标用户本人和已经是直接好友的用户。
4. 对剩余候选人执行 `counts[candidate] += 1`。
5. 用 `(-counts[v], v)` 作为排序键。

每条有效的两跳路径 `u -> friend -> candidate` 对应一个共同好友。集合邻接表确保同一个中间好友不会因重复边被重复计算。

遍历完所有直接好友后的不变量为：

$$
\operatorname{counts}[v] = |N(u) \cap N(v)|
$$

这里 `N(x)` 是用户 `x` 的直接好友集合，`v` 是满足推荐条件的候选人。在遍历过程中，计数等于已经处理的直接好友中，同时也是 `v` 好友的人数。

### English

Store the graph as an adjacency list of sets and use a hash counter to count mutual friends for each candidate.

1. Read the target user's direct friends: `friends = graph[u]`.
2. For each `friend`, traverse every `candidate` in `graph[friend]`.
3. Exclude the target user and existing direct friends.
4. Increment `counts[candidate]` for each remaining candidate.
5. Sort candidates using the key `(-counts[v], v)`.

Each valid two-hop path `u -> friend -> candidate` contributes one mutual friend. Set-based adjacency lists prevent duplicate edges from counting the same intermediary twice.

After processing all direct friends, the invariant is `counts[v] = |N(u) intersect N(v)|` for every eligible candidate `v`. During traversal, the count equals the number of processed direct friends who are also friends with `v`.

## 3. 代码实现 / Code Implementation

图只需构建一次，随后可以查询多个用户。默认返回 `(用户 ID, 共同好友数)`；`normalize=True` 时返回 `(用户 ID, 归一化分数)`。只需要题目的 ID 列表时，提取每个元组的第一个元素即可。

Build the graph once and reuse it for multiple users. By default, return `(user_id, mutual_friend_count)` pairs. With `normalize=True`, return `(user_id, normalized_score)` pairs. Extract the first element of each pair when only user IDs are required.

```python
from collections import Counter


def build_graph(n, edges):
    graph = [set() for _ in range(n)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def recommend(graph, u, normalize=False):
    friends = graph[u]
    if not friends:
        return []

    counts = Counter()
    for friend in friends:
        for candidate in graph[friend]:
            if candidate != u and candidate not in friends:
                counts[candidate] += 1

    ranked = sorted(counts, key=lambda v: (-counts[v], v))
    if normalize:
        return [(v, counts[v] / len(friends)) for v in ranked]
    return [(v, counts[v]) for v in ranked]


edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 5)]
graph = build_graph(6, edges)

print(recommend(graph, 0))
# [(3, 2), (4, 1), (5, 1)]

print([v for v, _ in recommend(graph, 0)])
# [3, 4, 5]

print(recommend(graph, 0, normalize=True))
# [(3, 1.0), (4, 0.5), (5, 0.5)]
```

可选的归一化定义为：

$$
\operatorname{score}(u, v)
= \frac{|N(u) \cap N(v)|}{|N(u)|}
$$

共同好友数不超过目标用户的好友数，所以分数在 `[0, 1]` 内。对于同一目标用户，分母固定，因此不会改变原排序。这个值表示目标用户的好友中，有多少比例也认识候选人，不是经过校准的交友概率。没有好友时直接返回空列表，避免除以零。

The mutual-friend count cannot exceed the target user's degree, so the score lies in `[0, 1]`. For a fixed target user, the denominator is constant and preserves the original ranking. The score measures the fraction of the target user's friends who also know the candidate; it is not a calibrated probability of becoming friends. An isolated user returns an empty list without division.

## 4. 复杂度与易错点 / Complexity and Pitfalls

### 中文

设 `n` 是用户数，`m` 是边数，`c` 是候选人数，`deg(x)` 是用户 `x` 的度数。令：

$$
S = \sum_{f \in N(u)} \deg(f)
$$

- 建图：期望时间 `O(n + m)`，存储空间 `O(n + m)`。
- 单用户查询：期望时间 `O(deg(u) + S + c log c)`；无自环的无向图中 `deg(u) <= S <= 2m`，因此通常写为 `O(S + c log c)`。
- 查询额外空间：计数器、排序结果和返回列表均为 `O(c)`，不包含已经构建的图。

易错点：

- 必须排除用户本人：`u -> friend -> u` 也是两跳路径。
- 必须排除已有好友：三角形中的直接好友也会通过两跳路径被遇到。
- 不能对候选人使用全局 `visited` 去重；同一个候选人通过不同好友出现，需要分别计数。
- 邻接表用集合去重边，但计数器保留不同共同好友的贡献。
- 排序键是 `(-counts[v], v)`，两个排序方向不同。
- 不能只根据目标用户的度数断言查询是 `O(deg(u)^2)`；他的朋友可能有很高的度数。
- 孤立用户或没有候选人的用户返回空列表；如果外部接口要求打印 ID，则打印空行。

### English

Let `n` be the number of users, `m` the number of edges, `c` the number of candidates, and `deg(x)` the degree of user `x`. Define `S = sum(deg(f) for f in N(u))`.

- Graph construction takes expected `O(n + m)` time and `O(n + m)` space.
- A single-user query takes expected `O(deg(u) + S + c log c)` time. In a loop-free undirected graph, `deg(u) <= S <= 2m`, so this is usually written as `O(S + c log c)`.
- Query auxiliary space is `O(c)` for counting and ranking, excluding the existing graph.

Common pitfalls:

- Exclude the target user because `u -> friend -> u` is a two-hop path.
- Exclude direct friends because triangles can also produce two-hop paths to them.
- Do not globally deduplicate candidates with a `visited` set. Different intermediate friends must contribute separately.
- Deduplicate edges in adjacency sets while retaining contributions from distinct mutual friends.
- Use `(-counts[v], v)` to implement descending counts and ascending IDs.
- Do not claim `O(deg(u)^2)` based only on the target user's degree; their friends can have much higher degrees.
- Return an empty list for isolated users or users without eligible candidates. Print an empty line if the external interface requires IDs on one line.

## 5. 最小测试集 / Minimal Test Cases

| 场景 / Scenario | 验证内容 / Expected Behavior |
| --- | --- |
| 正常推荐 / Standard recommendation | 共同好友多的排在前面 / More mutual friends rank first |
| 同分排序 / Tie-breaking | 共同好友数相同，ID 小的优先 / Smaller IDs rank first on ties |
| 没有好友 / Isolated user | 返回空列表 / Return an empty list |
| 全部候选人已经是好友 / All reachable users are friends | 排除本人和直接好友，返回空列表 / Exclude self and direct friends |

以下测试在代码实现之后运行，复用上面的 `build_graph` 和 `recommend`。

Run these tests after the implementation, reusing `build_graph` and `recommend` above.

```python
def test_recommend():
    # Standard recommendation and multiple mutual friends.
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 5)]
    graph = build_graph(6, edges)
    assert recommend(graph, 0) == [(3, 2), (4, 1), (5, 1)]
    assert recommend(graph, 0, normalize=True) == [
        (3, 1.0), (4, 0.5), (5, 0.5)
    ]

    # Tie-breaking must not depend on insertion or traversal order.
    graph = build_graph(5, [(0, 1), (1, 4), (1, 2)])
    assert recommend(graph, 0) == [(2, 1), (4, 1)]

    # Isolated user, including normalized output.
    graph = build_graph(3, [])
    assert recommend(graph, 0) == []
    assert recommend(graph, 0, normalize=True) == []

    # A triangle reaches only the user and existing direct friends.
    graph = build_graph(3, [(0, 1), (0, 2), (1, 2)])
    assert recommend(graph, 0) == []


test_recommend()
```

## 6. 面试追问 / Interview Follow-Ups

### 6.1 推荐分数与 NDCG / Candidate Scores vs. NDCG

**中文：** 推荐分数用于给候选人排序，可以使用共同好友数或上面的归一化定义。NDCG 是整份推荐列表的排序质量指标，不是每个候选人的独立分数。仅有推荐列表和预测分数，没有真实相关性标签时，无法计算有意义的 NDCG。

**English:** Candidate scores are used to rank users and can be mutual-friend counts or normalized scores. NDCG evaluates the quality of the entire ranking; it is not an independent score for each candidate. A recommendation list and predicted scores alone are insufficient without ground-truth relevance labels.

采用线性 gain 时，定义如下，其中 `rel(v_i)` 是第 `i` 个推荐用户的非负真实相关性：

Using linear gain, define the metric as follows, where `rel(v_i)` is the nonnegative ground-truth relevance of the user at rank `i`:

$$
DCG@k = \sum_{i=1}^{k} \frac{rel(v_i)}{\log_2(i+1)}
$$

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

**中文：** `IDCG@k` 是全部参与评估的候选人按真实相关性降序排列后，前 `k` 个的 DCG，不能只排序实际返回的推荐结果。非负标签下 NDCG 在 `[0, 1]` 内；理想排序为 `1`。当所有标签为零时，本文约定返回 `0`。也有使用 `2^rel - 1` 作为 gain 的版本，面试时应确认采用哪一种定义。

**English:** `IDCG@k` is the DCG of the top `k` users after sorting the entire evaluation candidate set by true relevance, not merely the returned recommendations. With nonnegative labels, NDCG lies in `[0, 1]`, and an ideal ranking scores `1`. This implementation returns `0` when all labels are zero. Another convention uses `2^rel - 1` as gain; clarify which definition is expected.

[参考 / Reference: scikit-learn NDCG documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)

```python
from math import log2


def ndcg_at_k(ranked, relevance, k):
    # relevance contains labels for the entire evaluation candidate set.
    if k <= 0:
        return 0.0
    if len(set(ranked)) != len(ranked):
        raise ValueError("Ranked user IDs must be unique")
    if any(v not in relevance for v in ranked):
        raise ValueError("Every ranked user must have a relevance label")
    if any(r < 0 for r in relevance.values()):
        raise ValueError("Relevance labels must be nonnegative")

    def dcg(labels):
        return sum(r / log2(i + 2) for i, r in enumerate(labels))

    actual = dcg([relevance[v] for v in ranked[:k]])
    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


relevance = {3: 0, 4: 1, 5: 1}
print(ndcg_at_k([3, 4, 5], relevance, 3))
# Approximately 0.6934264036

print(ndcg_at_k([4, 5, 3], relevance, 3))
# 1.0
```

### 6.2 真实标签从哪里来 / Where Ground-Truth Labels Come From

**中文：** 一个可解释的二元标签是：目标用户是否在评估窗口内与候选人成为好友，成为好友记为 `1`，否则记为 `0`。离线评估可以在时刻 `t` 的图上生成推荐，再用 `(t, t + delta]` 内新增的好友关系构造标签。标签对应的用户对必须处于明确的评估候选集合中；没有曝光或没有足够观察时间的用户，不能轻易视为确定的负例。

**English:** An interpretable binary label is whether the target user becomes friends with a candidate during the evaluation window: `1` for a new friendship and `0` otherwise. For offline evaluation, generate recommendations from the graph at time `t` and derive labels from new friendships during `(t, t + delta]`. Define the evaluation candidate set explicitly, and do not automatically treat users without exposure or sufficient observation time as confirmed negatives.

**中文：** 生成推荐时不能使用未来的边，否则会发生数据泄漏。也不要直接拿用于排序的共同好友数当真实标签：按同一分数正确排序的完整候选列表，其 NDCG 会恒为 `1`，无法评价真正的交友效果。

**English:** Do not use future edges when generating recommendations, because that leaks evaluation data. Using the same mutual-friend counts as both ranking scores and relevance labels gives a correctly sorted complete candidate list an NDCG of `1`, which does not evaluate actual friendship outcomes.

### 6.3 只返回 Top-K / Returning Only Top-K

**中文：** 计数步骤不变，完成计数后用堆选择前 `k` 个候选人，无需对全部 `c` 个候选人排序。Python 可以使用以下代码，排序规则仍是共同好友数降序、ID 升序：

**English:** Keep the counting step unchanged, then use a heap to select the top `k` candidates instead of sorting all `c` candidates. The same count-descending and ID-ascending ordering applies:

```python
import heapq

# counts is the counter produced by the recommendation traversal.
top_ids = heapq.nsmallest(k, counts, key=lambda v: (-counts[v], v))
```

**中文：** 当 `1 <= k < c` 时，查询时间可写为 `O(S + c log(k + 1))`；只取一个候选人时可线性扫描。当 `k >= c` 时使用全排序即可。堆选择阶段需要 `O(k)` 额外空间，但共同好友计数字典仍需要 `O(c)` 空间。

**English:** For `1 <= k < c`, query time can be written as `O(S + c log(k + 1))`; selecting a single candidate can use a linear scan. For `k >= c`, a full sort is appropriate. Heap selection needs `O(k)` additional space, while the mutual-friend counter still requires `O(c)` space.
