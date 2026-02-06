---
title: "Transformer"
published: 2026-02-04
description: "Transformer"
image: ""
tags: ["llm","Transformer"]
category: llm
draft: false
lang: ""
---

# Transformer架构

![image-20260206003119904](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206003119904)



# 概念理解

## 模型怎么从训练W_Q和W_K

**训练前：** W 矩阵是随机的，Q乱找，K乱接，V乱传。

**训练中：** 通过海量数据的**“完形填空”**任务，如果预测错了就通过**反向传播**修改 W 矩阵。

**结果：** 被迫形成统计规律——看到“苹果”的 X，自动生成指向“动作”的 Q。

------

### 1. 初始状态：它什么都不知道

在模型刚出生时（初始化阶段），W_Q 和 W_K 里的数字都是随机生成的。

-   **苹果的 X：** `[1, 0, 2]` (代表食物)
-   **吃的 X：** `[0, 1, 0]` (代表动作)

这时候，W_Q 是乱的。

-   “苹果”生成的 Q 可能是：`[0.1, -0.5, 0]` —— 这代表它可能在找“铁块”或者“云朵”，反正不是找“吃”。
-   **结果：** 苹果 Q 和 吃 K 的匹配分数极低。模型没注意到“吃”。

### 2. 训练过程：海量数据的“暴力洗脑”

现在，我们将几十亿句人类的话喂给模型看。其中包括成千上万句类似这样的话：

>   “我**吃**了一口**苹果**。”
>
>   “他把**苹果**削皮后**吃**了。”
>
>   “**苹果**很好**吃**。”

**第一轮训练（失败）：**

1.  模型看到“苹果”。
2.  模型瞎猜：我觉得“苹果”后面应该接“爆炸”。
3.  **正确答案：** 后面是“吃”。
4.  **计算惩罚（Loss）：** 老师（损失函数）狠狠地打了一把尺子：“错！苹果和爆炸没关系！苹果和吃才有关系！”

### 3. 关键时刻：反向传播（修改参数）

这时候，数学上的**梯度下降（Gradient Descent）**开始工作了。它会回溯整个过程，问一个问题：

>   **“为了让‘苹果’和‘吃’不仅在物理距离上近，在注意力上也‘吸’在一起，我需要怎么改 W_Q 和 W_K 的参数？”**

系统会自动微调矩阵里的数字：

-   **修改 W_Q：** “喂！下次再看到 X 是 `[1,0,2]` (苹果) 这种‘食物向量’进来的时候，你算出来的 Q 必须往‘动作类’的方向偏！”
-   **修改 W_K：** “喂！下次再看到 X 是 `[0,1,0]` (吃) 这种‘动作向量’进来的时候，你算出来的 K 必须能接住‘食物类’的查询！”

### 4. 最终结果：形成了“搭配默契”

经过几百万次的“瞎猜 -> 被骂 -> 改正”循环后，W_Q 终于学乖了（也就是参数收敛了）。

现在的 W_Q 变成了一个**“经验丰富的媒婆”**：

-   一旦输入是 **“苹果”**（特征：食物、圆的），W_Q 就会自动生成一个 Q 向量，这个向量的**第 2 维（动作维）**数值很高。
-   一旦输入是 **“吃”**，W_K 生成的标签 K，恰好在**第 2 维**也是高数值。

**点积运算（Match）：**

![image-20260206003135785](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206003135785)

------

### 总结

机器“知道”要找“吃”，并不是因为它理解了这一口咬下去的脆甜口感。

而是因为它在几亿次训练中发现：

**如果不把“苹果”的 Query 指向“吃”的 Key，它就会预测错误，Loss 就会变高。**

为了生存（降低 Loss），它**被迫学会**了：

-   看到“主语”，就去找“谓语”。
-   看到“苹果”，就去找“吃/买/洗/削”。
-   看到“红色的”，就去找“花/衣服/苹果”。

这就是所谓的**“统计规律”**。也就是我们之前那个矩阵例子里，为什么输入是 0，乘以 W_Q 却变成了 5 的根本原因——是海量阅读训练出来的**条件反射**。



## 原始输入X是什么

原始输入 X 是一个**“大杂烩”**。它包含了这个词的所有信息：词性、语义、单复数、情感色彩、位置等等。

The original input ( X ) is a **“mixed bundle.”**It contains all the information about the token: part of speech, semantics, singular or plural form, sentiment, position, and so on.





## 为什么不直接用 X 计算？

因为要**解耦（Decoupling）**：

-   **Q != K：** **“我是谁”不等于“我想找谁”**。如果不拆开，模型只能做简单的“找同类”，做不了复杂的逻辑推断（如主谓搭配）。
-   **X != V：** **“原始输入”不等于“有用信息”**。原始 X 是个包含位置、语法、语义的大杂烩；V 是经过矩阵 W_V **提纯**后的结果，只保留当前任务需要的特征。



## QKV运作机制

**输入：** 原始向量 X。

**变换：** X 分别乘以三个**参数矩阵** (W_Q, W_K, W_V)，生成 Q、K、V。

**注意力：** Q 和 K 算出分数（匹配度），根据分数去加权混合 V。

**输出：** 得到一个新的、融合了上下文信息的向量。



## QKV矩阵存了什么信息

-   **Q (我想找什么？意图):** 你在搜索框里输的内容（比如“好吃的川菜”）。
-   **K (我是什么？用于匹配的索引):** 网页给自己打的标签/标题（比如“麻婆豆腐”、“宫保鸡丁”）。
-   **V (我有什么货？实际的内容):** 网页里的正文内容（具体的菜谱做法）。

### 情况1：如果只有 Q 和 V（没有 K）

**这意味着：K = V**。 也就是：**网页的标签 = 网页的正文**。

-   **逻辑后果：** 你是拿你的搜索词（Q）直接去和正文（V）做匹配。
-   **为什么不行？**
    -   **匹配困难：** 假设你想搜“美国总统”，但某个网页的正文（V）里全是“乔·拜登...”。如果你搜“总统”，可能根本匹配不到这个网页，因为正文里没怎么提“总统”这两个字，全是人名。
    -   **信息冗余：** 匹配不仅仅需要语义，还需要功能性的标签。V 往往包含了太多细节，拿来做索引（匹配）太嘈杂了。
-   **结论：** 我们需要一个**高度抽象的摘要（Key）**来专门负责被检索，而不是让原本的内容（Value）亲自出来应付检索。



### 情况 2：如果只有 K 和 V（没有 Q）

**这意味着：Q = K**（或者 Q 是静态的）。 也就是：**我（作为一个词）去搜索别人的时候，只能用我自己的“身份标签”去搜。**

-   **逻辑后果：** “匹配”变成了“找同类”。
-   **例子：** 句子是“猫 吃 鱼”。
    -   **“吃”**这个词，如果 Q=K，它的搜索请求就是“我是动词，我是吃”。
    -   它去匹配**“鱼”**，鱼的 K 是“我是名词，我是鱼”。
    -   **问题来了：** “吃”和“鱼”在向量空间里通常长得很不一样（语义距离远）。如果强行用“吃”的向量去点乘“鱼”的向量，分数可能很低。
-   **为什么需要 Q？**
    -   Q 允许“吃”这个词**“变身”**。
    -   “吃”的 K 说：我是“吃”。
    -   “吃”的 Q 说：我想找一个“食物”。
    -   这时候，Q（食物）就能完美匹配“鱼”的 K（食物）。
-   **结论：** **你是谁（Key）**和**你想找谁（Query）**通常是两码事。如果没有 Q，模型只能做“连连看”（找相似），而不能做“逻辑组合”（找互补）。





### 情况 3：如果只有 Q 和 K（没有 V）

**这意味着：V = K**（或者 V = Q）。 也就是：**算出了关注度之后，直接把标签（Key）拿过来加权求和。**

-   **逻辑后果：** 传递的信息量被锁死了。
-   **例子：** 还是“猫 吃 鱼”。
    -   “吃”关注到了“鱼”。
    -   但“鱼”传回来的信息（V）必须等于它的标签（K）。
    -   假设 K 只是一个简单的标签向量（比如代表“名词”），那么“吃”在这个过程中只学到了“后面有个名词”，而丢失了“这是一条红烧鲤鱼”这种丰富的语义细节（原本应存储在 V 中）。
-   **结论：** **用于匹配的特征（Key）**通常是低维的、抽象的；而**需要传递的内容（Value）**需要是丰富的、具体的。让它们分开，各自就能在不同的维度空间里优化。





## Token Embedding 

```python
token_embed = embedding_table[token]      # 词向量
```

### 1）Token Embedding 是怎么得到的？

本质：一个「查表操作」训练前会有一个矩阵：

```
embedding_table.shape = [vocab_size, d_model]
```

例如：

```
vocab_size = 50000
d_model    = 768
```

所以：

```
embedding_table = 50000 × 768 的矩阵
```



### 2）每个 token 对应一行向量

假设：

```
"今天" → token_id = 1032
```

那么：

```python
token_embed = embedding_table[1032]
```

得到：

```
[0.12, -0.33, 0.98, ..., 0.07]   # 768维向量
```



### 3）这些向量怎么来的？

**不是人工设定的**，而是：

>   在训练语言模型时，通过反向传播自动学出来的。

训练目标：

```
让模型更容易预测下一个词
```

于是 embedding 会逐渐学到：

-   “今天”和“昨天”向量接近
-   “苹果”和“香蕉”接近
-   “跑”和“吃”距离远

------

## Position Encoding

Transformer **本身不懂顺序**，因为：

```
Self-Attention = 全连接
```

所以必须告诉模型：

>   “谁在第几个位置”。

------

### 方法1）经典正弦位置编码（论文原版）

公式：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

意思：

-   每个位置生成一个 **d_model维向量**
-   用不同频率的 **sin / cos 波**

优点：

-   不用训练
-   能外推到更长序列

------

### 方法2）：可学习位置编码（GPT常用）

直接定义：

```
position_encoding.shape = [max_seq_len, d_model]
```

比如：

```
[2048 × 768]
```

然后：

```python
pos_embed = position_encoding[i]
```

这些向量：

>   和 embedding 一样，通过训练学出来。

------

## 为什么要把Token embeding&Position embedding两者相加？

**用向量叠加表达“语义 + 位置”** Use vector addition to represent “semantics + position.”

```python
embeddings.append(token_embed + pos_embed)
```

原因：

token 代表“是什么词”

```
"今天" → 语义
```

position 代表“在第几位”

```
第3个词 → 顺序信息
```

相加 = 同时包含两种信息

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)

        positions = torch.arange(seq_len)
        positions = positions.unsqueeze(0)

        return self.token_embed(token_ids) + self.pos_embed(positions)
```

------



## 为什么 Attention 一定要用 Q·Kᵀ，而不是别的相似度？





## 为什么除以Square root of d?

Scaled Dot-Product Attention

![image-20260206003157672](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206003157672)

如果维度增大而不缩放，点积会变得很大，导致 softmax 输出非常极端，让模型只关注少数位置，忽略其他信息。这样会使模型难以平衡地捕捉全局，因此我们缩放点积，让注意力分布更均匀，从而更有效学习上下文关系。

### 为什么当softmax中存在数值较大的元素时， 梯度会趋近于0？

```
# Softmax的梯度公式
∂softmax(x_i) / ∂x_j = {
    softmax(x_i) * (1 - softmax(x_i))   if i == j
    -softmax(x_i) * softmax(x_j)        if i ≠ j
}

# 当x中有很大的值时
x = [1, 2, 100]  # 100非常大
p = softmax(x) = [0, 0, 1]  # 几乎是one-hot

# 对最大值位置(i=2)求梯度:
∂p_2/∂x_2 = p_2 * (1 - p_2)
          = 1.0 * (1 - 1.0)
          = 1.0 * 0
          ≈ 0  ← 梯度消失!

# 对其他位置(i=0)求梯度:
∂p_0/∂x_0 = p_0 * (1 - p_0)
          = 0.0 * (1 - 0.0)
          = 0.0 * 1.0
          ≈ 0  ← 梯度也消失!
```







## 什么是logit?

logit = Softmax / Sigmoid 之前的原始分数（未归一化的输出）

>线性层输出 → logit → 再经过 Softmax / Sigmoid → 概率



| 任务       | 概率结构       | 激活    | logit 含义                      |
| ---------- | -------------- | ------- | ------------------------------- |
| **二分类** | 一个 (p)       | Sigmoid | **log-odds**                    |
| **多分类** | (\sum p_i=1)   | Softmax | **log-probabilities（差常数）** |
| **多标签** | 每个独立 (p_k) | Sigmoid | **多个 log-odds**               |

### Sigmoid logit

**The logit in Sigmoid equals the log-odds of the probability.**Used for **binary or multi-label classification**, where probabilities **do not need to sum to 1**.



### softmax logit

**Softmax logits are unnormalized log-probabilities,** differing from true log-probabilities by **a constant shared across all classes**(the **log-partition function**). **Softmax 的 logits 是未归一化的对数概率（log-probabilities），** **与真实的 log 概率只相差一个对所有类别相同的常数（log-partition constant）。**

<img src="https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260206003216337" alt="image-20260206003216337" style="zoom:50%;" />







## Encoder的"理解"

-   把输入文本映射成**固定的向量表示**(contextualized embeddings)
-   这个表示捕捉了文本的语义、语法、上下文信息
-   可以**双向看**整个句子(既看左边也看右边的词)
-   目标是得到一个好的表示,用于下游任务

**例子**: BERT看到"苹果"这个词时,会根据上下文判断是"公司"还是"水果",生成对应的向量表示

### Encoder的工作流程(以BERT为例)

```
# 输入句子
input_text = "我爱自然语言处理"
tokens = ["我", "爱", "自然", "语言", "处理"]

# ========== 第一步: Token Embedding ==========
token_embeddings = []
for token in tokens:
    embedding = embedding_table[token]  # 查表得到初始向量
    token_embeddings.append(embedding)
# 结果: 每个词变成一个向量,比如 512维

# ========== 第二步: 加上位置信息 ==========
for i in range(len(token_embeddings)):
    token_embeddings[i] += positional_encoding[i]

# ========== 第三步: 多层Transformer Encoder ==========
hidden_states = token_embeddings

for layer in range(12):  # 假设12层
    # --- Self-Attention (关键!) ---
    # 每个token可以看到所有其他token
    Q = hidden_states @ W_Q  # Query
    K = hidden_states @ W_K  # Key  
    V = hidden_states @ W_V  # Value
    
    # 计算注意力分数(每个词对其他所有词)
    attention_scores = softmax(Q @ K.T / sqrt(d_k))
    
    # 注意力矩阵示例:
    #           我    爱    自然  语言  处理
    #    我   [0.3  0.1   0.2   0.2   0.2]
    #    爱   [0.2  0.4   0.1   0.1   0.2]
    #  自然   [0.1  0.1   0.3   0.3   0.2]  <- "自然"关注"语言"
    #  语言   [0.1  0.1   0.3   0.3   0.2]  <- "语言"关注"自然"
    #  处理   [0.1  0.2   0.2   0.2   0.3]
    
    # 加权求和得到新的表示
    attention_output = attention_scores @ V
    
    # --- Feed Forward ---
    hidden_states = FeedForward(attention_output)
    # 加上残差连接和LayerNorm等...

# ========== 最终输出 ==========
final_representations = hidden_states
# 结果: 每个token都有一个"理解后"的向量
# final_representations[0] = "我"在这个句子中的语义向量
# final_representations[2] = "自然"在这个句子中的语义向量(知道后面跟着"语言处理")
```





## Decoder的"生成"

-   逐步**产生新的token**
-   只能**单向看**(只看已生成的左边部分)
-   目标是预测/生成下一个词

```
# ============================================
# Decoder生成过程(自回归)
# ============================================

def decoder_generate(prompt, max_length=20):
    """
    prompt: 初始文本,如 "今天天气"
    max_length: 最多生成多少个词
    """
    
    # 初始化:已生成的token序列
    generated_tokens = tokenize(prompt)  # ["今天", "天气"]，真实LLM推理中是数字generated_tokens = [1032, 5871, 924, ...]
    
    # 逐个生成新token
    for step in range(max_length):
        
        # ===== 步骤1: 准备输入 =====
        input_sequence = generated_tokens  # 当前已生成的所有token
        seq_length = len(input_sequence)
        
        # ===== 步骤2: Token Embedding =====
        embeddings = []
        for i, token in enumerate(input_sequence):
            token_embed = embedding_table[token]      # 词向量
            pos_embed = position_encoding[i]          # 位置编码
            embeddings.append(token_embed + pos_embed)
        
        hidden_states = embeddings  # shape: [seq_length, d_model]
        
        # ===== 步骤3: 创建Causal Mask =====
        causal_mask = create_causal_mask(seq_length)
        """
        假设当前seq_length=3 ["今天", "天气", "很"]
        
        causal_mask:
              今天  天气   很
        今天 [  0  -inf -inf]
        天气 [  0    0  -inf]
        很   [  0    0    0 ]
        
        保证:
        - "今天"只能看自己
        - "天气"能看"今天"+"天气"
        - "很"能看"今天"+"天气"+"很"
        """
        
        # ===== 步骤4: 多层Decoder处理 =====
        for layer in decoder_layers:  # 假设12层
            
            # --- Masked Self-Attention ---
            Q = hidden_states @ W_Q
            K = hidden_states @ W_K
            V = hidden_states @ W_V
            
            # 计算注意力分数并加上mask
            attention_scores = (Q @ K.T) / sqrt(d_k)
            attention_scores = attention_scores + causal_mask  # 关键!
            
            # Softmax(上三角的-inf变成0)
            attention_weights = softmax(attention_scores, dim=-1)
            """
            attention_weights (假设当前处理"很"):
                  今天  天气   很
            很  [ 0.3  0.5  0.2]  ← 只能看左边!看不到未来
            """
            
            # 加权求和
            attention_output = attention_weights @ V
            
            # --- Feed Forward Network ---
            hidden_states = FeedForward(attention_output)
        
        # ===== 步骤5: 预测下一个token =====
        # 只用最后一个位置的hidden state
        last_hidden_state = hidden_states[-1]  # 最后一个token的表示
        
        # 通过输出层得到词表上的概率分布
        logits = last_hidden_state @ output_matrix  # shape: [vocab_size]
        probabilities = softmax(logits)
        """
        probabilities:
        {
            "好": 0.35,
            "不错": 0.25,
            "冷": 0.15,
            "热": 0.10,
            ...
        }
        """
        
        # 选择概率最高的token(或采样)
        next_token = argmax(probabilities)  # 贪心:选概率最大的
        # 或者:
        # next_token = sample(probabilities)  # 采样:按概率随机选
        
        # ===== 步骤6: 添加到序列 =====
        generated_tokens.append(next_token)
        
        # 如果生成了结束符,停止
        if next_token == "<EOS>":
            break
    
    return generated_tokens


# ============================================
# 辅助函数
# ============================================

def create_causal_mask(seq_length):
    """创建因果掩码"""
    # 下三角为0,上三角为-inf
    mask = []
    for i in range(seq_length):
        row = []
        for j in range(seq_length):
            if j <= i:
                row.append(0)      # 可以看到
            else:
                row.append(-inf)   # 看不到未来
        mask.append(row)
    return mask


# ============================================
# 运行示例
# ============================================

prompt = "今天天气"
result = decoder_generate(prompt, max_length=5)

print(result)
# 可能输出: ["今天", "天气", "很", "好", "呢", "<EOS>"]
```

```
# 生成循环的本质:

第1轮:
输入: ["今天", "天气"]
     ↓ Decoder处理(只看左边)
预测: "很"
结果: ["今天", "天气", "很"]

第2轮:
输入: ["今天", "天气", "很"]
     ↓ Decoder处理(只看左边)
预测: "好"
结果: ["今天", "天气", "很", "好"]

第3轮:
输入: ["今天", "天气", "很", "好"]
     ↓ Decoder处理(只看左边)
预测: "呢"
结果: ["今天", "天气", "很", "好", "呢"]

...循环直到生成<EOS>或达到最大长度
```









## Encoder 和 Decoder 的区别

| Property                | Encoder (BERT)                                 | Decoder (GPT)                                    |
| ----------------------- | ---------------------------------------------- | ------------------------------------------------ |
| **Attention direction** | Bidirectional (can attend to the full context) | Unidirectional (attends only to previous tokens) |
| **Masking**             | No causal mask (may use padding mask only)     | Causal mask (upper-triangular masking)           |
| **Primary objective**   | Understanding / classification                 | Next-token generation                            |
|                         |                                                |                                                  |



## Causal Mask



### 方法1: 用1表示"保留",0表示"遮挡"

```python
# Boolean mask(布尔掩码)
mask = np.tril(np.ones((5, 5)))

"""
        我   爱  自然 语言 处理
   我 [[ 1.  0.  0.  0.  0.]    ← 1=保留, 0=遮挡
   爱  [ 1.  1.  0.  0.  0.]
 自然  [ 1.  1.  1.  0.  0.]
 语言  [ 1.  1.  1.  1.  0.]
 处理  [ 1.  1.  1.  1.  1.]]
"""

# 使用时:
scores = Q @ K.T / sqrt(d_k)
# 把0的位置替换成-inf
scores = scores.masked_fill(mask == 0, -inf)
attention_weights = softmax(scores)
```

### 方法2: 用0表示"保留",-inf表示"遮挡"

```python
# Additive mask(加法掩码)
mask = np.triu(np.ones((5, 5)) * -np.inf, k=1)

"""
        我    爱   自然  语言  处理
   我 [[  0. -inf -inf -inf -inf]   ← 0=保留, -inf=遮挡
   爱  [  0.   0. -inf -inf -inf]
 自然  [  0.   0.   0. -inf -inf]
 语言  [  0.   0.   0.   0. -inf]
 处理  [  0.   0.   0.   0.   0.]]
"""

# 使用时:直接相加
scores = Q @ K.T / sqrt(d_k)
scores = scores + mask  # 直接加上-inf
attention_weights = softmax(scores)
```

### 主流框架的选择

#### PyTorch (方法1为主)

```python
import torch
import torch.nn.functional as F

seq_len = 5
# 创建下三角矩阵: 1=保留, 0=遮挡
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

"""
tensor([[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]])
"""

# 使用
scores = torch.randn(seq_len, seq_len)
scores = scores.masked_fill(~mask, float('-inf'))  # ~mask反转:False变True
attn = F.softmax(scores, dim=-1)
```

#### Transformer库的实现

```python
# HuggingFace Transformers 常用方法2
def _prepare_decoder_attention_mask(attention_mask, input_shape):
    # 创建因果掩码
    batch_size, seq_length = input_shape
    
    # 下三角为0,上三角为-inf
    causal_mask = torch.full(
        (seq_length, seq_length), 
        float('-inf')
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    
    """
    tensor([[  0., -inf, -inf, -inf, -inf],
            [  0.,   0., -inf, -inf, -inf],
            [  0.,   0.,   0., -inf, -inf],
            [  0.,   0.,   0.,   0., -inf],
            [  0.,   0.,   0.,   0.,   0.]])
    """
    
    return causal_mask
```

#### 直观对比

```python
# 方法1: Boolean Mask
下三角=1(保留)  上三角=0(遮挡)
[[ 1.  0.  0.]
 [ 1.  1.  0.]
 [ 1.  1.  1.]]

使用方式: scores.masked_fill(mask == 0, -inf)

# 方法2: Additive Mask  
下三角=0(保留)  上三角=-inf(遮挡)
[[  0. -inf -inf]
 [  0.   0. -inf]
 [  0.   0.   0.]]

使用方式: scores + mask
```

#### 为什么会有两种方式?

1.  **方法1(Boolean)**: 更直观,1就是"要",0就是"不要"
2.  **方法2(Additive)**: 更高效,直接加法操作,不需要条件判断

**Causal Mask本身没有固定的0/1定义**,关键看:

-   **保留(能看到)的位置**: 1 或 0
-   **遮挡(看不到)的位置**: 0 或 -inf

但**最终效果（Softmax后的注意力权重）都一样**:

```python
# Softmax后的注意力权重
[[1.0  0   0  ]    ← 第1个词只看自己
 [0.4 0.6  0  ]    ← 第2个词看前2个
 [0.2 0.3 0.5]]    ← 第3个词看前3个
```

上三角永远是0(注意力权重为0),这才是causal mask的本质!





## 为什么GPT不需要encoder

-   它的自注意力机制在预测下一个词时,已经对前文进行了编码和理解
-   不需要先用encoder把整个输入"理解"成向量,再传给decoder





## Encoder-Decoder 架构

-   编码器把输入序列转换为上下文表示 `memory`，解码器利用该表示并结合已生成的目标序列逐步预测下一个输出。
-   自回归生成机制。解码器在每个时间步都会把之前生成的目标 token 作为输入的一部分，再预测当前 token。

------

### EncoderDecoder 类

#### 1）类的核心作用

该类封装完整的编码与解码流程，统一管理编码器、解码器、嵌入层和最终生成器。

#### 2）初始化 `__init__`

-   `encoder` 负责理解源序列
-   `decoder` 根据上下文生成目标序列
-   `src_embed` 提供源词向量表示
-   `tgt_embed` 提供目标词向量表示
-   `generator` 将解码结果映射为词表概率

初始化阶段的本质是把模型各个功能模块组装成完整结构。

------

#### 3）forward 前向传播

模型先对输入序列进行编码得到上下文表示，再利用该表示指导解码器生成目标序列。

```python
return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
```

------

#### 4）encode 方法

将文本转换成包含语义信息的向量表示。

-   `self.tgt_embed(tgt)`：对目标序列 `tgt` 进行词嵌入，将离散的 token 映射为连续向量表示，作为解码器的输入。
-   `memory`：编码器输出的上下文表示，包含源序列的全局语义信息，用于解码阶段的注意力查询。
-   `src_mask`：源序列的 padding 掩码，用于避免解码器在注意力计算时关注填充位置，从而保证注意力仅作用于真实 token。
-   `tgt_mask`：目标序列的自回归掩码，通过屏蔽未来位置，使当前位置只能依赖已生成的历史 token，从而保证序列按时间步逐步生成。

```python
return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

调用解码器后，内部主要依次执行以下计算过程：

1.  **Masked Multi-Head Self-Attention**
    在目标序列内部进行带掩码的自注意力计算，仅关注已生成的历史位置。

2.  **Encoder–Decoder Attention**
    利用当前解码状态对编码器输出的上下文表示 `memory` 进行注意力查询，以引入源序列信息。

3.  **Feed Forward Network**
    对每个位置的表示进行逐位置的非线性变换，以提升表达能力。

4.  **Add & Norm**
    通过残差连接与层归一化稳定训练过程，并保留原始特征信息。

    

------

#### 5）decode 方法

解码器根据目标序列嵌入信息和编码得到的上下文信息逐步生成输出表示。

-   `self.tgt_embed(tgt)`：对目标序列 `tgt` 进行 **词嵌入（embedding）**，把离散 token 转成连续向量表示，作为解码器输入。

-   `memory `：这是 **编码器输出的上下文表示**，包含整句源序列的信息，供解码器在生成每个词时进行注意力查询。

-   `src_mask`源：序列的 **padding mask**，用于

    -   防止解码器在注意力计算时关注到源序列中的填充位置

    -   保证注意力只落在真实 token 上

-   `tgt_mask`：目标序列的 **自回归 mask**，作用是：屏蔽未来位置

    -   让当前位置只能看到**已经生成的历史 token**
    -   保证序列按时间步逐步生成

```python
return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

调用 decoder 后，内部会依次执行：

1.  **Masked Multi-Head Self-Attention**
     只关注目标序列中已生成的部分。
2.  **Encoder-Decoder Attention**
     使用当前解码状态去关注 `memory`（源句信息）。
3.  **Feed Forward 网络**
     对每个位置做非线性变换。
4.  **Add & Norm 残差归一化**
     稳定训练并保留原信息。



------

### Generator 类

#### 1）类的作用

该模块把解码器输出的隐藏向量转换为词汇表上的概率分布。

#### 2）初始化

```python
self.proj = nn.Linear(d_model, vocab)
```

线性层将模型隐藏维度映射到词表大小。

#### 3）forward 计算

```python
return log_softmax(self.proj(x), dim=-1)
```

先通过线性变换得到各词得分，再用 `log_softmax` 转换为对数概率分布。









# 相关概念

## QKV



用 Q 和 K 的相似度决定权重， 再用这些权重对 V 做加权求和。

**Q·K → 谁更相关**

**softmax → 关注比例**

**加权 V → 得到新表示**





## V 在“当前层”到底装的是什么？

现在可以精确定义：

>   **V 向量装的是：
>    该词在这一语义加工阶段
>    能提供给整句理解的那部分信息。**

例如：

-   浅层 V(man) → `[名词, 人类]`
-   中层 V(man) → `[宾语, 被看见对象]`
-   深层 V(man) → `[事件参与者, 语义角色]`

Transformer 的每一层
 都在重新解释“这个词现在意味着什么”。

## 更深一层：模型在学什么？

训练其实在学三件事：

W_Q 在学：**什么信息值得去找**

W_K 在学：**我应该以什么方式被别人匹配**

W_V 在学：**如果被关注，我该提供什么语义内容**

























