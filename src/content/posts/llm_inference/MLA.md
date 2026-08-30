---
title: "MLA"
published: 2026-08-29
description: "MLA"
image: ""
tags: ["llm_inference","MLA"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-29T05:43:12.305.265420138Z"
---

MLA 指 **Multi-head Latent Attention**。它基本沿用了 DeepSeek-V3 的 MLA 思路：**把每个 token 的 K/V 压缩到一个低维 latent，只缓存 latent，需要做 Attention 时再恢复/吸收到计算中**。

我们现在把前面所有东西串成一个完整的 MLA Attention 数值例子：

$\boxed{ Hidden \rightarrow Q \rightarrow c^{KV} \rightarrow K/V \rightarrow QK^T \rightarrow Softmax \rightarrow AV \rightarrow Attention\ Output }$

为了能手算，我们造一个“迷你 GLM”：

$hidden\_size=4$$num\_heads=2$$kv\_lora\_rank=2$

每个 Q/K/V head 都是 2 维。

暂时先不加 RoPE，最后再告诉你真实 GLM 的 RoPE 加在哪里。

## 1）输入三个 token

假设当前这一层已经得到了：

“猫”：

$h_{\text{猫}}=[1,2,0,1]$

“鱼”：

$h_{\text{鱼}}=[2,0,1,1]$

现在来了当前 token“吃”：

$h_{\text{吃}}=[1,1,2,0]$

这里的 hidden state 都是：

$1\times4$

我们统一使用横着写。

------

# 第一部分：先处理历史 token 的 KV

## 2）模型已经训练好了 $W^{DKV}$

假设：

$W^{DKV} = \begin{bmatrix} 1&0\\ 0&1\\ 1&0\\ 0&1 \end{bmatrix}$

形状：

$4\times2$

它负责：

$h_4 \rightarrow c^{KV}_2$

这就是玩具版的：

$kv\_lora\_rank=2$

------

## 3）“猫”压缩成 KV latent

$h_{\text{猫}}=[1,2,0,1]$

计算：

$c_{\text{猫}}^{KV} = h_{\text{猫}}W^{DKV}$

即：

$[1,2,0,1] \begin{bmatrix} 1&0\\ 0&1\\ 1&0\\ 0&1 \end{bmatrix}$

第一维：

$1\times1+ 2\times0+ 0\times1+ 1\times0 =1$

第二维：

$1\times0+ 2\times1+ 0\times0+ 1\times1 =3$

所以：

$\boxed{ c_{\text{猫}}^{KV}=[1,3] }$

------

## 4）“鱼”也压缩

$h_{\text{鱼}}=[2,0,1,1]$

计算：

$c_{\text{鱼}}^{KV} = h_{\text{鱼}}W^{DKV}$

第一维：

$2\times1+0\times0+1\times1+1\times0=3$

第二维：

$2\times0+0\times1+1\times0+1\times1=1$

所以：

$\boxed{ c_{\text{鱼}}^{KV}=[3,1] }$

------

# 第二部分：KV Cache

## 5）这时候 MLA 缓存什么？

历史里有：

“猫”：

$[1,3]$

“鱼”：

$[3,1]$

因此 KV Cache：

$\boxed{ C^{KV} = \begin{bmatrix} 1&3\\ 3&1 \end{bmatrix} }$

第一行是猫，第二行是鱼。

注意：

此时我们没有缓存：

$K_{\text{猫},1},K_{\text{猫},2}$

也没有缓存：

$V_{\text{猫},1},V_{\text{猫},2}$

只缓存：

$\boxed{c^{KV}}$

这就是 MLA 最关键的地方。

------

# 第三部分：当前 token 产生 Q

现在处理“吃”。

## 6）模型还有一个 $W^Q$

假设：

$W^Q= \begin{bmatrix} 1&0&0&1\\ 0&1&1&0\\ 1&0&1&0\\ 0&1&0&1 \end{bmatrix}$

因为有两个 Q heads，每个 2 维，所以总 Q 维度：

$2\times2=4$

计算：

$Q_{\text{吃}} = h_{\text{吃}}W^Q$

其中：

$h_{\text{吃}}=[1,1,2,0]$

------

## 7）逐项计算 Q

第一维：

$1\times1+1\times0+2\times1+0\times0 =3$

第二维：

$1\times0+1\times1+2\times0+0\times1 =1$

第三维：

$1\times0+1\times1+2\times1+0\times0 =3$

第四维：

$1\times1+1\times0+2\times0+0\times1 =1$

所以：

$Q_{\text{吃}} = [3,1,3,1]$

拆成两个 head：

$\boxed{Q_1=[3,1]}$$\boxed{Q_2=[3,1]}$

实际模型当然不会这么巧一样，只是我们的数字比较简单。

------

# 第四部分：从 latent 得到 K

## 8）训练好的 $W^{UK}$

假设：

$W^{UK} = \begin{bmatrix} 1&0&1&1\\ 0&1&1&-1 \end{bmatrix}$

形状：

$2\times4$

因为：

$c^{KV}:2$

要产生：

$2\text{ 个 head}\times2=4$

个 K 数字。

------

## 9）计算“猫”的 K

$c_{\text{猫}}=[1,3]$

计算：

$[1,3]W^{UK}$

得到：

第一维：

$1\times1+3\times0=1$

第二维：

$1\times0+3\times1=3$

第三维：

$1\times1+3\times1=4$

第四维：

$1\times1+3\times(-1)=-2$

所以：

$K_{\text{猫}}=[1,3,4,-2]$

拆开：

$\boxed{K_{\text{猫},1}=[1,3]}$$\boxed{K_{\text{猫},2}=[4,-2]}$

------

## 10）计算“鱼”的 K

$c_{\text{鱼}}=[3,1]$

计算：

$[3,1]W^{UK}$

得到：

$[3,1,4,2]$

所以：

$\boxed{K_{\text{鱼},1}=[3,1]}$$\boxed{K_{\text{鱼},2}=[4,2]}$

现在有：

Head 1：

$K_{\text{猫},1}=[1,3]$$K_{\text{鱼},1}=[3,1]$

Head 2：

$K_{\text{猫},2}=[4,-2]$$K_{\text{鱼},2}=[4,2]$

------

# 第五部分：真正开始 Attention

到这里才开始你熟悉的：

$QK^T$

------

## 11）Head 1 算 Attention Score

当前：

$Q_1=[3,1]$

先看它和“猫”的相关性：

$Q_1K_{\text{猫},1}^T$

也就是：

$[3,1]\cdot[1,3]$

得到：

$3\times1+1\times3 = 6$

再和“鱼”：

$[3,1]\cdot[3,1]$

得到：

$3\times3+1\times1 = 10$

所以：

$score_1=[6,10]$

------

## 12）还要除以 $\sqrt{d}$

标准 Attention：

$\frac{QK^T}{\sqrt{d_k}}$

这里：

$d_k=2$

所以：

$\sqrt2\approx1.414$

因此：

$score_1 = \left[ \frac6{1.414}, \frac{10}{1.414} \right]$

大约：

$[4.24,7.07]$

------

## 13）Softmax

计算：

$softmax([4.24,7.07])$

大约得到：

$\boxed{ A_1=[0.056,0.944] }$

意思是 Head 1：

对“猫”：

$5.6\%$

对“鱼”：

$94.4\%$

所以这个 head 明显更关注“鱼”。

------

# 第六部分：计算 V

## 14）模型训练好了 $W^{UV}$

还是使用前面的：

$W^{UV} = \begin{bmatrix} 1&1&0&2\\ 1&0&1&1 \end{bmatrix}$

也是：

$2\times4$

------

## 15）“猫”的 V

$c_{\text{猫}}=[1,3]$

计算：

$[1,3]W^{UV}$

得到：

$[4,1,3,5]$

拆成：

$\boxed{V_{\text{猫},1}=[4,1]}$$\boxed{V_{\text{猫},2}=[3,5]}$

------

## 16）“鱼”的 V

$c_{\text{鱼}}=[3,1]$

计算：

$[3,1]W^{UV}$

得到：

$[4,3,1,7]$

所以：

$\boxed{V_{\text{鱼},1}=[4,3]}$$\boxed{V_{\text{鱼},2}=[1,7]}$

------

# 第七部分：Head 1 用 Attention 权重对 V 求和

刚才 Head 1 得到：

$A_1=[0.056,0.944]$

所以：

$O_1 = 0.056V_{\text{猫},1} + 0.944V_{\text{鱼},1}$

代入：

$O_1 = 0.056[4,1] + 0.944[4,3]$

第一维：

$0.056\times4+0.944\times4 = 4$

第二维：

$0.056\times1+0.944\times3$$=0.056+2.832$$=2.888$

所以：

$\boxed{ O_1=[4,2.888] }$

这就是 Head 1 的 Attention 输出。

------

# 第八部分：Head 2 也做同样事情

## 17）Head 2 算 score

假设：

$Q_2=[3,1]$

“猫”的 K：

$K_{\text{猫},2}=[4,-2]$

所以：

$3\times4+1\times(-2) = 10$

“鱼”的 K：

$K_{\text{鱼},2}=[4,2]$

所以：

$3\times4+1\times2 = 14$

得到：

$[10,14]$

除：

$\sqrt2$

得到：

$[7.07,9.90]$

Softmax 仍然大约：

$[0.056,0.944]$

------

## 18）Head 2 对 V 加权

Head 2：

$V_{\text{猫},2}=[3,5]$$V_{\text{鱼},2}=[1,7]$

因此：

$O_2 = 0.056[3,5] + 0.944[1,7]$

第一维：

$0.056\times3+0.944\times1$$=1.112$

第二维：

$0.056\times5+0.944\times7$$=6.888$

所以：

$\boxed{ O_2=[1.112,6.888] }$

------

# 第九部分：两个 Head 拼起来

现在：

$O_1=[4,2.888]$$O_2=[1.112,6.888]$

拼起来：

$O= [4,2.888,1.112,6.888]$

然后真实 Transformer 还有一个：

$W^O$

输出投影：

$O_{final}=OW^O$

然后 residual：

$h'=h+O_{final}$

再继续后面的网络层。

------

# 第十部分：现在把整个 MLA 串起来

历史 token“猫”：

$h_{\text{猫}}=[1,2,0,1]$

经过：

$W^{DKV}$

得到：

$\boxed{c_{\text{猫}}=[1,3]}$

历史 token“鱼”：

$h_{\text{鱼}}=[2,0,1,1]$

得到：

$\boxed{c_{\text{鱼}}=[3,1]}$

KV Cache 实际主要存：

$\boxed{ \begin{bmatrix} 1&3\\ 3&1 \end{bmatrix} }$

然后当前 token“吃”：

$h_{\text{吃}}$

产生：

$Q_1,Q_2$

接下来逻辑上：

$c^{KV} \xrightarrow{W^{UK}} K$$c^{KV} \xrightarrow{W^{UV}} V$

然后正常 Attention：

$QK^T$$\downarrow$$softmax$$\downarrow$$AV$$\downarrow$

Attention Output。

整个链路就是：

$\boxed{ h \rightarrow Q }$

同时历史 token：

$\boxed{ h \xrightarrow{W^{DKV}} c^{KV} }$

然后：

$\boxed{ c^{KV} \rightarrow K,V }$

最后：

$\boxed{ Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt d} \right)V }$

所以你之前问：

>   MLA 难道不需要 Attention 了吗？

答案就是：

$\boxed{\text{MLA 当然还是 Attention}}$

它改的主要不是：

$QK^T\rightarrow softmax\rightarrow V$

这一套基本思想。

它改的是：

$\boxed{\text{历史 K/V 怎么表示和怎么缓存}}$

------

## 19）和普通 MHA 最直观的区别

普通 MHA 对“猫”直接缓存：

$K_{\text{猫},1}$$V_{\text{猫},1}$$K_{\text{猫},2}$$V_{\text{猫},2}$

我们这里一共是：

$2\times(2+2)=8$

个数字。

而 MLA 对“猫”主要缓存：

$\boxed{c_{\text{猫}}=[1,3]}$

只有：

$2$

个数字。

“鱼”也一样。

所以在这个玩具例子：

$\boxed{ MHA:\ 每token缓存8个数 }$$\boxed{ MLA:\ 每token缓存2个数 }$

真实 GLM 则是把这个：

$2$

换成：

$512$

再加独立的 RoPE 部分。

因此真实 GLM 更接近：

$\boxed{ h_{6144} \rightarrow c^{KV}_{512} }$

然后缓存：

$\boxed{ 512+64 }$

而不是完整缓存所有 64 个 head 的 K/V。

还有最后一个非常重要的优化我们这里故意没加进去：**真实高效 MLA 不需要真的先把所有历史 $c^{KV}$ 展开成完整 K/V 再做 Attention，而可以把 $W^{UK}$、$W^{UV}$ 吸收到计算的另外一边。**前面这个完整版本先理解清楚，再看“矩阵吸收”就会容易很多。











有对应的权重，但代码里通常不叫 `W_UQKV` 和 `W_DQKV`。MLA 把它们拆成了几组投影。

可以这样对应：

| 数学名称    | 作用                                   | vLLM 代码名称                 |
| ----------- | -------------------------------------- | ----------------------------- |
| \(W_{DQ}\)  | 把 hidden state 压缩成低维 Q           | `fused_qkv_a_proj` 的 Q 部分  |
| \(W_{UQ}\)  | 把低维 Q 展开成多个 query head         | `q_b_proj`                    |
| \(W_{DKV}\) | 把 hidden state 压缩成共享的 KV latent | `fused_qkv_a_proj` 的 KV 部分 |
| \(W_{UK}\)  | 从 KV latent 恢复每个 head 的 K        | `kv_b_proj` 拆出的 `W_UK_T`   |
| \(W_{UV}\)  | 从 KV latent 恢复每个 head 的 V        | `kv_b_proj` 拆出的 `W_UV`     |

### 下投影：类似你说的 \(W_{DQKV}\)

代码在：

[attention.py (line 276)](/data/home/xli49/lxy/vllm/vllm/models/deepseek_v32/attention.py:276)

```
self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
    hidden_size,
    [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
)
```

运行时：

```
qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]

q_c, kv_c, k_pe = qkv_lora.split(
    [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
    dim=-1,
)
```

它把三个下投影融合到一次矩阵乘法中：

```
hidden_states
      │
      ├─ W_DQ  → q_c
      ├─ W_DKV → kv_c
      └─ W_KR  → k_pe
```

所以它可以粗略理解成一个融合的：

```
W_DQKV
```

但严格来说，它是 \(W_{DQ}\)、\(W_{DKV}\) 和位置相关 K 投影的拼接。

### Q 的上投影：\(W_{UQ}\)

代码：

```
self.q_b_proj = ColumnParallelLinear(
    q_lora_rank,
    num_heads * qk_head_dim,
)
```

运行时：

```
q = self.q_b_proj(q_c)[0]
```

流程是：

```
hidden_states
    ↓ W_DQ
低维 q_c
    ↓ W_UQ
多个 query heads
```

这就是明确的 \(W_{UQ}\)。

### KV 的上投影：\(W_{UK}\) 和 \(W_{UV}\)

代码：

```
kv_b_proj = ColumnParallelLinear(
    kv_lora_rank,
    num_heads * (qk_nope_head_dim + v_head_dim),
)
```

它包含两个权重：

```
KV latent
   ├─ W_UK → 每个 head 的 K
   └─ W_UV → 每个 head 的 V
```

不过在 MLA decode 中，vLLM 不一定真正把完整 K/V 展开出来，而是通过“权重吸收”优化：

```
Q × W_UK
```

先把 \(W_{UK}\) 吸收到 Q 侧，再执行 attention。

对应代码中的：

```
self.W_UK_T
self.W_UV
```

例如：

```
ql_nope = torch.bmm(
    q_nope.transpose(0, 1),
    self.W_UK_T,
).transpose(0, 1)
```

注意力结束后再执行：

```
torch.bmm(attn_out, self.W_UV, out=out)
```

因此整体可以表示为：

```
hidden_states
   │
   ├─ W_DQ  → q_c ── W_UQ ──→ Q
   │
   └─ W_DKV → kv_c
                    │
                    ├─ W_UK → K
                    └─ W_UV → V
```

所以答案是：

-   有 \(W_{UQ}\)、\(W_{UK}\)、\(W_{UV}\)；
-   有 \(W_{DQ}\)、\(W_{DKV}\)；
-   vLLM 将几个下投影融合为 `fused_qkv_a_proj`；
-   没有一个正式叫作 `W_UQKV` 的单一权重，因为 Q、K、V 的上投影结构并不相同。





# Q不需要存储，但W^DQ W^UQ需要

因为“不存储 Q”指的是：

>   不保存以前 token 计算出来的 Q 激活。

并不是说“不需要生成 Q”。

`W_DQ` 和 `W_UQ` 是生成 Q 的模型权重，相当于生产 Q 的机器；Q 是机器每次生产出来的临时产品。

计算过程是：

```
当前 token 的 hidden_state
          ↓ W_DQ
压缩的 q_c
          ↓ W_UQ
当前 token 的 Q
          ↓
执行注意力计算
          ↓
使用完后丢弃 Q
```

用公式表示：

```
q_c = hidden_state × W_DQ
Q   = q_c × W_UQ
```

合起来相当于：

```
Q = hidden_state × W_DQ × W_UQ
```

也可以理解为把一个巨大的 `W_Q` 拆成两个较小的矩阵：

```
W_Q ≈ W_DQ × W_UQ
```

例如，假设：

```
hidden_state 维度 = 8
压缩 q_c 维度     = 2
最终 Q 维度       = 6
```

那么：

```
W_DQ: [8, 2]
W_UQ: [2, 6]
```

每次生成 token：

```
hidden_state [1, 8]
    × W_DQ [8, 2]
    = q_c [1, 2]

q_c [1, 2]
    × W_UQ [2, 6]
    = Q [1, 6]
```

Q 用来查询历史 KV：

```
Q × 历史 K
    ↓
注意力权重
    ↓
加权历史 V
```

计算完成后，当前 Q 就没用了，因为下一个 token 会产生新的 `hidden_state` 和新的 Q。

所以存储关系是：

```
W_DQ、W_UQ：模型参数，每一步都要重复使用，必须长期保存
当前 Q：中间计算结果，只用一次，临时保存
历史 Q：以后不会再用，不需要缓存
历史 KV：后续 token 会反复查询，必须缓存
```

最简单的类比：

>   `W_DQ/W_UQ` 是咖啡机，Q 是刚做出来的一杯咖啡。咖啡喝完不用保存，但咖啡机还要留着，因为下一次还要继续制作。
