---
title: "deep learning questions"
published: 2026-05-31
description: "deep learning questions"
image: ""
tags: ["llm","deep learning questions"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-31T14:43:57.693.939769891Z"
---

# Fundamentals

## 1）**What is deep learning, and how is it different from traditional machine learning?**

Deep learning uses multi-layer neural networks to automatically learn features. Traditional machine learning usually requires manually designed features.
深度学习使用多层神经网络自动学习特征。传统机器学习通常需要人工设计特征。

“A neural network has many hidden layers, and each layer processes the features, allowing the model to transform simple features into complex ones.”  “神经网络有许多隐藏层，每一层处理特征，允许模型将简单特征转换为复杂特征。”

<br>

## 2）**What are the basic components of a neural network?**

A neural network includes input layers, hidden layers, output layers, weights, biases, and activation functions.
神经网络包括输入层、隐藏层、输出层、权重、偏置和激活函数。

The input layer takes in the raw data. Hidden layers are the internal layers that process the information. The output layer gives the final prediction. Weights are the adjustable parameters connecting neurons. Biases shift the output. Activation functions introduce non-linearity, allowing the network to model complex patterns. 

输入层接收原始数据。隐藏层是处理信息的内层。输出层给出最终的预测。权值是连接神经元的可调参数。偏见会改变输出。激活函数引入非线性，允许网络模拟复杂的模式。

<br>

## 3）**What is an artificial neuron, and how does it compute its output?**

Artificial neurons are mathematical or computer models that merely simulate the functions of biological neurons.

人工神经元是仅仅模拟生物神经元功能的数学或计算机模型。

A neuron receives inputs, multiplies them by weights, adds a bias, and then gets the output through an activation function.
神经元接收输入，乘以权重并加上偏置，再通过激活函数得到输出。

<br>

## 4）**What is forward propagation?**

Forward propagation passes input data through the neural networks and produces the predictions.
前向传播把输入数据传过神经网络，并产生预测结果。

<br>

## 5）**What is backpropagation?**

Backpropagation calculates gradients of the loss and updates model parameters.
反向传播计算损失函数的梯度，并更新模型参数。

A gradient tells the model how to change its parameters to reduce the loss. `new parameter = old parameter - learning rate × gradient` The model updates its parameters by moving in the opposite direction of the gradient to reduce the loss. 

梯度告诉模型如何改变参数以减少损失。“新参数=旧参数-学习率×梯度”模型通过与梯度相反的方向移动来更新参数，以减少损失。

<br>

## 6）**Why does backpropagation rely on the chain rule?**

Because neural networks are composite functions, the chain rule can calculate gradients layer by layer ==from the output back to earlier layers..==
因为神经网络是复合函数，链式法则可以逐层计算梯度。

<br>

## 7）**What is a computational graph?**

A computational graph is a graph that represents mathematical operations and their dependencies. It helps organize computations and enables automatic differentiation.

计算图是一种表示数学运算及其依赖关系的图。它有助于组织计算并实现自动微分。

<br>

## 8）**What are parameters and hyperparameters?**

Parameters are values learned by the model during training, such as weights and biases. Hyperparameters are set before training, such as the learning rate, batch size, and number of epochs.

参数是模型在训练过程中学习到的值，比如权重和偏差。在训练前设置超参数，如学习率、批大小和epoch数。

<br>

## 9）**What are epoch, batch, and iteration?**

==An epoch means training once on the full dataset. A batch means a small group of training examples.An iteration is one update of the model parameters using one batch..==
epoch 是完整训练一遍数据集。batch 是一小批数据。iteration 是一次参数更新。

<br>

## 10）**What is mini-batch training?**

Mini-batch training updates model parameters using a small batch of data at a time. It is efficient and relatively stable.
Mini-batch训练每次使用少量数据更新模型参数。它效率高，并且相对稳定。

<br>

## Activation Functions

11）**Why do neural networks need activation functions?**
Activation functions introduce nonlinearity and allow neural networks to learn complex patterns.
激活函数引入非线性，使神经网络能够学习复杂模式。

12）**Why must activation functions be nonlinear?**
Without nonlinearity, multiple layers are still equivalent to one linear layer.
如果没有非线性，多层网络仍然等价于一个线性层。

13）**What are the characteristics and limitations of Sigmoid?**
Sigmoid maps values to the range from zero to one, but it can cause gradient vanishing.
Sigmoid 把数值映射到零到一之间，但容易导致梯度消失。

14）**How does Tanh improve upon Sigmoid?**
Tanh maps values to the range from minus one to one and is centered around zero.
Tanh 把数值映射到负一到一之间，并且以零为中心。

15）**Why is ReLU widely used?**
ReLU is simple and fast, and it can reduce gradient vanishing in the positive region.
ReLU 简单快速，并且可以在正数区域缓解梯度消失。

16）**What is dying ReLU?**
Dying ReLU means some neurons always output zero and can no longer learn.
神经元死亡指某些神经元一直输出零，无法继续学习。

17）**What are Leaky ReLU, PReLU, and ELU?**
Leaky ReLU allows small negative outputs. PReLU learns the negative slope. ELU makes negative outputs smoother.
Leaky ReLU 允许负数区域有小输出。PReLU 会学习负数斜率。ELU 让负数输出更平滑。

18）**Why is GELU used in Transformers?**
GELU is smoother than ReLU and is commonly used in Transformer models.
GELU 比 ReLU 更平滑，并且常用于 Transformer 模型。

19）**What is Softmax used for?**
Softmax converts logits into probabilities for multi-class classification.
Softmax 把未归一化分数转换为多分类概率。

20）**What is the difference between Softmax and Sigmoid?**
Softmax is used for mutually exclusive multi-class classification. Sigmoid is used for binary classification or multi-label classification.
Softmax 用于互斥的多分类任务。Sigmoid 用于二分类或多标签分类任务。

## Loss Functions

21）**What is a loss function?**
A loss function measures the difference between predictions and true labels and guides model training.
损失函数衡量预测结果和真实标签之间的差异，并指导模型训练。

22）**What losses are used for regression?**
Common regression losses include mean squared error, mean absolute error, and Huber loss.
常见回归损失包括均方误差、平均绝对误差和 Huber 损失。

23）**What losses are used for classification?**
Common classification losses include cross-entropy loss and binary cross-entropy loss.
常见分类损失包括交叉熵损失和二元交叉熵损失。

24）**What is the difference between mean squared error and mean absolute error?**
Mean squared error penalizes large errors more. Mean absolute error is more robust to outliers.
均方误差对大误差惩罚更强。平均绝对误差对异常值更鲁棒。

25）**Why is cross-entropy suitable for classification?**
Cross-entropy measures the difference between predicted probability distribution and true label distribution.
交叉熵衡量预测概率分布和真实标签分布之间的差异。

26）**What is the difference between binary cross-entropy and categorical cross-entropy?**
Binary cross-entropy is used for binary classification. Categorical cross-entropy is used for multi-class classification.
二元交叉熵用于二分类。多分类交叉熵用于多分类。

27）**What is KL divergence?**
KL divergence measures the difference between two probability distributions.
KL 散度衡量两个概率分布之间的差异。

28）**What is focal loss?**
Focal loss focuses more on hard samples and is often used to handle class imbalance.
焦点损失更关注困难样本，常用于处理类别不平衡。

29）**What is contrastive loss?**
Contrastive loss pulls similar samples closer and pushes different samples farther apart.
对比损失让相似样本更接近，让不同样本更远离。

30）**What is triplet loss?**
Triplet loss uses an anchor sample, a positive sample, and a negative sample to learn better representations.
三元组损失使用锚样本、正样本和负样本来学习更好的表示。

## Optimization and Training

31）**What is gradient descent?**
Gradient descent updates parameters in the opposite direction of the gradient to reduce the loss.
梯度下降沿着梯度的反方向更新参数，以降低损失。

32）**What are batch gradient descent, stochastic gradient descent, and mini-batch stochastic gradient descent?**
Batch gradient descent uses all data. Stochastic gradient descent uses one sample. Mini-batch stochastic gradient descent uses a small batch of data.
批量梯度下降使用全部数据。随机梯度下降使用一个样本。小批量随机梯度下降使用一小批数据。

33）**Why is learning rate important?**
Learning rate controls the step size of parameter updates and directly affects convergence.
学习率控制参数更新的步长，并直接影响收敛。

34）**What happens if the learning rate is too large or too small?**
If the learning rate is too large, training may diverge. If the learning rate is too small, training will be slow.
如果学习率太大，训练可能发散。如果学习率太小，训练会很慢。

35）**What are common learning rate schedules?**
Common learning rate schedules include step decay, exponential decay, cosine decay, and warmup.
常见学习率策略包括阶梯衰减、指数衰减、余弦衰减和学习率预热。

36）**What is warmup?**
Warmup gradually increases the learning rate at the beginning of training to make training more stable.
学习率预热是在训练初期逐渐增大学习率，使训练更稳定。

37）**What is momentum?**
Momentum uses previous gradients to smooth the update direction and speed up training.
动量利用历史梯度平滑更新方向，并加快训练。

38）**What is RMSProp?**
RMSProp uses recent squared gradients to adjust the learning rate of each parameter.
RMSProp 使用近期梯度平方来调整每个参数的学习率。

39）**How does Adam work?**
Adam combines momentum and adaptive learning rates to update parameters.
Adam 结合动量和自适应学习率来更新参数。

40）**What is the difference between Adam and stochastic gradient descent?**
Adam adaptively adjusts the learning rate for each parameter. Stochastic gradient descent uses a simpler update rule.
Adam 为每个参数自适应调整学习率。随机梯度下降使用更简单的更新规则。

41）**What is the difference between AdamW and Adam?**
AdamW separates weight decay from gradient updates and makes regularization more reasonable.
AdamW 将权重衰减和梯度更新分离，使正则化更合理。

42）**What is weight decay?**
Weight decay penalizes large weights and helps reduce overfitting.
权重衰减惩罚过大的权重，并帮助减少过拟合。

43）**What is gradient clipping?**
Gradient clipping limits the size of gradients and prevents gradient explosion.
梯度裁剪限制梯度大小，并防止梯度爆炸。

44）**What are local minima, saddle points, and flat minima?**
Local minima are points with lower loss nearby. Saddle points are flat in some directions. Flat minima usually have better generalization.
局部最小值是在附近损失更低的点。鞍点在某些方向较平坦。平坦极小值通常泛化更好。

45）**Why are saddle points common in deep learning?**
In high-dimensional space, there are many directions, so saddle points are more likely than bad local minima.
在高维空间中方向很多，所以鞍点比差的局部最小值更常见。

## Initialization and Normalization

46）**Why is weight initialization important?**
Good weight initialization keeps gradients stable and helps the model converge.
好的权重初始化能保持梯度稳定，并帮助模型收敛。

47）**What problems come from poor initialization?**
Poor initialization can cause gradient vanishing, gradient explosion, or slow training.
不好的初始化会导致梯度消失、梯度爆炸或训练缓慢。

48）**What is Xavier initialization?**
Xavier initialization keeps the variance of activations and gradients stable across layers.
Xavier 初始化让各层激活值和梯度的方差保持稳定。

49）**Why is He initialization suitable for ReLU?**
He initialization considers that ReLU removes negative values, so it keeps signal variance stable.
He 初始化考虑了 ReLU 会去掉负值，因此能保持信号方差稳定。

50）**What is Batch Normalization?**
Batch Normalization normalizes activations using the mean and variance of a batch.
批归一化使用一个批次的均值和方差对激活值进行归一化。

51）**How is Batch Normalization different during training and inference?**
During training, it uses the statistics of the current batch. During inference, it uses running statistics.
训练时使用当前批次的统计量。推理时使用滑动统计量。

52）**Why can Batch Normalization accelerate training?**
Batch Normalization stabilizes activation distributions and allows a larger learning rate.
批归一化稳定激活分布，并允许使用更大的学习率。

53）**What is the difference between Layer Normalization and Batch Normalization?**
Batch Normalization normalizes across the batch dimension. Layer Normalization normalizes across the feature dimension of one sample.
批归一化在批次维度上归一化。层归一化在单个样本的特征维度上归一化。

54）**Why is Layer Normalization common in Transformers?**
Layer Normalization does not depend on batch size and is suitable for sequence models.
层归一化不依赖批大小，并且适合序列模型。

55）**What is RMSNorm?**
RMSNorm uses root mean square for normalization and does not subtract the mean.
RMSNorm 使用均方根进行归一化，并且不减去均值。

## Regularization and Generalization

56）**What are overfitting and underfitting?**
Overfitting means the model performs well on training data but poorly on new data. Underfitting means the model is too simple and performs poorly on training data.
过拟合是模型在训练数据上表现好，但在新数据上表现差。欠拟合是模型太简单，在训练数据上表现也差。

57）**How do you detect overfitting?**
If training loss is low but validation loss is high, the model is likely overfitting.
如果训练损失低但验证损失高，模型可能过拟合。

58）**How can overfitting be prevented?**
Overfitting can be reduced by using more data, data augmentation, dropout, weight decay, early stopping, or simpler models.
可以通过使用更多数据、数据增强、随机失活、权重衰减、早停或更简单的模型来减少过拟合。

59）**What is Dropout?**
Dropout randomly disables some neurons during training and makes the model more robust.
随机失活在训练时随机关闭部分神经元，并使模型更鲁棒。

60）**How is Dropout different during training and inference?**
During training, some neurons are randomly dropped. During inference, all neurons are used.
训练时随机丢弃部分神经元。推理时使用全部神经元。

61）**Why does data augmentation improve generalization?**
Data augmentation creates more diverse samples and helps the model learn stable features.
数据增强生成更多样的样本，并帮助模型学习稳定特征。

62）**What is Early Stopping?**
Early stopping stops training when validation performance no longer improves.
早停是在验证集表现不再提升时停止训练。

63）**What is label smoothing?**
Label smoothing makes hard labels softer and prevents the model from becoming too confident.
标签平滑让硬标签更柔和，并防止模型过度自信。

64）**What is model ensembling?**
Model ensembling combines multiple models to improve robustness and performance.
模型集成结合多个模型，以提升鲁棒性和性能。

65）**What are training, validation, and test sets used for?**
The training set is used to train the model. The validation set is used to tune hyperparameters. The test set is used to evaluate final performance.
训练集用于训练模型。验证集用于调超参数。测试集用于评估最终性能。

## CNN

66）**Why are CNNs suitable for images?**
CNNs use local connections and weight sharing, which are suitable for image structure.
卷积神经网络使用局部连接和权重共享，适合图像结构。

67）**What is convolution?**
Convolution uses filters to scan local regions and extract features.
卷积使用卷积核扫描局部区域并提取特征。

68）**What are kernel size, stride, and padding?**
Kernel size is the size of the filter. Stride is the moving step. Padding adds values around the input.
卷积核大小是滤波器的大小。步长是移动距离。填充是在输入周围补充数值。

69）**How do you calculate convolution output size?**
Output size equals input size plus padding, minus kernel size, divided by stride, plus one.
输出尺寸等于输入尺寸加上填充，减去卷积核大小，再除以步长，最后加一。

70）**What is receptive field?**
Receptive field is the input region that affects one output unit.
感受野是影响一个输出单元的输入区域。

71）**What is pooling used for?**
Pooling reduces spatial size, reduces computation, and improves translation robustness.
池化降低空间尺寸，减少计算量，并提升平移鲁棒性。

72）**What is the difference between max pooling and average pooling?**
Max pooling keeps the strongest feature. Average pooling keeps average information.
最大池化保留最强特征。平均池化保留平均信息。

73）**What is one by one convolution?**
One by one convolution changes channel dimensions and mixes channel information.
一乘一卷积改变通道维度，并融合通道信息。

74）**What is depthwise separable convolution?**
Depthwise separable convolution separates spatial filtering and channel mixing to reduce computation.
深度可分离卷积把空间滤波和通道融合分开，以减少计算量。

75）**What problem does ResNet solve?**
ResNet uses residual connections to solve the degradation problem in deep networks.
残差网络使用残差连接来解决深层网络的退化问题。

## RNN and Sequence Models

76）**What data are RNNs suitable for?**
RNNs are suitable for sequential data, such as text, speech, and time series.
循环神经网络适合序列数据，比如文本、语音和时间序列。

77）**Why do vanilla RNNs struggle with long-term dependencies?**
Vanilla RNNs easily suffer from gradient vanishing, so they are bad at keeping long-term information.
普通循环神经网络容易出现梯度消失，所以不擅长保留长期信息。

78）**What is the core structure of LSTM?**
LSTM uses cell states and gates to control information flow.
长短期记忆网络使用细胞状态和门控机制来控制信息流动。

79）**What do LSTM gates do?**
The input gate controls new information. The forget gate controls old information removal. The output gate controls final output.
输入门控制新信息进入。遗忘门控制旧信息丢弃。输出门控制最终输出。

80）**What is the difference between GRU and LSTM?**
GRU has fewer gates and no separate cell state, so it is simpler and faster.
门控循环单元的门更少，并且没有单独的细胞状态，所以更简单更快。

81）**What is a bidirectional RNN?**
A bidirectional RNN processes a sequence forward and backward to use past and future context.
双向循环神经网络从前向后和从后向前处理序列，以利用过去和未来上下文。

82）**What is Seq2Seq?**
Seq2Seq uses an encoder and a decoder to convert an input sequence into an output sequence.
序列到序列模型使用编码器和解码器，把输入序列转换为输出序列。

83）**Why does Seq2Seq need attention?**
Attention helps the decoder focus on relevant input tokens instead of relying only on one fixed vector.
注意力机制帮助解码器关注相关输入词，而不是只依赖一个固定向量。

84）**What is Teacher Forcing?**
Teacher forcing feeds the true previous token into the decoder during training.
教师强制是在训练时把真实的前一个词输入给解码器。

85）**What are common RNN training problems?**
Common problems include gradient vanishing, gradient explosion, slow training, and poor parallelism.
常见问题包括梯度消失、梯度爆炸、训练慢和并行能力差。

## Attention and Transformer

86）**What is attention?**
Attention allows the model to focus on the most relevant parts of the input.
注意力机制让模型关注输入中最相关的部分。

87）**What is self-attention?**
Self-attention calculates relationships between tokens in the same sequence.
自注意力计算同一个序列中词与词之间的关系。

88）**How does scaled dot-product attention work?**
Scaled dot-product attention compares queries and keys, applies Softmax, and uses the weights to combine values.
缩放点积注意力比较查询和键，经过 Softmax 得到权重，再用权重组合值。

89）**Why divide by the square root of the key dimension?**
This prevents dot products from becoming too large and keeps Softmax stable.
这样可以防止点积过大，并保持 Softmax 稳定。

90）**What is multi-head attention?**
Multi-head attention allows the model to focus on different positions and feature spaces at the same time.
多头注意力让模型同时关注不同位置和不同特征空间。

91）**Why does Transformer need positional encoding?**
Transformer has no recurrence or convolution, so positional encoding provides word order information.
Transformer 没有循环结构或卷积结构，所以位置编码提供词序信息。

92）**What is the difference between absolute and relative positional encoding?**
Absolute positional encoding represents exact positions. Relative positional encoding represents distances between tokens.
绝对位置编码表示具体位置。相对位置编码表示词之间的距离。

93）**What is the difference between Transformer encoder and decoder?**
The encoder understands the input. The decoder generates the output and uses masked self-attention.
编码器理解输入。解码器生成输出，并使用掩码自注意力。

94）**What is masked attention?**
Masked attention prevents the model from seeing future tokens during generation.
掩码注意力防止模型在生成时看到未来的词。

95）**What are the pros and cons of Transformers?**
Transformers have strong parallelism and can model long-range dependencies, but they need more data and computation.
Transformer 并行能力强，并且能建模长距离依赖，但需要更多数据和计算资源。

## Large Models and Engineering Practice

96）**What are pretraining and fine-tuning?**
Pretraining learns general knowledge from large-scale data. Fine-tuning adapts the model to a specific task.
预训练从大规模数据中学习通用知识。微调让模型适应具体任务。

97）**What is transfer learning?**
Transfer learning transfers knowledge learned from one task to another task.
迁移学习把一个任务中学到的知识迁移到另一个任务中。

98）**What are parameter-efficient fine-tuning and LoRA?**
Parameter-efficient fine-tuning updates only a small number of parameters. LoRA adds low-rank matrices instead of updating all weights.
参数高效微调只更新少量参数。LoRA 通过加入低秩矩阵来代替更新全部权重。

99）**How would you debug poor validation performance?**
I would check data quality, data split, labels, learning rate, overfitting, underfitting, and model capacity.
我会检查数据质量、数据划分、标签、学习率、过拟合、欠拟合和模型容量。

100）**What should be considered when deploying a deep learning model?**
Deployment should consider latency, throughput, model size, memory, accuracy, monitoring, and model drift.
部署时需要考虑延迟、吞吐量、模型大小、内存、准确率、监控和模型漂移。
