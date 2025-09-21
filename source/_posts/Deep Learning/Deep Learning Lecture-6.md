---
title: Deep Learning Lecture-6
date: 2025-06-21
tags:
  - DeepLearning
math: true
syntax_converted: true

---

## Transformers

### Transformers: Attention is All You Need

[[Deep Learning Lecture-5#Attention]]

再次理解Attention的概念：类似于”查字典“的操作，对于Query $q$, Key $k$和Value $v$，计算相关性，也就是重要性，对于输出序列中的第$i$个输出有价值的信息：
$$
w_{ij} = a(q_i, k_j)
$$
其中$a$是一个函数，可以是内积、*Additive Attention*等。对于输出序列中的第$i$个输出，计算当前的输出的$q_i$，计算与输入序列中的$k_j$的相关性，然后对于$v_j$进行加权求和（这是一种寻址操作），得到的$c_i$是查字典所得到的信息：
$$
c_i = \sum_{j=1}^T w_{ij}v_j
$$
**希望找到一种更好的计算方法**。

在[[Deep Learning Lecture-5#RNN with Attention]]中问题在于：
- 太复杂的模型
- 某种意义上使用的Attention已经足够使用，不再需要循环网络
- 循环网络的计算是串行的，不能有效加速
#### Self-Attention

计算的是同一条序列中的不同位置之间的相关性，也就是自注意力。对于输入序列中的第$i$个位置，计算与其他位置的相关性，然后对于所有的位置进行加权求和：
规定Query $Q = [q_1 \dots q_n]$，Key $K = [k_1 \dots k_n]$，Value $V = [v_1 \dots v_k]$，则：

![Pasted image 20250323133751](Pasted image 20250323133751.png)

#### Scaled Dot-Product Attention

我们认为使用一个网络来计算相关性太复杂了，当两个向量是相同维度的时候可以直接计算内积。在这里，在计算先引入参数，使得其维度是一样的，从而可以计算内积：

*Scaled Dot-Product* :
$$
a(q,k) = \frac{q^T k}{\sqrt{d_k}}
$$
使得变换前后的方差是一样的，这样可以使得梯度更加稳定，否则可能进入激活函数的饱和区。
$$
Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
上面的式子将得到的$n \times n$的矩阵进行softmax操作，在归一化的过程中，**是某一个query在所有的key上的注意力分配一定是$\mathbf{1}$**。后面是对于Value的加权求和。**在上面的公式中，$Q$、$K$和$V$中的向量都是行向量，进行softmax操作时也是在同一行上操作**。

![Pasted image 20250323141917](Pasted image 20250323141917.png)

对于同一组输入，经过不同的线性变换得到的不同的Query、Key和Value，在样本数量为$m$的情况下，可以进行计算：

$$
\begin{aligned}
&W^Q \in \mathbb{R}^{d_k \times d_{\text{input}}}, \\
&W^K \in \mathbb{R}^{d_k \times d_{\text{input}}}, \\
&W^V \in \mathbb{R}^{d_v \times d_{\text{input}}} \\
&Q = X W^Q \in \mathbb{R}^{m \times d_k}, \\
&K = X W^K \in \mathbb{R}^{m \times d_k}, \\
&V = X W^V \in \mathbb{R}^{m \times d_v}. \\
&QK^T \in \mathbb{R}^{m \times m}, \\
&\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \\
&\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \in \mathbb{R}^{m \times d_v}.

\end{aligned}
$$

 **维度总结表**

| 矩阵/操作                          | 维度                          | 说明                                    |
| ------------------------------ | --------------------------- | ------------------------------------- |
| 输入矩阵 $X$                       | $m \times d_{\text{input}}$ | 包含 $m$ 个样本，每个样本维度为 $d_{\text{input}}$ |
| 查询矩阵 $Q$                       | $m \times d_k$              | 每个样本的查询向量维度为 $d_k$                    |
| 键矩阵 $K$                        | $m \times d_k$              | 每个样本的键向量维度为 $d_k$                     |
| 值矩阵 $V$                        | $m \times d_v$              | 每个样本的值向量维度为 $d_v$                     |
| 注意力得分矩阵 $QK^T$                 | $m \times m$                | 样本间的注意力强度矩阵                           |
| 最终输出 $\text{Attention}(Q,K,V)$ | $m \times d_v$              | 聚合所有样本的加权值信息，输出维度为 $d_v$              |

#### Multi-Head Attention

注意到上面的注意力的表达能力是相当有限的，在language model同一个词和其他不同的词之间可能有很多种不同的关系，仅仅用一种简单的关系来表示是不够的。所以我们引入多头注意力，希望能在不同的侧面上进行表达。
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$
其中：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) 
$$
其中$W_i^Q, W_i^K, W_i^V$是不同的线性变换，$W^O$是最后的线性变换，最后进行的维度的规约操作。
与CNN相比，CNN的不同的通道之间与上一层的每一个通道之间都是有连接的；但是在这里，不同的头之间是没有连接的，这样可以使得不同的头可以关注不同的信息。

不同的头之间是可以并行计算的，这样可以加速计算；但是缺点是内存占用会很大。

#### Position-wise Feed-Forward Networks

![Pasted image 20250323145847](Pasted image 20250323145847.png)
在标准 Transformer Block 中，注意力层之后会接一个前馈神经网络（FFN），其结构如下：

1. 输入张量形状：  
$$
   X \in \mathbb{R}^{B \times L \times H \times D}
   $$
   - $B$：Batch size  
   - $L$：序列长度  
   - $H$：Attention 头数  
   - $D$：每个头的维度  

2. 将多头输出合并：  
   $$
   X' = \mathrm{reshape}(X,\, (B,\,L,\,H\cdot D)) \in \mathbb{R}^{B \times L \times (H\!D)}
   $$

3. 两层“卷积”全连接结构  
   这里所谓“卷积”，实际上等价于在最后一维上对每个位置独立地做 1×1 卷积（与 RNN 中在不同时间步共享参数的思想一致），并在卷积核后加入非线性 ReLU。  
   $$
   \begin{aligned}
   Z_1 &= \mathrm{ReLU}\bigl(X' W_1 + b_1\bigr),\quad W_1\in\mathbb{R}^{(H\!D)\times d_{ff}},\; b_1\in\mathbb{R}^{d_{ff}}\\
   Z_2 &= Z_1 W_2 + b_2,\quad W_2\in\mathbb{R}^{d_{ff}\times(H\!D)},\; b_2\in\mathbb{R}^{(H\!D)}
   \end{aligned}
   $$
   - 第一层升维到中间维度 $d_{ff}$（例如 $2048$）  
   - 第二层降维回原始维度 $H\!D$（例如 $512$）  

4. 加残差 & LayerNorm  
   $$
   Y = \mathrm{LayerNorm}\bigl(X' + Z_2\bigr)\in\mathbb{R}^{B\times L\times(H\!D)}
   $$

5. （可选）reshape 回多头排列：  
   $$
   Y' = \mathrm{reshape}(Y,\, (B,\,L,\,H,\,D))
   $$


- **线性限制**：除去 Attention 中的 SoftMax，若只堆线性层，模型表达能力较弱；引入 ReLU 后能够拟合更复杂的非线性函数。  
- **稀疏权重**：Attention 的 SoftMax 本质上生成了一组“稀疏”权重，负责学习不同位置间的依赖；FFN 则负责在每个位置上“干净”地抽取该词的内部特征，避免无谓的跨词干扰。  
- **卷积视角**：将 FFN 看作**对最后一维的 1×1 卷积**，等同于对每个位置独立但在所有位置共享参数，这与 RNN 在时间步上共享权重的假设一致。  
- **增强表达**：Attention 解决了上下文依赖，FFN 则补强了单位置特征提取，两者协同提升了 Transformer 的整体表达能力。

---

> **注意**：以上操作对每个批次（$B$）中每个序列位置（$L$）都独立执行，参数在所有位置间共享。


#### Residual Connection

在上面的操作中，这些操作都是有排列不变性。
残差是一个标准的操作，这样可以让网络更好地记录位置编码。

#### Layer Normalization

目的是使得每一层经过Attention和Feed-Forward之后的输出的分布是一样的，这样可以使得梯度更加稳定。
[[Deep Learning Lecture-5#Layer Normalization]]

![Pasted image 20250323151056](Pasted image 20250323151056.png)

#### Value Embedding

- **目的**：将原始输入特征（如温度、流量、股价等数值）从低维（通常是 1 或几个通道）映射到高维向量空间，使模型能在更大维度下学习更丰富的特征表示。
- **做法**：通过一个线性层或 1×1 卷积（`TokenEmbedding`）把每个时间步的原始向量映成长度为 `d_model` 的向量。

- **实现方式**：通过一个线性映射或 $1 \times 1$ 卷积完，对于有$c_{in}$个特征维度的输入，使用卷积核的大小是$[d_{model},c_{in},1]$
- **输入维度**：$(\text{batch\_size}, \text{seq\_len}, c_{in})$
- **输出维度**：$(\text{batch\_size}, \text{seq\_len}, d_{model})$

#### Positional Encoding /Position Embedding

位置信息是顺序信息的一种泛化的形式。如果采用独热编码，这是一种类别信息而不是一个顺序信息，不同的是不可以比的。所以引入*position embedding*，这是一个矩阵，效果类似于一个查找表。查找操作在这里就是一个矩阵乘上一个独热编码的操作，这是因为GPU在矩阵乘法操作上是非常高效的。
但是独热编码会带来下面的问题
- **高维稀疏性**：  独热编码的维度等于序列最大长度（如512），导致向量稀疏且计算效率低下（尤其对长序列）。
- **无法泛化到未见长度**：  若训练时序列最大长度为512，模型无法处理更长的序列


**引入归纳偏好**：
- 每个位置的编码应该是独一无二的且是确定的
- 认为两个位置的距离应该是一致的
- 应该生成一个有界的值，位置数随着序列长度的增加而增加

Google的实现是使用的正弦和余弦函数的组合：
$$
e_i(2j) = \sin\left(\frac{i}{10000^{2j/d_{\text{model}}}}\right)
$$
$$
e_i(2j+1) = \cos\left(\frac{i}{10000^{2j/d_{\text{model}}}}\right)
$$
上述公式中的$i$指的句子中的第$i$个位置，$j$指的是位置编码的维度，$d_{\text{model}}$是位置编码的维度。这样的编码是满足上面的归纳偏好的。

#### Temporal Embedding

- **目的**：针对时间序列中特有的“时间属性”——小时、星期几、月份、季节等——进行编码，让模型学到周期性（如日周期、周周期、年周期）和节假日效应等信息。
- **做法**：通常把每个时间属性（hour-of-day, day-of-week, month 等）也映射到 `d_model` 维度，然后把这些属性向量加起来或拼接后再降维。`TemporalEmbedding` 类会根据 `embed_type`（如 `'fixed'` 或 `'learned'`）和 `freq`（如 `'h'`、`'d'`）来决定具体细节。

#### Encoder

编码器中使用的是多头注意力、逐位置前馈网络和位置编码。在这个编码器中是一个直筒式的网络，好处是调参较为简单。

缺点：
- 二次复杂度
- 参数量过大
- 很多的头是冗余的

训练阶段要使用多个头，发现有些头的权重较低，可以在推理阶段去掉这些头。

#### Decoder

##### Autoregressive

![Pasted image 20250323161530](Pasted image 20250323161530.png)

预测阶段一定要使用滚动预测，这是一个自回归的状态，但是这是一个串行的操作，会比较慢。但是在训练阶段这样是不能接受的，我希望训练的不同阶段可以并行计算，但是这里要求在一开始输入所有的序列，所以这里需要**遮挡**。
在算Attention的时候，对于当前的位置，只能看到之前的位置，不能看到之后的位置。

![Pasted image 20250323163151](Pasted image 20250323163151.png)

在编码器上是不能用的，因为防止解码器在训练时利用未来的目标序列信息（即“作弊”），确保模型逐步生成的能力与推理阶段一致。训练过程中仍然需要真实标签作为目标输出，但掩码限制了模型在生成当前词时对未来的访问。

#### Encoder-Decoder Attention

计算的是解码器的输出和编码器的输出之间的相关性，这里的Query是解码器的输出，Key和Value是编码器的输出。

![Pasted image 20250323185530](Pasted image 20250323185530.png)


![Pasted image 20250323185720](Pasted image 20250323185720.png)

注意这里是将编码器的输出输入到解码器中的每一层的Encoder-Decoder Attention中。这里是神经网络中的**特征重用**思想，并且解码器中的网络是直筒式的，所以这些特征是可以重用的。

#### RNN vs. Transformer

- RNN是串行的，Transformer是并行的
- 对于有严格偏序关系的序列，RNN可能更适合
- 对于长序列，Transformer更适合
- 对于较小的数据量，Transformers参数量较大，表现可能不如RNN

![Pasted image 20250323190623](Pasted image 20250323190623.png)

![Pasted image 20250323190856](Pasted image 20250323190856.png)
### X-formers Variance with Improvements

[[2106.04554] A Survey of Transformers](https://arxiv.org/abs/2106.04554)

![Pasted image 20250323191632](Pasted image 20250323191632.png)

![Pasted image 20250323191637](Pasted image 20250323191637.png)

#### Lineariezd Attention

#### Flow Attention

### GPT: Generative Pre-trained Transformer

#### Transfer Learning

先将一个模型预训练好，然后在特定的任务上进行微调。一般而言，预训练的过程是无监督的，优点是可以使用大规模数据。

#### Pre-Training

![Pasted image 20250324190821](Pasted image 20250324190821.png)

- 直接使用的是Transformers中的block，但是这里使用12层
- 只使用decoder没有encoder，因为这不是一个机器翻译的任务
- 在计算损失函数的过程中，使用的似然函数是最大似然估计，在实际中使用一个参数化的网络来近似需要的概率。

#### Supervised Fine-Tuning

对于不同的任务，需要更换模型的输出头，并且还要使用新的损失函数。关注上下文建模。
![Pasted image 20250324193704](Pasted image 20250324193704.png)

最后是使用无监督训练的损失函数和有监督训练的损失函数的加权和，这是一个**多任务学习**。当微调的数据比较少的时候，可以使用无监督训练的损失函数的权重较大。

对于不同的下游任务，要进行任务适配*Task Specific Adaptation*。对于不同的下游任务，可以使用不同的头。
![Pasted image 20250324194137](Pasted image 20250324194137.png)

#### GPT-2 & GPT-3

Zero-shot learning：在没有看到训练数据的情况下，直接在测试集上进行预测。通过在预训练阶段使用大规模的数据，可以使得模型具有更好的泛化能力，这样可以提高在一些常见问题上的表现。

### BERT: Bidirectional Encoder Representations from Transformers

与GPT不同的是，BERT是双向的，可以看到上下文的信息。
![Pasted image 20250324195035](Pasted image 20250324195035.png)

BERT在encoder阶段就使用了mask，这样可以使得模型在训练的时候不会看到未来的信息。在训练的过程中随机地mask掉一些词，然后预测这些词。如果遮挡的词太少，那么模型得到的训练不够， 如果遮挡的词太多，那么得到的上下文就很少。
在训练的过程中就使用了102种语言。
特征工程：使用了更多的特征，引入了更多的embedding
是多个任务的联合训练，这样可以使得模型更加通用。

#### RoBERTa: A Robustly Optimized BERT Pretraining Approach

经过充分的调参和更长的训练时间，使得模型的表现更好。
证明了BERT中的下句预测是没有用的，因为在RoBERTa中去掉了这个任务。
mask的pattern可以动态调整

#### ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

低秩分解，减少参数量
![Pasted image 20250324203227](Pasted image 20250324203227.png)

跨层参数共享：可以让模型更加稳定

#### T5: Text-to-Text Transfer Transformer

迁移是泛化的高级形式：可以将多种文本任务统一为文本到文本的形式，这样可以使得模型更加通用。

架构层面的创新：
![Pasted image 20250324203639](Pasted image 20250324203639.png)

这里使用的是prefix-LM，这样可以使得模型更加通用。

### Vision Transformer

#### ViT

将一个图像变成一个patch
增加一个Position Embedding，于是得到各个patch的特征的加权平均。
 主要的贡献是将图像转换为序列，从而可以使用transformers来进行建模。在这个之前，普遍的观点是transformers只能用于文本数据，而CNN用于图像数据。

#### Swim Transformer

将CNN中的一些归纳偏好引入，可以使用局部的注意力，但是在一定程度上能捕捉全局的信息，通过Shifted Window Mechanism来实现。
层次化特征：
![Pasted image 20250324205304](Pasted image 20250324205304.png)

密集预测任务对于层次化特征需求更高，于是这个模型的表现是更好地。

#### DETR

![Pasted image 20250324205845](Pasted image 20250324205845.png)

![Pasted image 20250324205921](Pasted image 20250324205921.png)

### Fundation Models

 $$
 e^{\alpha+ \beta i } = e^{\alpha} e^{\beta i} = e^{\alpha}(\cos \beta + i \sin \beta)
 $$