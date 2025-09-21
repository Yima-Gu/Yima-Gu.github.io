---
title: Deep Learning Lecture-3
date: 2025-06-21
tags:
  - DeepLearning
math: true
syntax_converted: true

---
## Convolutional Neural Networks

卷积网络最早是用来处理图像的问题。目前较为成功的研究是物体识别问题，对于物体之间的关系推断依然是计算机视觉的前沿领域。

在生物的研究中，存在**感受野***receptive field*的概念，这是指神经元对于输入的局部区域的敏感程度。在卷积网络中，我们也引入了这个概念。相邻的神经元处理的是相邻的图像区域，这样的设计使得网络能够捕捉到图像的空间结构。在模拟人类的视觉系统中，重要的是**提取不同程度的特征**，这就是卷积网络的核心思想。

在MLP中，由于对于原图像像素进行展开，会损失部分的空间信息。在处理视角、光照的变化时，MLP的效果会变差。

![{71895E3A-0EFA-438D-8E6C-EDBD4A337F5A}]({71895E3A-0EFA-438D-8E6C-EDBD4A337F5A}.png)

从左到右，逐步提取的是从较低到较高的层次特征。在不同的任务中使用不同的特征。在识别人物中，应该使用较高层次的特征。

##### Local Connectivity

每一层的神经元只连接到上一层的局部区域。具体的连接方式为：每一个神经元只关注上一层对应区域的部分神经元。

##### Parameter Sharing

一个神经元对应的某组参数代表的是某种特征，我们认为提取的特征的在不同的位置是一样的。是一种**平移不变性**。

#### Convolution

$f$是输入的图像，$g$是卷积核，卷积操作定义为：
*Continuous function* :
$$
(f*g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau
$$
*Discrete function* :
$$
(f*g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m] 
$$
上述操作的目的为对于输入函数

##### Cross Correlation

互相关用来分析两个信号的相似性，定义为：
*Continous function* : 
$$
(f\star g)(t) = \int_{-\infty}^{\infty} \overline {f(\tau)}g(t+\tau)d\tau
$$
*Discrete function* :
$$
(f\star g)[n] = \sum_{m=-\infty}^{\infty} \overline {f[m]}g[n+m]
$$
对于卷积和互相关的关系为：
$$
[f(t) \star g(t)] (t) = [\overline{f(-t)} * g(t)](t)
$$
事实上，在卷积网络中使用的是互相关操作。

##### Notation

- 输入的图片一般为三层，分别为RGB。这个*Volume*是一个张量*Tensor*。
- 卷积核的大小为$5 \times 5 \times 3$（举例），即3个通道，每个通道大小为$5 \times 5$。然后在整个图片上进行平移，计算卷积操作。（类似于卷积的定义）
- 卷积的计算操作为：卷积核在图片上平移，计算对应位置的乘积和。
![{7AC9125F-FA2D-42F9-B0CF-085ABFFB668F}]({7AC9125F-FA2D-42F9-B0CF-085ABFFB668F}.png)
上面的公式是在移动到某个位置$(w,h)$的时候，计算对应位置的乘积和。然后将结果存储在一个新的矩阵中。

*在实际操作的过程中，并不要将卷积核在图片上“滑动”（这是一个串行算法），可以为每一个为每一个局域来分配一个神经元，这样的操作是并行的，可以加速计算。*

得到的是一个Feature Map，这个Feature Map是一个新的张量，使用6个卷积核得到的图片的特征图的大小为$28 \times 28 \times 6$。*如何计算的？参数量是多少？*
使用的卷积核数量是一个超参数，可以调整。

对于一个$w*h*c$的输入，使用$k*k*c*d$个卷积核，得到的输出的特征图的大小为$(w-k+1)*(h-k+1)*d$，参数量为$(k*k*c+1)*d$。但是CNN在实践过程中的显存占用是较大的，因为需要存储特征图。

![{7202EECA-CD25-4EE1-A5C9-02F0263C968C}]({7202EECA-CD25-4EE1-A5C9-02F0263C968C}.png)
#### Dilated& Stride Convolution

在卷积操作中，我们可以使用不同的步长来进行卷积操作，这样的操作会改变输出的大小。在Dilated Convolution中，我们可以使用不同的扩张率来进行卷积操作。扩张率为1的时候*1-dilated*是正常的卷积操作，扩张率为2的时候，卷积核的间隔为2 *2-dilated*。这样的操作可以**增加卷积核的感受野**，但是减少了输出的大小。

*Stride*是卷积核的步长，可以让输出的特征图迅速变小。
可能出现不匹配的情况，这样可以进行*Padding*是在边缘填充0，这样的操作可以保持输出的大小。

##### Activation Functions

在卷积网络中，激活函数可以使用ReLU函数，这样的操作可以避免梯度消失的问题。有时候使用的是Leaky ReLU函数，这样的操作在某些模型中可以减少神经元死亡的问题。
$$
LeakyReLU(x) = \begin{cases} x & x>0 \\ 0.01x & x \leq 0 \end{cases}
$$


![{CFAAFBDA-A81C-4931-9C2C-CBD64F40E29C}]({CFAAFBDA-A81C-4931-9C2C-CBD64F40E29C}.png)

##### Pooling

池化操作一般是用来减少空间尺寸，这样的操作可以减少参数量，减少过拟合的问题。

- Max Pooling：取局部区域的最大值
- Average Pooling：取局部区域的平均值
![{01688551-C69C-4B62-8660-637A51409678}]({01688551-C69C-4B62-8660-637A51409678}.png)

**Spatial Pyramid Pooling**：对于不同的通道进行池化操作，这样的操作可以增加特征的多样性。这是用来减少多尺度问题的。 
在现实生活中，多尺度是一个基本的特征，例如在大气系统、气候系统中，有较大的气流和较小部分的湍流。

### Back Propagation
[Backpropagation, step-by-step | DL3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

![{1197B7EA-3F5B-40C9-8ADF-861B8AACBDC2}]({1197B7EA-3F5B-40C9-8ADF-861B8AACBDC2}.png)

在卷积网络的前向传播中每一层的计算公式如下：
$$
z_d^{(l+1)} = a^{(l)}\star \theta^{(l)}_d + b^{(l)}_d
$$

在一个具体的例子中，可以计算一个$3 \times 3 \times 1$的图像在经过一个$2\times 2 \times 1$的卷积核得到的$2 \times 2 \times 1$的特征图的计算过程。

- Consider single input channel $c=1$ :
$$
\left[\begin{array}{ll}
z_{11}^{(l+1)} & z_{12}^{(l+1)} \\
z_{21}^{(l+1)} & z_{22}^{(l+1)}
\end{array}\right]=\left[\begin{array}{lll}
a_{11}^{(l)} & a_{12}^{(l)} & a_{13}^{(l)} \\
a_{21}^{(l)} & a_{22}^{(l)} & a_{23}^{(l)} \\
a_{31}^{(l)} & a_{32}^{(l)} & a_{33}^{(l)}
\end{array}\right] \star\left[\begin{array}{ll}
\theta_{11}^{(l)} & \theta_{12}^{(l)} \\
\theta_{21}^{(l)} & \theta_{22}^{(l)}
\end{array}\right]
$$
- Expand above to be clearer:
$$
\left\{\begin{array}{l}
z_{11}^{(l+1)}=a_{11}^{(l)} \theta_{11}^{(l)}+a_{12}^{(l)} \theta_{12}^{(l)}+a_{21}^{(l)} \theta_{21}^{(l)}+a_{22}^{(l)} \theta_{22}^{(l)} \\
z_{12}^{(l+1)}=a_{12}^{(l)} \theta_{11}^{(l)}+a_{13}^{(l)} \theta_{12}^{(l)}+a_{22}^{(l)} \theta_{21}^{(l)}+a_{23}^{(l)} \theta_{22}^{(l)} \\
z_{21}^{(l+1)}=a_{21}^{(l)} \theta_{11}^{(l)}+a_{22}^{(l)} \theta_{12}^{(l)}+a_{31}^{(l)} \theta_{21}^{(l)}+a_{32}^{(l)} \theta_{22}^{(l)} \\
z_{22}^{(l+1)}=a_{22}^{(l)} \theta_{11}^{(l)}+a_{23}^{(l)} \theta_{12}^{(l)}+a_{32}^{(l)} \theta_{21}^{(l)}+a_{33}^{(l)} \theta_{22}^{(l)}
\end{array}\right.
$$

对于上述的例子计算残差网络的过程为：
$$
\begin{aligned}

\delta_{ij}^{(l)} & = \frac{\partial J(\theta,b)}{\partial z_{ij}^{(l)}}  =  \frac{\partial J(\theta,b)}{\partial a_{ij}^{(l)}} \frac{\partial a_{ij}^{(l)}}{\partial z^{(l)}_{ij}}\\
&= \sum_{z_{pq}^{(l+1)} \in  Pa(a_{ij}^{(l)})} \frac{\partial J(\theta,b)}{\partial z_{pq}^{(l+1)}} \frac{\partial z_{pq}^{(l+1)}}{\partial a_{ij}^{(l)}} \frac{\partial a_{ij}^{(l)}}{\partial z^{(l)}_{ij}} \\
&= \sum_{z_{pq}^{(l+1)} \in Pa(a_{ij}^{(l)}) } \delta_{pq}^{(l+1)} \frac{\partial z_{pq}^{(l+1)}}{\partial a_{ij}^{(l)}} g'(z_{ij}^{(l)})
\end{aligned}
$$

与全连接层的计算公式的不同在于，CNN是一个局部的计算过程，即每一个神经元的输出值并不会对后一层的所有神经元产生影响。

对于上述计算得到的残差，可以写成一个矩阵：
$$
\delta^{(l)} = \left[\begin{array}{lll}
\delta_{11}^{(l)} & \cdots & \delta_{1n}^{(l)} \\
\vdots & \ddots & \vdots \\
\delta_{m1}^{(l)} & \cdots & \delta_{mn}^{(l)}
\end{array}\right]
$$
在上面的例子中，我们可以计算得到：
$$
\left[\begin{array}{lll}
\delta_{11}^{(l)} & \delta_{12}^{(l)} & \delta_{13}^{(l)} \\
\delta_{21}^{(l)} & \delta_{22}^{(l)} & \delta_{23}^{(l)} \\
\delta_{31}^{(l)} & \delta_{32}^{(l)} & \delta_{33}^{(l)}
\end{array}\right]=\left[\begin{array}{cccc}
0 & 0 & 0 & 0 \\
0 & \delta_{11}^{(l+1)} & \delta_{12}^{(l+1)} & 0 \\
0 & \delta_{21}^{(l+1)} & \delta_{22}^{(l+1)} & 0 \\
0 & 0 & 0 & 0
\end{array}\right] \star\left[\begin{array}{cc}
\theta_{22}^{(l)} & \theta_{21}^{(l)} \\
\theta_{12}^{(l)} & \theta_{11}^{(l)}
\end{array}\right] \odot g^{\prime}\left(z^{(l)}\right)
$$
比较紧凑地写出：
$$
\delta^{(l)} = \delta^{(l+1)} \star rot180(\theta^{(l)}) \odot g^{\prime}(z^{(l)})
$$
其中$rot180$是对卷积核进行旋转180度的操作。
$\odot$对应的是两个矩阵的逐个元素乘

在有多张特征图的情况下（输出有$d$个通道），输入的图像有$c$通道，我们可以将上述的计算过程进行扩展，得到：
$$
\delta^{(l)} = \sum_{d} \delta^{(l+1)}_d \star rot180(\theta^{(l)}) \odot g^{\prime}(z^{(l)})
$$
在计算目标函数对于参数的导数的过程中，我们可以得到：
$$
\frac{\partial J(\theta,b)}{\partial \theta_d^{(l)}} = \frac{\partial J(\theta,b)}{\partial z_d^{(l+1)}} \frac{\partial z_d^{(l+1)}}{\partial \theta_d^{(l)}}
= \delta_d^{(l+1)} \star a^{(l)}_d
$$
可以更加详细地计算：
$$
\begin{aligned}
\frac{\partial J(\theta,b)}{\partial \theta_{i,j,k,d}^{(l)}} &= 
\sum_{m,n} \frac{\partial J(\theta,b)}{\partial z_{m,n}^{(l+1)}} \frac{\partial z_{m,n}^{(l+1)}}{\partial \theta_{i,j,k,d}^{(l)}} \\
&= \sum_{m,n} \delta_{m,n}^{(l+1)} a_{m+i-1,n+j-1,k}^{(l)} \\
&= \delta_d^{(l+1)} \star a^{(l)}_d 
\end{aligned}
$$
对于偏置项的计算过程为
$$
\frac{\partial J(\theta,b) }{\partial b_d^{(l)}} = \sum_{m,n} \frac{\partial J(\theta,b)}{\partial z_{m,n}^{(l+1)}} \frac{\partial z_{m,n}^{(l+1)}}{\partial b_d^{(l)}} = \sum_{m,n} \delta_{m,n}^{(l+1)} 
$$

从而可以使用**动量的梯度**下降的方法来进行参数的更新。

### Invariance and Equivariance

对于不同的任务，有时候需要不变性和等变性。在图像处理中，对于图像的旋转、平移、缩放等操作，神经网络的输出应该是不变的。在神经网络中，可以通过数据增强的方法来进行处理。对于等变性，可以通过卷积神经网络来进行处理。CNN在设计的过程中，引入池化层*pooling*希望获得不变性。但是并没有获得很好的效果。
- **Invariance**：对于输入的变化，输出不变
$$
f(T(X)) = f(x)
$$
- **Equivariance**：对于输入的变化，输出也会发生相应的变化
$$
f(T(X)) = T(f(x))
$$
在实践中，有时候会有噪声、形变、翻转、光照条件、视角变化、遮挡、尺度变化、类内类间差距、奇异等问题。

#### Data Augmentation

在实践中，可以通过数据增强的方法来进行处理。对于图像的旋转、平移、缩放、翻转等操作，可以增加数据的多样性。在训练的过程中，可以使用不同的数据增强的方法来进行训练。用这样的方法期望获得一种不变性。

常用的方法有：裁剪、旋转、翻转、缩放、平移、仿射变换、弹性变换、颜色变换等。
CNN识别主要使的是纹理的识别方法，希望将人类的对于形状识别的能力融入到CNN中。可以使用的数据集为*Augmentation by Stylization*，希望获得模型对于纹理的不变性。
CNN对上下文是敏感的，对于经常出现在一起的事物，CNN可以很好地进行识别，但是对于不常见的事物，CNN的效果会变差。一种极端的方式是将不经常出现的东西组合在一起。目前的深度网络一定程度上利用了*spurious correlation*，从而进行bench mark在数据集上过拟和。

#### Architecture Revolution

##### MAGA Making convolutional networks shift-invariant again

在CNN中，在平移过程中，网络很难获得平移不变性。在采用模糊之后的*pooling*可以获得一定程度上的平移不变性。

##### Capsule Network
CNN的缺点还有：不能保持物体之间的相对关系。在分类任务中，一般情况下强调的是不变性，而在一些细粒度识别任务中，强调的是等变性。在Capsule Network中，引入了胶囊的概念，这样的操作可以保持物体之间的相对关系。


## CNN Architectures

### AlexNet

![{382F2347-B42F-46CB-83DB-F25DDE49A91E}]({382F2347-B42F-46CB-83DB-F25DDE49A91E}.png)

- 一般会在输入层使用较大的卷积核，然后在后面的层使用较小的卷积核
- 通道数随着网络逐渐增加然后减少
- 第一次使用ReLU激活函数
- 使用了大量的数据增强*Data Augmentation*
- 使用GPU进行训练
- 采用SGD with momentum 0.9进行训练
- 使用dropout 0.5，一般在MLP层都需要使用dropout
- 使用0.01的学习率，然后在训练的过程中逐渐减小学习率，当loss不再下降的时使用学习率的0.1倍
#### ZFNet

在输入层使用了更小的卷积核，一般不要使用小的卷积核（更大的stride）会导致信息的丢失。
在中间层数使用了更多的通道数。

### VGGNet

用更小的卷积核来代替较大的卷积核，这样的操作可以减少参数量，增加网络的深度。在VGGNet中，使用了3个$3 \times 3$的卷积核来代替一个$7 \times 7$的卷积核。这样的操作可以增加网络的深度，减少参数量。**一个较大的感受野可以通过多个较小的感受野来代替**。但是现在又发现事实上没有这么好的效果，所以现在又开始使用较大的卷积核。

采用了预训练的方法。在训练的过程中，首先在较小的数据集上进行过拟和（在这个训练集上的损失函数接近0），然后在较大的数据集上进行训练。

在早期的网络中，池化层的使用是较多的，现在已经很少使用。
### NIN Network in Network

在使用全连接层时，会有较多的参数，基于这样的思路提出$1\times 1$卷积这个操作。这样的操作只改变通道数，不改变空间尺寸，通常在不希望改变空间尺寸而增大通道数的时候可以使用。

在使用卷积增加步长、使用*pooling*层时候，会导致信息的丢失。在NIN中希望获得一个尺寸小但是通道数多的特征图。在NIN中使用了$1\times 1$的卷积核来增加通道数，这样可以使用空间全局池化。例如一个$256\times 6 \times 6$的特征图，使用$6\times 6$的全局池化，可以得到一个$256\times 1 \times 1$的特征图。

> [!NOTE]
> - 传统CNN末尾通常使用全连接层（FC）进行分类，但全连接层参数量大，易过拟合。
> - 全局池化可直接将特征图转换为通道维度的向量（如$1\times 1 \times1024$ $\rightarrow$  1024维向量），再接一个分类层，大幅减少参数。

#### GoogLeNet

![{43BF44F4-D211-4165-AE84-257E702582A0}]({43BF44F4-D211-4165-AE84-257E702582A0}.png)
这是一个较为深的网络，删去了全连接层，参数量是较小的。
- 引入Multi-passway，使用多路的卷积核来提取特征。采用了特征增广*Feature Augmentation*的方法。
- 在特征图通道数目不一样的情况下，使用padding的方法。

![{6F8AFF72-4C3D-4FAD-8FC9-90F7FED6E8B2}]({6F8AFF72-4C3D-4FAD-8FC9-90F7FED6E8B2}.png)

![{39D210C3-7C1B-4913-AC78-12E71A8FD13B}]({39D210C3-7C1B-4913-AC78-12E71A8FD13B}.png)

![{E3B121C1-5782-4762-A983-042799DA4ED9}]({E3B121C1-5782-4762-A983-042799DA4ED9}.png)

使用计算量越来越小的卷积核，使用计算量越来越小的卷积核，使用越来越多的层数，使用越来越多的通道数。

#### Highway Network

在平坦的网络通路中，可能有信息通路瓶颈问题。在Highway Network中，引入了门控机制，这样的操作可以使得信息的流动更加顺畅。
$$
y= H(x,W_H)T(x,W_T) + x(1-T(x,W_T))
$$
其中$H(x,W_H)$是一个MLP，$T(x,W_T)$是一个门控函数，这样的操作可以使得信息的流动更加顺畅。

- $x$: 输入向量。
- $H(x, W_H)$: 非线性变换（如全连接层或卷积层）。
- $T(x, W_T)$: Transform Gate（变换门），控制非线性变换的权重，通常用Sigmoid激活（输出值在0到1之间）。
- $C(x, W_C)$: Carry Gate（携带门），控制原始输入 x 的权重。通常设定为 $C = 1 - T$，以减少参数量。

- 当 $T(x)→0$ 时，输出$y≈x$，即当前层几乎不进行变换（信息直接跳过该层）。
- 当 $T(x)→1$ 时，输出$y≈H(x)$，即信息完全经过当前层的非线性变换。
- 通过这种方式，网络可以自适应地选择浅层或深层的特征。

$$
T(x,W_T) = \sigma(W_T^T x + b_T)
$$
动机是在网络中**提高信息的流动性**

- **ResNet**：使用恒等跳跃连接*Identity Skip Connection*，公式为 $y=H(x)+x$，无门控机制。
- **Highway Network**：通过门控动态调节跳跃连接的权重，更灵活地控制信息流。
#### ResNet

![{6767A6D4-59B6-4B62-9FE9-427D009BB837}]({6767A6D4-59B6-4B62-9FE9-427D009BB837}.png)

- 56层的模型比20层的模型效果更差，既然如此，先将20层的模型训练好，然后再增加36层的*identity*网络，之后再训练整个网络。发现这样的操作的效果反而更差。得出的结论是56层的网络更加难以训练，网络的拟和能力不足。**平坦的网络很难拟和**
- 残差是更加容易拟和的。
- 继续将这些残差网络堆叠在一起，可以得到一个更深的网络。
- 卷积、池化是算子*operator*，残差是一个块*block*，网络是一个层*layer*

![{8F67BE27-E8CA-4921-99B2-C57FFFAE5E7B}]({8F67BE27-E8CA-4921-99B2-C57FFFAE5E7B}.png)

![{681C5361-F2ED-427F-86C2-48C9EC68AADF}]({681C5361-F2ED-427F-86C2-48C9EC68AADF}.png)

![{5DE8620F-D5FB-4E82-96D7-0E03CCCCAF82}]({5DE8620F-D5FB-4E82-96D7-0E03CCCCAF82}.png)
对于网络会有多个维度进行评价：
- 准确率top-1、top-5准确率
- VGG网络参数量很大，但是由于是平坦的网络，所以有一定的计算整齐度。在网络有较多的分叉时，计算整齐度会变差。较为整齐的网络计算是较快的。
- 目前的显卡对于$3 \times 3$的卷积核是有硬件加速的

### LandScape Visualization

[[1712.09913] Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)

## Lightweight for Deployment

### Pruning

卷积神经网络是一个很适合做剪枝的网络。说明神经网络的有很多的参数是冗余的。剪枝配合上重训练可以减少网络的参数量，并且在一定的程度上提升网络的性能。

#### Quantization and Encoding

k-means算法可以将权重量化，将权重量化为几个值，这样的操作可以减少网络的参数量。在实际的操作中，可以将权重量化为8位，这样的操作可以减少网络的参数量。

编码操作，如Huffman编码，可以减少网络的参数量。
[[1510.00149] Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)

神经网络一般是在训练的时候使用较大的网络，然后再裁剪为较小的网络，反之效果不一定会好。

我们相信在裁剪的过程中，保留下来的参数是重要的参数，这样的操作可以提升网络的性能。在实验中

[[1803.03635] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)

先设计一个较小的网络，之后推广到较大的网络。在实际的操作中，可以使用较小的网络，然后再增加网络的深度。这种设计是硬件友好的，由于这样的设计是计算对齐的。

#### Group Convolution

标准的卷积层有这样的不足：对于不同的输入通道，后面的输出都使用到了所有的输入通道。在Group Convolution中，将输入通道分为几个组，然后对于每一个组使用一个卷积核。这样的操作可以减少网络的参数量，减少计算量。

但是通道之间不交流，这样的操作可能会导致网络的性能下降。

#### Depthwise Separable Convolution

- 引入了$1 \times 1$卷积核*pointwise*，这样的操作可以减少参数量
- 令通道数等于分组数目*channelwise* or *depthwise*

上述两个算子是轻量化网络的基础。

![Pasted image 20250316132626](Pasted image 20250316132626.png)

[[2201.03545] A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

![Pasted image 20250316134751](Pasted image 20250316134751.png)

宏观设计：**减少空间尺寸的衰减有利于学习不同的特征**。

## Advanced Modules

### 3D Modeling

普通的二维卷积网络在处理视频时，会丢失时间信息。在3D卷积网络中，引入了时间维度，这样的操作可以保留时间信息。

直接使用3D卷积可以进行直接的对应。

#### Deforamble Convolution

在CNN中，形变较难进行建模，对于形变的物体效果会变差。在识别的过程中，在识别的过程中可能引入噪音。在Spatial Transformer Network中，引入了形变的概念，类似于视觉中进行防抖的操作。

在CNN中引入偏置的概念，在对应的特征图上进行坐标的偏置操作
$$
y(p_0) = \sum_{p_i \in \Omega} w_i x(p_i+p_0+\Delta p_0)
$$
#### Attention

要建模大范围的特征范围，就是high-level的特征，但是在使用CNN的过程中，有效感受野并没有那么大。

$$
y_i = \frac{1}{\mathcal{C}(x)}\sum_{\forall j} f(x_i,x_j)g(x_j)
$$
对于注意力函数$f(x_i,x_j)$可以计算为两个参数的内积。上述操作实际上是一种全局建模，可以在CNN的最后几步使用。


#### CAM

作为一种可解释性的工作，CAM可以将CNN的输出映射到输入图像上，这样的操作可以直观地看到CNN的输出。计算不同部分的权重，可以得到不同部分的重要性。
