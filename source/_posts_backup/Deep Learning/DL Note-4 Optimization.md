---
title: DL Note-4 Optimization
date: 2025-07-21 14:30
tags:
  - DeepLearning
  - Optimization
categories:
  - Deep Learning
math: true
---
## Optimization

**Objective Function**

在深度学习中，训练模型本质上是一个优化过程。我们的核心目标是寻找一组模型参数$\theta$，使得在给定数据集$\mathcal{D}$上的目标函数$\mathcal{O}(\mathcal{D}, \theta)$最小化。该目标函数通常由两部分构成：
$$
\arg \min \mathcal{O(D,\theta)}=  \sum_{i=1}^{N} L(y_i, f(x_i,\theta)) + \Omega(\theta)
$$
- **损失项 (Loss Term)**: $\sum_{i=1}^{N} L(y_i, f(x_i,\theta))$，衡量模型在训练数据上的表现。$L$是损失函数，用于计算模型预测值$f(x_i, \theta)$与真实标签$y_i$之间的差异。
- **正则项 (Regularization Term)**:  $\Omega(\theta)$ 用于对模型复杂度进行惩罚，防止过拟合，从而提升模型的泛化能力。

将目标函数 $\mathcal{O}$ 视为以参数 $\theta$ 为变量的超高维曲面，我们的任务就是找到这个曲面的最低点。然而，深度学习中的目标函数是**非凸 (Non-convex)** 的，这意味着它存在大量的局部极小值 (local minima) 和鞍点 (saddle points)。因此，优化算法的性能不仅取决于能否快速收敛，还取决于能否找到一个“好”的极小值——通常指那些不仅损失值低，而且所处区域较为平坦宽阔 (flat minima) 的极小值，这样的模型泛化性能更强。
### First-Order Optimization

可以将$J(\theta)$展开为泰勒级数：
$$
J(\theta) = J(\theta_0) + \nabla J(\theta_0)^T(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0) 
$$
沿着梯度的方向进行更新：
$$
J(\theta - \eta g) = J(\theta ) -  \eta g^Tg \leq J(\theta)
$$

**梯度下降的主要挑战**: 

1. **鞍点停滞**: 在高维空间中，鞍点远比局部极小值更常见。在鞍点处梯度为零，标准梯度下降会在此停滞。 
2. **学习率敏感性**: $\eta$ 的选择至关重要。过大可能导致在极小值附近震荡甚至发散；过小则收敛速度过慢。 对于学习率的调整，常见的策略是**学习率衰减 (Learning Rate Decay)**，例如当损失不再下降时，按比例减小学习率 (Step Strategy)。

#### Learning Rate Decay

初始学习率较大，随着迭代次数的增加，学习率逐渐减小。有相对应的衰减策略。
*Exponential decay*:
$$
\eta_t = \eta_0 \cdot e^{-\alpha t}
$$
*Inverse decay*:
$$
\eta_t = \frac{\eta_0}{1+\alpha t}
$$

#### Warm Restarts

使用的策略为：*Cosine Annealing*  与其让学习率一路单调下降，不如让它**周期性地“回暖”一下**。
$$
\eta_t = \eta_{min}^i + \frac{1}{2}(\eta_{max}^i - \eta_{min}^i)(1 + \cos(\frac{T_{cur}}{T_{i}}\pi))
$$

其中$T_{cur}$为当前的迭代次数，$T_{i}$为当前的周期数，$\eta_{min}^i$和$\eta_{max}^i$分别为第$i$个周期的最小和最大学习率。在每个周期结束后，学习率被重置 (Warm Restart) 为一个较大的值。这种“重启”有助于优化过程跳出当前的局部极小值，去探索更优的解空间。

#### Convergence Rate

假设函数 $J(\theta)$ 是凸函数、可微，且梯度满足 $L$-Lipschitz 连续条件。对于梯度下降法，可以证明其收敛速率为： $$\frac{1}{T} \sum_{t=0}^{T-1} J(\theta^t) - J(\theta^*) \leq \mathcal{O}(\frac{1}{\sqrt{T}})$$ 这表明，为了使平均损失与最优损失的差距减小10倍，需要的迭代次数 $T$ 大约要增加100倍。这是一种次线性 (sub-linear) 的收敛速率，相对较慢。

- We assume that $J(\theta)$ is convex, differentiable and Lipchitz by constant $L$. And domain of $\theta$ is bounded by radius $R$. With gradient descent update:
$$
\theta^{t+1}=\theta^t-\eta \nabla J\left(\theta^t\right)
$$

$$
\begin{aligned}
&J\left(\theta^t\right)-J\left(\theta^*\right)  & \text { (Gap btw minimum value } \left.J\left(\theta^*\right)\right) \\

\leq & \nabla J\left(\theta^t\right)\left(\theta^t-\theta^*\right) & \text { (Convexity) } \\

= & \frac{1}{\eta}\left(\theta^t-\theta^{t+1}\right)^{\mathrm{T}}\left(\theta^t-\theta^*\right)  & \text { (Definition of GD) } \\

=& \frac{1}{2 \eta}\left(\left\|\theta^t-\theta^*\right\|^2+\left\|\theta^{t+1}-\theta^t\right\|^2-\left\|\theta^{t+1}-\theta^*\right\|^2\right) & \text { (Properties of Norm) } \\
= & \frac{1}{2 \eta}\left(\left\|\theta^t-\theta^*\right\|^2-\left\|\theta^{t+1}-\theta^*\right\|^2\right)+\frac{\eta}{2}\left\|\nabla J\left(\theta^t\right)\right\|^2 & \text { (Definition of GD) } \\

\leq &\frac{1}{2 \eta}\left(\left\|\theta^t-\theta^*\right\|^2-\left\|\theta^{t+1}-\theta^*\right\|^2\right)+\frac{\eta}{2} L^2 &  \text { (Lipchitz Property) }
\end{aligned}
$$


- From previous computation, we get the following inequality for every step $t$ :

$$
J\left(\theta^t\right)-J\left(\theta^*\right) \leq \frac{1}{2 \eta}\left(\left\|\theta^t-\theta^*\right\|^2-\left\|\theta^{t+1}-\theta^*\right\|^2\right)+\frac{\eta}{2} L^2
$$


- Recall $\max _{\theta, \theta^{\prime}}\left(\left\|\theta-\theta^{\prime}\right\|\right) \leq R$. Assume we update parameters for $T$ steps. We add all equations for all $t \in\{0,1, \ldots, T-1\}$ :

$$
\begin{aligned}
& \sum_t\left(J\left(\theta^t\right)-J\left(\theta^*\right)\right) \leq \frac{1}{2 \eta}\left(\left\|\theta^0-\theta_*^*\right\|^2-\left\|\theta^T-\theta^*\right\|^2\right)+\frac{\eta L^2 T}{2} \\
& \frac{1}{T} \sum_t J\left(\theta^t\right)-J\left(\theta^*\right) \leq \frac{1}{2 \eta T}\left(R^2+0\right)+\frac{\eta L^2}{2} \\
& \frac{1}{T} \sum_t J\left(\theta^t\right)-J\left(\theta^*\right) \leq \frac{R^2}{2 \eta T}+\frac{\eta L^2}{2}
\end{aligned}
$$

- let $\eta = \frac{R}{L\sqrt{T}}$:
$$
\frac{1}{T} \sum_t J\left(\theta^t\right)-J\left(\theta^*\right) \leq \frac{L}{R} \sqrt{T}
$$

### Second-Order Optimization

二阶优化算法同时利用一阶导数（梯度）和二阶导数（海森矩阵）的信息，能够感知到损失曲面的曲率，从而进行更高效的更新。

函数$J(\theta)$的二阶泰勒展开为：
$$
J(\theta) = J(\theta_0) + \nabla J(\theta_0)^T(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0) 
$$
能感知到“地形图”中的曲率。

对于海森矩阵可以进行特征值分解：
$$
H = Q \Lambda Q^T \quad \text{and} \quad H^{-1} = Q \Lambda^{-1} Q^T
$$
特征值中较大和较小的特征值如果相差较大，称为病态矩阵；如果从最大到最小的变化较为平缓，则较为光滑。

其中 H 是海森矩阵，其包含了损失曲面的曲率信息。通过海森矩阵的特征值，我们可以判断一个临界点（梯度为0的点）的性质：
- **正定矩阵 (所有特征值为正)**: 局部极小值。
- **负定矩阵 (所有特征值为负)**: 局部极大值。
- **不定矩阵 (特征值有正有负)**: 鞍点。

事实上，使用的梯度方法为局部的方法，下降是相对较慢的。

#### Newton's Method

牛顿法的计算方法为：
$$
\hat{J}(\theta) = J(\theta_0) + \nabla J(\theta_0)^T(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0)
$$
求导得到：
$$
\nabla_{\theta} \hat{J}(\theta) = \nabla_{\theta} J(\theta_0) + H(\theta - \theta_0) = 0
$$
求解得到：
$$
\theta^{t+1} = \theta^{t} - H^{-1} \nabla_{theta} J(\theta^{t})
$$
牛顿法的优点在于收敛速度快，但是缺点在于计算复杂度高，需要计算海森矩阵的逆矩阵。计算复杂度为$O(d^3)$，其中$d$为参数的个数。这在深度学习模型中（$d$ 通常是百万甚至亿级别）是完全不可接受的。**在深度学习时代基本上不再使用**。

#### Quasi-Newton Method

对于海森矩阵的逆矩阵，我们可以使用拟牛顿法进行近似：

$$
H_{t+1}^{-1} = H_t^{-1} + \frac{y_t y_t^T}{y_t^T s_t} - \frac{H_t^{-1}s_t s_t^T H_t^{-1}}{s_t^T H_t^{-1}s_t}
$$

其中 $s_t = \theta_{t+1} - \theta_t$，$y_t = \nabla J(\theta_{t+1}) - \nabla J(\theta_t)$。尽管比牛顿法高效，但存储近似矩阵 $B_t$ 仍需要 $O(d^2)$ 的空间，对于大规模模型依然有挑战 (L-BFGS通过只存储最近的若干个 $s_t, y_t$ 向量对解决了此问题)。

## Optimization in Deep Learning

[[DL Note-2 MLP#Optimization in Practice]]

在深度学习这个复杂的“非凸山区”里，我们的目标不是找到那个虚无缥缈的“全局最低点”，而是找到一个**足够好**的局部最低点。一个好的“山谷”通常具备两个特点：

1. **足够低**：即模型的损失值足够小。
2. **足够宽阔平坦**：像一个“盆地”而不是一个“深坑”。这样的模型泛化能力更强，因为它对测试数据的微小变化不那么敏感。

下面是一些在实践中真正被广泛使用的优化策略。

### mini-batch

**mini-batch SGD**：在每一轮遍历*epoch*后，对数据进行随机的打乱*Shuffle*，然后分成若干个batch，对每一个batch进行参数的更新。这样可以减少计算的时间，同时可以减少过拟合的问题。

- mini-batch的大小对于训练的影响，一般而言较大的mini-batch会有更好的收敛性，但是计算复杂度更高。
- 由于不一样的小样本选择会引入一定的随机性，这样是有利于跳出局部极值的。
- 由于mini-batch的选择是有随机性的，不同的batch的难度不一样，所以这时候出现Loss的规律性的震荡是很正常的。
- 矩阵最大奇异值与最小奇异值的比值称为矩阵的条件数，条件数越大，矩阵越病态。对于病态矩阵，SGD的收敛速度会变慢。

### SGD (Stochastic Gradient Descent) with Momentum

标准SGD在穿越狭长、陡峭的沟壑时，梯度方向会来回震荡，导致收敛缓慢。Momentum 算法引入了“惯性”的概念，模拟物理世界中物体的运动。更新时不仅考虑当前梯度，还累积了历史梯度的方向。

$$ \begin{aligned} v^t &= \beta v^{t-1} + (1-\beta) \nabla J(\theta^t) \\ \theta^{t+1} &= \theta^t - \eta v^t \end{aligned} $$
- $v^t$ 是速度向量，是梯度指数移动平均。 
-  $\beta$是动量系数 (通常取0.9左右)，控制“惯性”的大小。

**Nesterov Momentum:**
$$
\begin{aligned}
&\tilde{\theta}^{t} = \theta^{t} - \beta \Delta^{t-1} \\
&\Delta^{t} = \beta \Delta^{t-1} + (1-\beta)\nabla J^t (\tilde{\theta}^t)\\
&\theta^{t+1} = \theta^t - \eta \Delta^t\\
\end{aligned}
$$
在深度学习的实现中使用的一般是这种。走动量的方向可以减少震荡，同时可以加速收敛。当到达了比较好的局部极值时候又会在这个值的附近抖动。

- 超参数：$\beta$，一般而言$\beta$取0.9是比较好的，越大的值越容易进行震荡。
- 学习率0.01、0.003、0.001一般按照指数变化。

是在每次更新完$\theta$之后（进行试探之后才进行计算）才进行梯度的计算，可**以避免一些*overshoot***。核心的思想为多获取一些二次的信息。


### Adaptive Learning Rate

直观理解为在不同的“地形”上需要使用的学习率（步长）是不一样的。对于不同的参数使用不同的学习率。**Adagrad**算法的核心思想为：
$$
\begin{aligned}
&r^t = r^{t-1} + \nabla J^t(\theta^t) \odot \nabla J^t(\theta^t)\\
&h^t = \frac{1}{\sqrt{r^t} + \delta} \\
&\Delta^t = h^t \odot \nabla J^t(\theta^t)\\
&\theta^{t+1} = \theta^t - \eta \Delta^t
\end{aligned}
$$
*上述公式中的第二行为逐元素操作*
其中$\odot$为对应元素相乘，$\delta$为一个很小的数，防止分母为0。这样可以保证在不同的地形上使用不同的学习率。**本质上为探索"地形图"**。但是Adagrad的问题在于随着迭代次数的增加，分母会变得越来越大，导致学习率会变得越来越小，最终会导致学习率为0，这样就不再更新了。

**RMSprop**算法的核心思想为：对Adagrad的分母进行指数滑动平均：
$$
\begin{aligned}
&r^t = \rho r^{t-1} + (1-\rho)\nabla J^t(\theta^t) \odot \nabla J^t(\theta^t)\\
&h^t = \frac{1}{\sqrt{r^t} + \delta} \\
&\Delta^t = h^t \odot \nabla J^t(\theta^t)\\
&\theta^{t+1} = \theta^t - \eta \Delta^t
\end{aligned}
$$

**Adam**算法的核心思想为：结合了SGD with Momentum和RMSprop：
$$
\begin{aligned}
& r^t = \rho r^{t-1} + (1-\rho)\nabla J^t(\theta^t) \odot \nabla J^t(\theta^t)\\
& h^t = \frac{1}{\sqrt{r^t} + \delta} \\
& s^t = \varepsilon s^{t-1} + (1-\epsilon)\nabla J^t(\theta^t)\\
& \Delta^t = h^t \odot s^t \\
& \theta^{t+1} = \theta^t - \eta \Delta^t
\end{aligned}
$$
实际使用的参数为$\rho=0.9$，$\varepsilon = 0.9$，$\rho = 0.999$
对于学习率的下降，还是要使用对应的算法，对于实际使用的算法，还需要对$r$、$s$进行无偏修正。

Nadam算法为Adam算法的变种，对于SGD with Momentum的更新进行了修正。

在选择优化器时，Adam 通常是一个稳健的默认选项。然而，精心调参的 SGD with Momentum 在某些任务上（如计算机视觉）仍然可能达到更好或更快的收敛效果。理解每种优化器的内在机制和优缺点，是进行高效模型训练的关键。

### *Weight Decay*

$$
\arg \min \mathcal{O(D,\theta)}=  \sum_{i=1}^{N} L(y_i, f(x_i,\theta)) + \Omega(\theta)
$$

加入正则项，对于参数的更新进行限制，控制假设空间的大小，可以防止过拟合。但是在深度学习中并不够。

这里的 $\lambda$ 就是**权重衰减系数**，一个由我们自己设定的超参数，用来控制“惩罚”的强度。

**请注意这个公式的核心变化**：

在计算“损失的梯度”并更新权重**之前**，`旧权重`会先乘以一个**小于1的系数 `(1 - 学习率 * λ)`**。
- 这个步骤独立于数据本身，它只跟权重当前的大小有关。
- 每一次参数更新，权重都会因为这个乘法而“自动”缩小一点点。
- 这就是“**衰减 (Decay)**”这个词的由来。它像一种阻力，持续不断地将所有权重往零的方向拉。
一个权重只有在“损失的梯度”提供的更新量足够大，能够抵消掉这种衰减效应时，它才能维持在一个较大的值。这意味着，**模型只允许那些对于减小损失确实有巨大贡献的特征，拥有较大的权重。**

*L1 regularization*
$$
\Omega(\theta) = \lambda \sum_{l=1}^{L} \sum_{i=1}^{n_l} \sum_{j=1}^{n_{l+1}} |\theta_{ij}^{(l)}|
$$
*L2 regularization*
$$
\Omega(\theta) = \lambda \sum_{l=1}^{L} \sum_{i=1}^{n_l} \sum_{j=1}^{n_{l+1}} (\theta_{ij}^{(l)})^2
$$