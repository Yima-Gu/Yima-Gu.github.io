---
title: Deep Learning Lecture-4
date: 2025-06-21
tags:
  - DeepLearning
math: true
---

## Optimization

**Objective Function**
$$
\arg \min \mathcal{O(D,\theta)}=  \sum_{i=1}^{N} L(y_i, f(x_i,\theta)) + \Omega(\theta)
$$
上述目标可以可视化为以$\theta$为横坐标、$\mathcal{O}$为纵坐标的函数图像，我们的目标是找到函数图像的最低点。这是一个**非凸优化**问题，是初值敏感的。“地形图”是否简单是网络训练是否容易的关键。

### First-Order Optimization

可以将$J(\theta)$展开为泰勒级数：
$$
J(\theta) = J(\theta_0) + \nabla J(\theta_0)^T(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0) 
$$
沿着梯度的方向进行更新：
$$
J(\theta - \eta g) = J(\theta ) -  \eta g^Tg \leq J(\theta)
$$

**梯度下降算法的问题在于**：
- 容易在鞍点处停滞
- 对于较为简单的凸优化问题，学习率的选择不好都会有发散的问题，训练对于学习率是很敏感的

对于学习率下降的算法，主流使用的是*Step Strategy*，即损失函数不下降了就减少学习率。

#### Warm Restarts

使用的策略为：*Cosine Annealing*:
$$
\eta_t = \eta_{min}^i + \frac{1}{2}(\eta_{max}^i - \eta_{min}^i)(1 + \cos(\frac{T_{cur}}{T_{i}}\pi))
$$

其中$T_{cur}$为当前的迭代次数，$T_{i}$为当前的周期数，$\eta_{min}^i$和$\eta_{max}^i$分别为第$i$个周期的最小和最大学习率。
学习率的衰减不能是线性的，是先快后慢的。

#### Convergence Rate

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

不仅要关注一次的梯度信息，还要关注二次的信息。二阶优化算法的核心是Hessian矩阵，可以分辨是不是鞍点。
函数的展开为：
$$
J(\theta) = J(\theta_0) + \nabla J(\theta_0)^T(\theta - \theta_0) + \frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0) 
$$
能感知到“地形图”中的曲率。
对于海森矩阵可以进行特征值分解：
$$
H = Q \Lambda Q^T \quad \text{and} \quad H^{-1} = Q \Lambda^{-1} Q^T
$$
特征值中较大和较小的特征值如果相差较大，称为病态矩阵；如果从最大到最小的变化较为平缓，则较为光滑。
如果特征值全为正值，那么就是凸函数；如果有正有负，那么就是鞍点。
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
牛顿法的优点在于收敛速度快，但是缺点在于计算复杂度高，需要计算海森矩阵的逆矩阵。计算复杂度为$O(d^3)$，其中$d$为参数的个数。**在深度学习时代基本上不再使用**。

#### Quasi-Newton Method
对于海森矩阵的逆矩阵，我们可以使用拟牛顿法进行近似：
$$
H_{t+1}^{-1} = H_t^{-1} + \frac{y_t y_t^T}{y_t^T s_t} - \frac{H_t^{-1}s_t s_t^T H_t^{-1}}{s_t^T H_t^{-1}s_t}
$$
其中$y_t = \nabla J(\theta_{t+1}) - \nabla J(\theta_t)$，$s_t = \theta_{t+1} - \theta_t$。
**在矩阵计算的时候要将较小的矩阵先乘，这样可以计算复杂度**

## Optimization in Deep Learning

[[Deep Learning Lecture-2#Optimization in Practice]]

是非凸优化问题，优化的目的在于找到一个较好的局部极值。比较好的局部极值是比较低的、比较平缓的局部极值，对于比较陡峭的局部极值泛化能力比较差（对测试数据的微小变化敏感）。

好的局部极值有一些特性：
- 值比较低
- 是”盆地“，这样有利于模型的泛化

### mini-batch

**mini-batch SGD**：在每一轮遍历*epoch*后，对数据进行随机的打乱*Shuffle*，然后分成若干个batch，对每一个batch进行参数的更新。这样可以减少计算的时间，同时可以减少过拟合的问题。

- mini-batch的大小对于训练的影响，一般而言较大的mini-batch会有更好的收敛性，但是计算复杂度更高。
- 由于不一样的小样本选择会引入一定的随机性，这样是有利于跳出局部极值的。
- 由于mini-batch的选择是有随机性的，不同的batch的难度不一样，所以这时候出现Loss的规律性的震荡是很正常的。
- 矩阵最大奇异值与最小奇异值的比值称为矩阵的条件数，条件数越大，矩阵越病态。对于病态矩阵，SGD的收敛速度会变慢。

### Learning Rate Decay
初始学习率较大，随着迭代次数的增加，学习率逐渐减小。有相对应的衰减策略。
*Exponential decay*:
$$
\eta_t = \eta_0 \cdot e^{-\alpha t}
$$
*Inverse decay*:
$$
\eta_t = \frac{\eta_0}{1+\alpha t}
$$
### SGD Stochastic Gradient Descent
#### SGD with Momentum

**SGD with Momentum**:

对于下面的更新公式：

$$
\theta_{ij} = \theta_{ij} - \eta \Delta
$$
在高维中，地形是相对而言较为崎岖的，这里的学习率一般是比较小的，否则容易发散。在接近于局部极值的时候。较大的学习率学习的是较为粗糙的特征，较小的学习率学习的是较为细致的特征。

$$
\Delta = \beta \Delta - \eta \frac{\partial J(\theta)}{\partial \theta_{ij}}
$$
$\beta$是动量参数，可以理解为之前的梯度的累积。

**Nesterov Momentum:**
$$
\begin{aligned}
&\tilde{\theta}^{t} = \theta^{t} - \beta \Delta^{t-1} \\
&\Delta^{t} = \beta \Delta^{t-1} + (1-\beta)\nabla J^t (\tilde{\theta}^t)\\
&\theta^{t+1} = \theta^t - \eta \Delta^t\\
\end{aligned}
$$
在深度学习的实现中使用的一般是这种。走动量的方向可以减少震荡，同时可以加速收敛。当到达了比较好的局部极值时候又会在这个值的附近抖动。
超参数：$\beta$，一般而言$\beta$取0.9是比较好的，越大的值越容易进行震荡。
学习率0.01、0.003、0.001一般按照指数变化。

是在每次更新完$\theta$之后（进行试探之后才进行计算）才进行梯度的计算，可以避免一些*overshoot*。核心的思想为多获取一些二次的信息。

### *Weight Decay*

加入正则项，对于参数的更新进行限制，控制假设空间的大小，可以防止过拟合。但是在深度学习中并不够。

*L1 regularization*
$$
\Omega(\theta) = \lambda \sum_{l=1}^{L} \sum_{i=1}^{n_l} \sum_{j=1}^{n_{l+1}} |\theta_{ij}^{(l)}|
$$
*L2 regularization*
$$
\Omega(\theta) = \lambda \sum_{l=1}^{L} \sum_{i=1}^{n_l} \sum_{j=1}^{n_{l+1}} (\theta_{ij}^{(l)})^2
$$

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

调参一般而言是，对于一个模型找到对于其最好的优化器。
