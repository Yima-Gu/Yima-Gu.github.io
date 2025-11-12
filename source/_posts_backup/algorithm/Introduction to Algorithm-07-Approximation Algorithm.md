---
title: Introduction to Algorithm-07-Approximation Algorithm
date: 2025-06-16 19:45
tags:
    - Approximation Algorithm
    - Algorithm
categories:
  - Algorithm
math: true
syntax_converted: true
---


## Definition

- 如果对规模为$n$的任意输入，近似算法所产生的近似解的代价$C$与最优解的代价$C^*$只差一个因子$\rho(n)$使得
$$
\max\left(\frac{C}{C^*}, \frac{C^*}{C}\right) \le\rho(n)
$$
  称这个算法为$\rho(n)$-近似算法。
- 对于$\varepsilon$, 使得对任何固定的$\varepsilon$, 该模式是一个$(1+\varepsilon)$近似算法。如果上述算法以实例规模以$n$的多项式时间运行，称之为多项式时间近似模式。
- 如果运行时间是$\frac{1}{\varepsilon}$的多项式并且是$n$的多项式，则称之为**完全多项式时间近似模式**。

## Vertex-Cover Problem

- **Vertex-Cover Problem**: 给定一个无向图$G=(V,E)$，寻找一个最小的顶点覆盖集$C\subseteq V$，使得对于任意一条边$(u,v)\in E$，至少有一个端点在$C$中。

<img src="Pasted image 20251107190856.png" alt="" width="500">

- 上述算法描述的过程相当简单，首先将所有的边加入到一个集合中，然后从集合中取出任意一条边$(u,v)$，将其两个端点加入到顶点覆盖集$C$中，并将所有与这两个端点相连的边从集合中删除，直到集合为空。
- 上述近似算法为2-近似算法。对于第四行选出的边，，一定没有公共顶点，对于最优覆盖$|C^*| \geq |A|$，而近似解给出的有$|C|= 2|A|$。可以证明该算法的近似比为2，即
$$
\max\left(\frac{|C|}{|C^*|}, \frac{|C^*|}{|C|}\right) \le 2
$$

## TSP

<img src="Pasted image 20251107190914.png" alt="" width="500">

近似算法的思路为先找出原图的一棵最小生成树，然后对这棵树上的结点进行先序遍历得到原图的一个TSP问题的近似解。

上面是在满足三角不等式情况下的近似算法，是2-近似算法。TSP问题的最优解一定是大于最小生成树的边权和的（H回路删掉一条边之后得到的是生成树）。并且对生成树进行**完全遍历**得到的代价恰好为生成树权值和的两倍，并且由于三角不等式，少经过一个结点不会增加总代价，于是得证。

> 定理 35.3 如果 $\mathcal{P}\neq \mathcal{NP}$, 则对任何常数$\rho$, 一般旅行商问题不存在具有近似比为$\rho$的多项式时间近似算法。

因为可以恰当地赋予权值，将一个一般图的哈密尔顿回路问题的实例归约到TSP问题，从而能在多项式时间内得到任意图的哈密尔顿回路问题的精确解法。
$$
c(u, v) = \begin{cases} 
1 & \text{如果 } (u, v) \in E \\ 
\rho |V| + 1 & \text{其他} 
\end{cases}
$$
## Set Cover Problem

集合覆盖问题的一个实例 $(X, \mathcal{F})$ 由一个有穷集 $X$ 和一个 $X$ 的子集族 $\mathcal{F}$ 构成，且 $X$ 的每一个元素至少属于 $\mathcal{F}$ 中的一个子集：

$$
X = \bigcup_{S \in \mathcal{F}} S
$$
我们说一个子集 $S \in \mathcal{F}$ 覆盖了它的元素。这个问题是要找到一个最小规模子集 $\mathcal{C} \subseteq \mathcal{F}$，使其成员覆盖 $X$ 的所有成员：
$$
X = \bigcup_{S \in \mathcal{C}} S
$$

<img src="Pasted image 20251107190959.png" alt="" width="500">

这个贪心近似算法的思路是寻找能覆盖最多尚未被覆盖的元素的集合S

可以证明上面的算法是一个$\rho =H(\max\{ |S| : S \in \mathcal{F}\})$
近似算法，其中$H(k)$是第$k$个调和数。对于任意的集合覆盖问题，近似比至少为$\ln |X|+1$，因此该算法是一个$\ln |X|+1$近似算法。

在贪心选择的过程中，每选择一个一个集合有代价为1，，并且将这个代价平均分给所有第一次覆盖的元素，如果某个元素被$S_k$第一次覆盖那么代价为
$$
c_x = \frac{1}{|S_k- (S_1 \cup S_2 \cup \cdots S_{k-1} )|}
$$

<img src="Pasted image 20251107191028.png" alt="">

## Randomized Approximation Algorithms

随机化算法可以计算期望代价$C$，如果期望代价在最优解代价的一个因子$\rho(n)$之内，则称这个算法为$\rho(n)$-近似算法。
### MAX-3-CNF-SAT

最大三合取范式问题是指给定一个布尔公式，寻找一个变量赋值使得公式中的子句数目最大化。

一个平凡的算法即将所有的变量随机赋值为真或假，然后计算公式的值。可以证明该算法是一个$\frac{8}{7}$近似算法。

事实上对于一个任意给定的子句，其赋值有8种可能，对应的成假赋值只有一种可能，因此期望值为

$$\mathbb{E}[C] = \frac{8}{7}C^*$$

### Weighed Vertex Cover

<img src="{C6FC1B3C-D4D5-4DF5-97A8-BE8B40B75F71}.png" alt="">

上述近似算法是一个$\rho = 2$近似算法。可以证明该算法的近似比为2

## Subset-Sum Problem

问题描述：给定一个整数集合 $S$ 和一个整数 $t$，判断是否存在一个子集 $S' \subseteq S$ 使得 $S'$ 中元素的和等于 $t$。

上面的问题是NP-完全的。可以给出指数时间内的解法：
<img src="Pasted image 20251107191050.png" alt="" width="500">
上面的迭代步骤为，利用$\{x_1 \cdots x_{i-1}\}$的子集和来计算$\{x_1 \cdots x_i\}$的子集和。

给出近似算法：
<img src="Pasted image 20251107191107.png" alt="" width="500">

上面的`trim`是将有序的集合进行筛选，将去掉与现存的较近的元素
<img src="Pasted image 20251107191123.png" alt="" width="500">
上面的近似算法是一个完全多项式时间近似算法