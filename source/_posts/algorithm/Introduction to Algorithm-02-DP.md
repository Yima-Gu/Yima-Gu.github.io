---
title: Introduction to Algorithm-02-DP
date: 2025-06-15 14:45
tags:
    - Dynamic Programming
    - Algorithm
categories:
  - Algorithm
math: true
syntax_converted: true
---

	
#### 最大子段和

考虑以数组中元素为结尾的最大子数组和，对于数组中的第i个元素，有两种选择：
- 以第$i-1$个元素结尾的最大子数组和加上第$i$个元素
- 第i个元素本身

$$
dp[i] = max(dp[i-1]+nums[i], nums[i])
$$
上述公式表达的是状态转移方程。

**如何构造出最优解？**

## Steps of Dynamic Programming

- Characterize the structure of an optimal solution.
- Recursively define the value of an optimal solution.
- Compute the value of an optimal solution (typically in a bottom-up fashion).
- Construct an optimal solution from computed information.

### Rod Cutting


描述：给定一根长度为n的钢条和一个价格表，求切割钢条的方案，使得收益最大
#### Characterize the structure of an optimal solution

假设我们知道最优解的切割方案，那么最优解一定是由最优解的子问题组成的。在这个问题中，对于这根钢条在任意的位置进行切割，那么一定有左右子问题都是最优解，否则违反最优性。
$$
r_n = max(p_n, r_1+r_{n-1}, r_2+r_{n-2}, ..., r_{n-1}+r_1)
$$

上述得到的状态转移方程依然有些复杂，继续考虑如何简化。
考虑在切割钢条的最左边的一刀，假设在长度为i的位置切割，那么问题就被分解为了两个子问题，一个是长度为$i$的钢条的价格，另一个是长度为$n-i$的钢条的最优解。这样就可以得到一个更简单的状态转移方程。
$$
r_n = max(p_i + r_{n-i}),\; i=1,2,...,n
$$
如果直接使用递归的方法求解，那么会有很多的重复的子问题，算法的复杂度为$O(2^n)$，所以我们需要使用动态规划的方法。
可以使用带有备忘录的自顶向下的方法，也可以使用自底向上的方法。
*如何构造出最优解？*
对于任意的长度，只需要记录下最左边的切割位置即可，可以使用一个数组来记录。

```python
def cut_rod(p, n):
		r = [0] * (n+1)
		s = [0] * (n+1)
		for j in range(1, n+1):
				q = -1
				for i in range(1, j+1):
						if q < p[i] + r[j-i]:
								q = p[i] + r[j-i]
								s[j] = i
				r[j] = q
		return r, s
```

使用动态规划的时间复杂度为$O(n^2)$。

### Matrix-chain Multiplication

描述：给定$n$个矩阵，求矩阵链乘法的最优解。

不同的加括号的方法运算量是不一样的：
举例：$A_{10 \times 100}$$B_{100 \times 5}$$C_{5 \times 50}$
- $(AB)C$需要$10 \times 100 \times 5 + 10 \times 5 \times 50 = 7500$
- $A(BC)$需要$100 \times 5 \times 50 + 10 \times 100 \times 50 = 75000$

对于$n$个矩阵的乘法的运算，加括号的方式为：
$$
P(n) = \begin{cases}
1 & n=1 \\
\sum_{k=1}^{n-1}P(k)P(n-k) & n \geq 2
\end{cases}
$$
得到的是一个卡特兰数，$P(n) = \frac{1}{n+1}\binom{2n}{n}$ 

#### Characterize the structure of an optimal solution

考虑一般性的问题，对于矩阵链乘法，我们可以将问题分解为两个子问题，一个是前半部分的最优解，另一个是后半部分的最优解。这样就可以得到一个状态转移方程。
其中的$p$数组保存的是矩阵的维度，$p_{i-1}$表示第$i$个矩阵的行数，$p_i$表示第$i$个矩阵的列数。
$$
m[i,j] = \begin{cases}
0 & i=j \\
\min_{i \leq k < j} (m[i,k]+m[k+1,j]+p_{i-1}p_kp_j) & i<j
\end{cases}
$$
上述的状态转移方程是一个自底向上的方法，可以使用一个二维数组来记录。
```python
def matrix_chain_order(p):
		n = len(p) - 1
		m = [[0] * n for _ in range(n)]
		s = [[0] * n for _ in range(n)]
		for l in range(2, n+1):
				for i in range(n-l+1):
						j = i + l - 1
						m[i][j] = float('inf')
						for k in range(i, j):
								q = m[i][ k]+ m[k+1][j] + p[i]*p[k+1]*p[j+1]
								if q < m[i][j]:
										m[i][j] = q
										s[i][j] = k #保存划分方式
		return m, s
```
上述算法的时间复杂度为$O(n^3)$


希望构造最优解只需要加一个数组保存每一个子问题的括号添加方式即可。使用的是`s`数组保存，`s[i][j]`保存的是第$i$个矩阵和第$j$个矩阵之间的分隔。

### Elements of Dynamic Programming

- Optimal substructure
	- 如果一个问题的最优解包含了其子问题的最优解，那么这个问题就具有最优子结构。
- Overlapping subproblems
	- 递归算法会反复地求解相同的子问题，而不是一直产生新的子问题。

**How to discover optimal substructure?**
- Make a choice to split the problem into subproblems.
- Just assume you are given the choice that leads to an optimal solution;
- Given this choice, try to best characterize the resulting space of subproblems;
- Show the subproblems chosen are optimal by using a “cut-and-paste” technique.

#### 最长简单路径和最短简单路径中的最优子结构分析


##### 1. 最短简单路径
- **是否具有最优子结构**：是 ，在正权简单图中，最短路径一定是简单路径。
- **原因**：  
  最短路径的最优解可以由其子路径的最优解构成。例如，若路径 $A \to B \to C$ 是 A 到 C 的最短路径，则 $A \to B$ 和 $B \to C$ 必须分别是 A 到 B 和 B 到 C 的最短路径。这一性质使得动态规划算法（如 Dijkstra、Bellman-Ford）能够高效求解。  

##### 2. 最长简单路径
- **是否具有最优子结构**：否  
- **原因**：  
  最长简单路径的子路径不一定是局部最优的。由于路径中不能重复节点，选择某条子路径可能会限制后续路径的选择，导致无法通过局部最优解组合成全局最优解。此外，最长路径问题在一般图中是 NP 难的，间接表明其缺乏最优子结构（否则可能存在多项式时间的动态规划解法）。
- 没有最优子结构，关键在于分解得到的子问题是不是**无关**的

### Longest Common Subsequence

问题描述：给定两个序列 $X = \{x_1, x_2, ..., x_m\}$ 和 $Y = \{y_1, y_2, ..., y_n\}$，求 X 和 Y 的最长公共子序列（LCS）。子序列是指从原序列中删除若干元素（可以为空）后得到的序列。
#### Characterize the structure of an optimal solution

对于两个序列的最长公共子序列，可以将问题分解为两个**子问题**，一个是去掉序列 $X$ 的最后一个元素，另一个是去掉序列 $Y$ 的最后一个元素。这样就可以得到一个状态转移方程。（子问题的分割不一定是从中间进行分割的，对于这个问题，是将最后的元素删除后再进行比较）

$$
c[i,j] = \begin{cases}
0 & i=0 \text{ or } j=0 \\
c[i-1,j-1]+1 & i,j>0 \text{ and } x_i=y_j \\
\max(c[i-1,j], c[i,j-1]) & i,j>0 \text{ and } x_i \neq y_j
\end{cases}
$$
上述状态转移方程考虑了三种情况：
- $x_i=y_j$，则最长公共子序列的长度为 $c[i-1,j-1]+1$；
- $x_i \neq y_j$，则最长公共子序列的长度为 $\max(c[i-1,j], c[i,j-1])$；
- $i=0$ 或 $j=0$，则最长公共子序列的长度为 0。

下面的代码使用了自底向上的方法求解最长公共子序列的长度。

```python
def lcs_length(X, Y):
		m, n = len(X), len(Y)
		c = [[0] * (n+1) for _ in range(m+1)]
		b = [[0] * (n+1) for _ in range(m+1)]
		for i in range(1, m+1):
				for j in range(1, n+1):
						if X[i-1] == Y[j-1]:
								c[i][j] = c[i-1][j-1] + 1
								b[i][j] = '↖'
						elif c[i-1][j] >= c[i][j-1]:
								c[i][j] = c[i-1][j]
								b[i][j] = '↑'
						else:
								c[i][j] = c[i][j-1]
								b[i][j] = '←'
		return c, b
```

上述算法的时间复杂度为 $O(mn)$。

计算最长公共子序列的长度后，可以使用下面的代码构造出最长公共子序列。

```python
def print_lcs(b, X, i, j):
		if i <mark> 0 or j </mark> 0:
				return
		if b[i][j] == '↖':
				print_lcs(b, X, i-1, j-1)
				print(X[i-1], end=' ')
		elif b[i][j] == '↑':
				print_lcs(b, X, i-1, j)
		else:
				print_lcs(b, X, i, j-1)
```

如何找到所有的最长公共子序列？

需要遍历所有的最长公共子序列，有可能通过↑或↓向前试探，可以使用集合来记录所有的情况。
### Optimal Binary Search Trees

问题描述：给定一个由 $n$ 个关键字组成的有序序列 $K = \{k_1, k_2, ..., k_n\}$，以及一个由 $n+1$ 个边界点组成的有序序列 $P = \{ p_1, ..., p_n\}$，其中 $p_i$ 表示关键字 $k_i$ 的概率，对于查找不到的的关键字，称为”哑元“，为$\{d_0 , d_1, d_3 \dots d_n \}$，对应的概率为$\{q_0, q_1, q_2 \dots q_n \}$，$d_i$代表的是在$k_i$和$k_{i+1}$之间的哑元的概率。有$\sum_{i=0}^{n}q_i + \sum_{i=1}^n p_i =1$，求一棵二叉搜索树，使得期望搜索代价最小。注意在二叉搜索树中，中序遍历的意义上整体输出是有序的。
目标是最小化期望搜索代价，搜索代价是指搜索的路径长度加1。
$$
E = \sum_{i=1}^{n}p_i \cdot (depth(k_i)+1) + \sum_{i=0}^{n}q_i \cdot (depth(d_i)+1)
$$
$$
E = 1 + \sum_{i=1}^{n}p_i \cdot depth(k_i) + \sum_{i=0}^{n}q_i \cdot depth(d_i)
$$
其中$depth(k_i)$表示关键字$k_i$在二叉搜索树中的深度，$depth(d_i)$表示哑元$d_i$在二叉搜索树中的深度。

#### Characterize the structure of an optimal solution

对于最优二叉搜索树问题，可以将问题分解为两个子问题，一个是左子树的最优解，另一个是右子树的最优解。这样就可以得到一个状态转移方程。
$$
e[i,j] = \begin{cases}
q_{i-1} & j=i-1 \\
\min_{i \leq r \leq j}(e[i,r-1]+e[r+1,j]+ w[i,j]) & i \leq j
\end{cases}
$$
$$
w[i,j] = \sum_{k=i}^{j}p_k + \sum_{k=i-1}^{j}q_k
$$
上述状态转移方程考虑了两种情况：
- $j=i-1$，表示没有关键字在区间 $[i,j]$ 中，此时搜索代价为 $q_{i-1}$；
- $i \leq j$，表示区间 $[i,j]$ 中有关键字，此时搜索代价为 $\min_{i \leq r \leq j}(e[i,r-1]+e[r+1,j]+ w[i,j])$。
- $w[i,j]$ 表示区间 $[i,j]$ 中所有关键字和哑元的概率之和。
- $r$ 表示根节点的位置。

下面的代码使用了自底向上的方法求解最优二叉搜索树的期望搜索代价。

```python
def optimal_bst(p, q):
		n = len(p)
		e = [[0] * n for _ in range(n)]
		w = [[0] * n for _ in range(n)]
		root = [[0] * n for _ in range(n)]
		for i in range(n):
				e[i][i] = q[i]
				w[i][i] = q[i]
		for l in range(1, n):
				for i in range(n-l):
						j = i + l
						e[i][j] = float('inf')
						w[i][j] = w[i][j-1] + p[j] + q[j]
						for r in range(i, j+1):
								t = e[i][r-1] + e[r+1][j] + w[i][j]
								if t < e[i][j]:
										e[i][j] = t
										root[i][j] = r
		return e, root
```

上述算法的时间复杂度为 $O(n^3)$。
通过应用 **Knuth 优化**，利用最优二叉搜索树的性质（根节点的位置具有单调性），可以将内层循环的遍历范围从 `[i, j]` 缩小到 `[root[i][j-1], root[i+1][j]]`，从而将总时间复杂度优化到$O(n²)$。

改进后的代码如下：

```python

def optimal_bst(p, q):
		n = len(p)
		e = [[0] * n for _ in range(n)]
		w = [[0] * n for _ in range(n)]
		root = [[0] * n for _ in range(n)]
		for i in range(n):
				e[i][i] = q[i]
				w[i][i] = q[i]
		for l in range(1, n):
				for i in range(n-l):
						j = i + l
						e[i][j] = float('inf')
						w[i][j] = w[i][j-1] + p[j] + q[j]
						for r in range(root[i][j-1], root[i+1][j]+1):
								t = e[i][r-1] + e[r+1][j] + w[i][j]
								if t < e[i][j]:
										e[i][j] = t
										root[i][j] = r
		return e, root
```

要保存最优解的结构，可以使用下面的代码。

```python
def construct_optimal_bst(root, i, j, k):
		if i == j:
				if i < k:
						print('d' + str(i), end=' ')
				else:
						print('k' + str(i), end=' ')
		elif i < j:
				r = root[i][j]
				if k == 0:
						print('k' + str(r), end=' ')
				else:
						print('d' + str(r), end=' ')
				construct_optimal_bst(root, i, r-1, r)
				construct_optimal_bst(root, r+1, j, r)
```

上述代码保存的是根节点的位置，可以通过递归的方式构造出最优二叉搜索树。

### 0-1 Kinapsack 

问题描述：给定n个物品，每个物品都有一个重量和一个价值，如何选择物品使得总重量不超过背包的容量，并且总价值最大？对于每个物品只能选择或者不选择。
#### Characterize the structure of an optimal solution

对于 0-1 背包问题，可以将问题分解为两个子问题：一个是**选择第 $i$ 个物品**，另一个是**不选择第 $i$ 个物品**。根据这两种情况，可以写出如下状态转移方程：

设：

- $dp[i][w]$：表示前 $i$ 个物品在容量为 $w$ 的背包中所能取得的最大价值；
- $v_i$：第 $i$ 个物品的价值；
- $w_i$：第 $i$ 个物品的重量。

状态转移方程为：

$$
dp[i][w] = 
\begin{cases}
dp[i-1][w], & \text{if } w_i > w \\\\
\max(dp[i-1][w],\ dp[i-1][w - w_i] + v_i), & \text{if } w_i \leq w
\end{cases}
$$

上述状态转移方程考虑了两种情况：

- $w_i > w$：表示当前物品的重量超过背包剩余容量，不能选择，只能继承不选第 $i$ 个物品的解；
- $w_i \leq w$：表示可以选择当前物品，最优解为“选”与“不选”两种方案中价值较大的那一个。

这个结构体现了最优子结构性质：

> 一个最优解包含了该问题的一个子问题的最优解，且子问题之间互不重叠。

可以使用自底而上的方式进行求解：

```python
for i in range(1, n+1):         # 遍历物品
    for w in range(0, W+1):     # 遍历容量
        if w_i > w:
            dp[i][w] = dp[i-1][w]
        else:
            dp[i][w] = max(dp[i-1][w], dp[i-1][w - w_i] + v_i)
```

算法的时间复杂度为复杂度为 $O(nW)$，其中 $n$ 是物品数量，$W$ 是背包容量。