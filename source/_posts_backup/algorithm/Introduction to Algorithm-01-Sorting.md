---
title: Introduction to Algorithm-01-Sorting
date: 2025-06-15 13:45
tags:
    - Sorting
    - Algorithm
categories:
  - Algorithm
math: true
syntax_converted: true

---

# notation

{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.27' %}
在 RAM 模型中，我们并不试图对当代计算机中常见的内存层次进行建模
{% endnote %}

> 在算法分析的时候不用考虑内外存的问题，但是在某些问题中，如B树需要考虑。

{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.42' %}
一般来说，当渐近记号出现在某个公式中时，我们将其解释为代表某个我们不关注名称的匿名函数。
{% endnote %}
> 记号一方面理解为集合，同时可以理解为是某个“匿名函数”

{% note warning '算法导论 第三版 可复制 有目录 (Thomas H. Cormen, Charles E. Leiserson etc.), p.43' %}
无论怎样选择等号左边的匿名函数，总有一种办法来选择等号右边的匿名函数使等式成立。因此，我们的例子（如 $2n^2 + \Theta(n) = \Theta(n^2)$）意指对任意函数 $f(n) \in \Theta(n)$，存在某个函数 $g(n) \in \Theta(n^2)$，使得对所有的 $n$，有 $2n^2 + f(n) = g(n)$ 。换句话说，等式右边比左边提供的细节更粗糙。
{% endnote %}

> 对于等号两边的记号，其实是地位不相同的

# Recurrence

求解递推式的三种方法：
- 代入法：猜测递推式的形式，使用数学归纳法证明
- 递归树法：将递推式转化为树的形式，计算每一层的复杂度
- 主定理：直接使用主定理来求解递推式：


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.63' %}
有时你可能正确猜出了递归式解的渐近界，但莫名其妙地在归纳证明时失败了。问题常常出在归纳假设不够强，无法证出准确的界。当遇到这种障碍时，如果修改猜测，将它减去一个低阶的项，数学证明常常能顺利进行。
注意，在处理这里的标记的时候要代入“确定的”参数，因为利用记号运算的时候可能有系数的变化！
{% endnote %}


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.63' %}
错误在于我们并未证出与归纳假设严格一致的形式
{% endnote %}


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.73' %}
因此我们可以看到在深度 $\log n$，问题规模至多是常数。
这里证明的是：这一层由于问题规模是常数，所以对应的递归式的值也是常数。
{% endnote %}


#### Changing variables

$T(n) =2T(\sqrt{n})+\lg n$
let $m =\lg n$, then $T(2^m) =2T(2^{\frac{m}{2}}) +m$
let $S(m)=T(2^m)$
 we have $S(m) =2S(m) +m$
 so $T(n)=\lg n\lg \lg n$
 
#### 大整数乘法

直接相乘或者直接将两个大整数分为大小相等的小整数，时间复杂度均为$\Theta(n^2)$

做拆分
$$
\begin{aligned}
&x=10^{\frac{n}{2}}a+b\\
&y=10^{\frac{n}{2}}c+d\\
&xy=10^n(ac)+10^{\frac{n}{2}}(ad+bc)+bd\\
&\text{求解}ad+bc = (a+b)(c+d)-ac-bd \\
&T(n) = 3T(\frac{n}{2}) +\Theta(n)\\
&T(n) = \Theta(n^{\lg 3})

\end{aligned}
$$

#### Finding the closest pair of points

[平面最近点对 - OI Wiki](https://oi-wiki.org/geometry/nearest-points/)


- 给定n个点，求解距离最近的两个点之间的距离。

- 暴力解法时间复杂度为$\Theta(n^2)$

分治法解决，按照x坐标选择中位数点，作划分，**关键在于分割线两侧的点最近距离的求解**
对于上述问题作适当的剪枝。假设左右的子问题已经解决，那么在中间的区域内，只需要考虑距离中线小于$\delta$的点。（$\delta$为左右子问题的最小距离）

- 你可以把 `inplace_merge(..., cmp_y)` 看作是在完成了基于 x 的分割和两侧子问题求解之后，**就地把当前段按 y 重新排列**，以便做带状区间的线性扫描。
- 此时，段内的 x 顺序确实被打乱，但这没关系，因为分割早就在递归入口时完成了，之后只需要按 y 顺序合并和检查就足够了。
```C++
void rec(int l, int r) {
  if (r - l <= 3) {
    for (int i = l; i <= r; ++i)
      for (int j = i + 1; j <= r; ++j)
        upd_ans(a[i], a[j]);
    sort(a + l, a + r + 1, &cmp_y);
    return;
  }

  int m = (l + r) >> 1;
  int midx = a[m].x;
  rec(l, m), rec(m + 1, r);
  inplace_merge(a + l, a + m + 1, a + r + 1, &cmp_y);

  static pt t[MAXN];
  int tsz = 0;
  for (int i = l; i <= r; ++i)
    if (abs(a[i].x - midx) < mindist) {
      for (int j = tsz - 1; j >= 0 && a[i].y - t[j].y < mindist; --j)
        upd_ans(a[i], t[j]);
      t[tsz++] = a[i];
    }
}

```


### 随机算法

#### 候选人选择


> 你承诺在任何时候，都要找最适合的人来担任这项职务。因此，你决定在面试完每个应聘者后，如果该应聘者比目前的办公助理更合适， 就会辞掉当前的办公助理，然后聘用新的。你愿意为该策略付费，但希望能够估算该费用会是多少。

对于随机化算法的分析，需要对于条件的概率假设。这里的假设是对于候选者的任意一种排列是等可能性的。下面分析的情况是每个候选者的分数标准化为一个0~1均匀分布的随机变量，可以计算得到后一个人比前一个人更优秀的概率为$\frac{1}{n}$，也就是每个候选者都有$\frac{1}{n}$的概率被选中。
$$
\begin{aligned}
&f(x_n)= nx_n^{n-1}  \\
&f(x_{n+1}) =1 \\
&f(x_n+1,x_n) = nx_n^{n-1} \\
&P(X_{n+1}>X_n) = \iint f(x_{n+1},x_n) dx_{n+1}dx_n = \int_{0}^{1}dx_n \int_{x_n}^{1} n(1-x_n)^{n-1}dx_{n+1} = \frac{1}{n+1}
\end{aligned}
$$
上面的分析过程同样是可以用来实现生成对应的分布，在离散的情况下可以从$0 \sim n^3$中选取对应的优先级。

对于上面的两种生成随机排列的等价性可以证明：
{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.84' %}
引理 5.4：假设所有优先级都不同，则过程 PERMUTE-BY-SORTING 产生输入的均匀随机排列
{% endnote %}


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.85' %}
你可能会这样想，要证明一个排列是均匀随机排列，只要证明对于每个元素 $A[i]$, 它排在位置 $j$ 的概率是$\frac{1}{n}$。练习 5. 3-4 证明这个弱条件实际上并不充分。
{% endnote %}

> 事实上对于上述时间的独立性也有要求，同时要求两个元素之间排序是相互独立的。


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.83' %}
我们让随机发生在算法上，而不是在输入分布上。
给定一个输入，如上面的 A3, 我们无法说出最大值会被更新多少次，因为此变量在每次运行该算法时都不同。第一次在 A3 上运行这个算法时，可能会产生排列 A1 并执行 10 次更新；但第二次运行算法时，可能会产生排列 A2 并只执行 1 次更新。第三次执行时，可能会产生其他次数的更新。
每次运行这个算法时，执行依赖于随机选择，而且很可能和前一次算法的执行不同。对于该算法及许多其他的随机算法，没有特别的输入会引出它的最坏情况行为。
即使你最坏的敌人也无法产生最坏的输入数组，因为随机排列使得输入次序不再相关。只有在随机数生成器产生一个“不走运”的排列时，随机算法才会运行得很差。
{% endnote %}


上面的文字解释的是，算法的效率仅由算法过程中的随机数生成器引起的，和输入不再相关。好处在于，无法仅通过输入某个特定的序列使得程序运行效果最差。
<img src="Pasted image 20251107185249.png" alt="" width="450">

上面的算法同样能给出一个随机排列的序列。

##### Online hiring

一种经典的选择为，先拒绝前$k$个候选者，然后在后面的候选者中选择第一个比前$k$个候选者都优秀的候选者。

- $S$：算法成功事件，即最终选中的是最优秀的人  
- $S_i$：第 $i$ 个候选人是最优秀且被选中  
- $B_i$：第 $i$ 人是最优秀的（发生概率为 $1/n$）  
- $O_i$：第 $i$ 人是你看到的第一个超过前 $k$ 个的人（你会选中他）

由乘法法则，有：
$$
\Pr\{S\} = \sum_{i=k+1}^{n} \Pr\{S_i\} = \sum_{i=k+1}^{n} \Pr\{B_i\} \cdot \Pr\{O_i\}
$$
其中：
$$
\Pr\{B_i\} = \frac{1}{n}, \quad \Pr\{O_i\} = \frac{k}{i - 1}
$$
代入可得：
$$
\Pr\{S\} = \sum_{i=k+1}^{n} \frac{1}{n} \cdot \frac{k}{i - 1} = \frac{k}{n} \sum_{i=k+1}^{n} \frac{1}{i - 1} = \frac{k}{n} \sum_{i=k}^{n-1} \frac{1}{i}
$$
使用积分对调和级数进行逼近，有：
$$
\int_k^n \frac{1}{x} \,dx \le \sum_{i=k}^{n-1} \frac{1}{i} \le \int_{k-1}^{n-1} \frac{1}{x} \,dx
$$
因此：
$$
\frac{k}{n}(\ln n - \ln k) \le \Pr\{S\} \le \frac{k}{n}(\ln(n - 1) - \ln(k - 1))
$$
我们希望最大化下界：
$$
\Pr\{S\} = \frac{k}{n} (\ln n - \ln k)
$$
对 $k$ 求导并令导数为 0，有：
$$
\frac{d}{dk} \left( \frac{k}{n} (\ln n - \ln k) \right) = \frac{1}{n} (\ln n - \ln k - 1)
$$
令导数为 0 解得：
$$
\ln n - \ln k - 1 = 0 \Rightarrow \ln k = \ln n - 1 \Rightarrow k = \frac{n}{e}
$$

# Sorting

| 排序方法 | 描述                                 | 最坏时间复杂度 $O(n^2)$ | 平均时间复杂度 $O(n \log n)$ | 最好时间复杂度 $O(n)$ | 空间复杂度 $O(1)$ | 稳定性              |
| ---- | ---------------------------------- | ---------------- | --------------------- | -------------- | ------------ | ---------------- |
| 冒泡排序 | 通过相邻元素的比较和交换来排序                    | $O(n^2)$         | $O(n^2)$              | $O(n)$         | $O(1)$       | 稳定               |
| 选择排序 | 每次从未排序部分选择最小元素放到已排序部分的末尾           | $O(n^2)$         | $O(n^2)$              | $O(n^2)$       | $O(1)$       | 不稳定              |
| 插入排序 | 将每个新元素插入到已排好序的部分中的适当位置             | $O(n^2)$         | $O(n^2)$              | $O(n)$         | $O(1)$       | 稳定               |
| 归并排序 | 分治法，将列表分成两部分分别排序后再合并               | $O(n \log n)$    | $O(n \log n)$         | $O(n \log n)$  | $O(n)$       | 稳定               |
| 快速排序 | 通过基准元素将数组分为两部分，递归地对这两部分进行排序        | $O(n^2)$         | $O(n \log n)$         | $O(n \log n)$  | $O(\log n)$  | 一般不稳定，但通过随机化可以改进 |
| 希尔排序 | 使用增量序列分组，对每组使用插入排序，最后一次性对所有组进行插入排序 | 通常 $O(n^{1.5})$  | $O(n \log n)$         | $O(n)$         | $O(1)$       | 不稳定              |
| 堆排序  | 利用堆这种数据结构设计的排序算法                   | $O(n \log n)$    | $O(n \log n)$         | $O(n \log n)$  | $O(1)$       | 不稳定              |
| 计数排序 | 计算每个元素的出现次数，根据出现次数确定元素的最终位置        | $O(n + k)$       | $O(n + k)$            | $O(n + k)$     | $O(k)$       | 稳定               |
| 桶排序  | 将数据分配到有限数量的桶里，然后对每个桶内的数据进行排序       | $O(n + k)$       | $O(n^2)$              | $O(n)$         | $O(n + k)$   | 稳定               |
| 基数排序 | 通过键值的部分信息，将数据分配到各个桶中，逐级排序          | $O(n \cdot k)$   | $O(n \cdot k)$        | $O(n \cdot k)$ | $O(n + k)$   | 稳定               |

[^1]


### QuickSort

- 分治算法
- 就地算法

算法为：
```python
def quick_sort(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

def partition(arr, low, high):
    pivot = arr[high]  # 选最后一个元素作为基准
    i = low - 1         # 小于等于 pivot 的区间尾部
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    # 把 pivot 放到中间
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

```

算法过程：
- 选取主元来划分数组为两个子数组，使得左侧的字数组总是小于主元小于右侧数组。
- 递归地对左右两边的子数组处理

对于极端不平衡的划分：
{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.112' %}
利用代入法可以直接得到递归式$T(n)=T(n-1)+\Theta(n)$的解为$T(n)=\Theta(n^2)$
{% endnote %}


{% note warning '算法导论 第三版 可复制 有目录 (Thmos.H.Cormen ,Charles E. Leiserson etc.) (Z-Library), p.113' %}
任何一种常数比例的划分都会产生深度为$\Theta (\lg n)$ 的递归树，其中每一层的时间代价都是$O(n)$ 。
{% endnote %}


对于任意常数比例的划分，例如1:9的划分可以做如下分析：
$$
T(n) = T(\frac{n}{10}) +T(\frac{9n}{10}) +\Theta(n)
$$
可以求解为：
$$
n\log_{10}n \leq T(n) \leq n\log_{\frac{10}{9}}
$$
也就是有渐进紧界。


对于最糟糕的情况和最好的情况**交替出现**的情况，仍然能获得$\Theta(n \lg n)$的时间复杂度。
#### Randomized QuickSort

对主元做出随机选取（假设选择主元的事件是相互独立的）。

- 无需对输入序列做出任何的假设
- 没有一种输入你能引发最差的情况
- 最差只是由随机数生成器引起，产生完全顺序或完全逆序数组时为最坏的情况

为了分析这种随机化的算法：
$$
\begin{equation*}

\begin{aligned}
 &x_k \quad \text{stands for indicator random variable}\\ 
&x_k= \begin{cases}
1 & \text {if we have k, n-k-1 split} \\
0 & \text{otherwise}
\end{cases}\\
& T(n) = \sum_{k=0}^{n-1} x_k (T(k)+T(n-1-k)+\Theta(n)) \\
\Rightarrow & E[T(n)] = \sum_{k=0}^{n-1} E[x_k] E[T(k)+T(n-k-1)+\Theta(n)] \\
\Rightarrow & E[T(n)] = \sum_{k=0}^{n-1} \frac{2}{n} E[T(k)] + \Theta(n) = \frac{2}{n} \sum_{k=2}^{n-1} E[T(k)] + \Theta(n)  \\
&\text{choose a big enough constant prove that:}\\
& E[T(n)] \leq a n \lg n \quad \text{use induction to   prove this}\\
&\text{use fact:}\sum_{k=2}^{n-1} k \lg k \leq \frac{1}{2}n^2 \lg n -\frac{1}{8}n^2 \\
&E[T(n) ] \leq \frac{2}{n} \sum_{k=2}^{n-1} a k\lg k +\Theta (n) \leq  \frac{a}{n}  (n^2 \lg n - \frac{n^2}{4}) + \Theta(n) \\
\Rightarrow &E[T(n)] \leq a n \lg n - \frac{a}{4}n +\Theta(n) \leq a n \lg n \quad \text{choose a big enough constant a}

\end{aligned}
\end{equation*}


$$

可以给出另一种证明方式：
根据quicksort的特性，算法的复杂度即为进行比较的次数X，算法复杂度为$O(X+n)$。$A=<z_1\dots z_n>$ 是有序的的数组。令$X_{ij}$作为$z_i,z_j$进行比较的次数。可以得到：
$$
X= \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} X_{ij}
$$
- 在快速排序的过程中，任意两个元素间之多进行一次比较（主元参与比较后不再参与比较）
- $z_i,z_{i+1}$一定发生比较
- $z_i,z_j$发生比较的概率为$\frac{2}{j-i+1}$（在$z_i \sim z_j$之间选择主元时恰好选到$z_i$或$z_j$的概率）
$$
E[X] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} E[X_{ij}] = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \frac{2}{j-i+1} = 2n\sum_{i=1}^{n-1} \sum_{j=1}^{n-i} \frac{1}{j} = 2n\sum_{i=1}^{n-1} H_{n-i} \leq 2n\sum_{i=1}^{n-1} H_i = 2n\lg n
$$

快速排序的优点在于：访问模式具有较好的局部性，对于CPU的**缓存**友好，可以显著提高速度。

### Comparison Sorting
只能使用比较来进行排序的算法，这种排序算法的时间复杂度下界为$\Omega(n\lg n)$
#### 决策树

- 当每次进行比较的时候，左侧子树代表$a_i\leq a_j$，右侧子树代表$a_i>a_j$分别是两种不同的情况。
- 决策树表示的是算法执行过程中所有的可能流程。
- 一个决策树的叶子节点数目为$n!$，因为有$n!$种排列方式。
- 每一种排序算法都可以对应一个决策树，而每一个决策树都可以对应一个排序算法。
- 算法的运行时间等于决策树的高度。

#### 证明
$$
\begin{aligned}
&\text{考虑决策树的高度为h，叶子节点数目为}n!\\
&n! \leq 2^h \\
\Rightarrow & h \geq \lg n! = \Theta(n\lg n)
\end{aligned}
$$
对于上面的证明是基于一个确定性的排序算法而言的，因为只考虑了一棵树。随机算法虽然可能产生不同的树，但是对于任意的树都有上述结论，所以仍然不可能突破下界。

### Sort in Linear time

#### Counting Sort

要求输入的数据是有确定范围的数。
算法流程：
- 统计每个元素出现的次数
- 对于这个数组做前缀和（对应的是相同大小的数的最大的位置）
- 将原数组的元素放到对应的位置上，并将对应的前缀和自减（逆序）
- 时间复杂度为$O(n+k)$，$n$为数组的长度，k为数组中元素的范围。
- 如果$k=O(n)$则时间复杂度为$O(n)$，是一个极好的排序算法。
- 稳定的排序算法

代码为：
```python
def counting_sort(arr, k):
		n = len(arr)
		count = [0] * (k + 1)
		output = [0] * n

		# 统计每个元素出现的次数
		for num in arr:
				count[num] += 1

		# 前缀和
		for i in range(1, k + 1):
				count[i] += count[i - 1]

		# 放置元素到输出数组，保证排序的稳定性
		for i in range(n - 1, -1, -1):
				output[count[arr[i]] - 1] = arr[i]
				count[arr[i]] -= 1

		return output
```

#### Radix Sort
- 首先对最后一位进行排序，然后对倒数第二位进行排序，直到第一位。
- 稳定的排序算法
- 整数的范围为$[0,2^b)$把整数用二进制表示，然后将r位比特放在一起（事实上是$2^r$进制的counting sort）

$$
\begin{aligned}
\text{时间复杂度：}& O (\frac{b}{r}(n+2^r))\\
=& O(\frac{b}{\lg n}(n+n)) \quad \text{要求 }n \geq 2^r \text{并取r的最大值}\\
=& O(\frac{b}{\lg n}n) \quad \text{如果数据在}O(2^b) \text{范围内}\\
= & O(nd) \quad \text{如果数据在}O(n^d) \text{范围内}\\

\end{aligned}
$$
#### Bucket Sort

- 将数据分为m个桶，每个桶内的数据范围是$\frac{1}{m}$，然后对每个桶内的数据进行排序。
- 对于每个桶内的数据进行排序，可以使用插入排序。

代码为：
```python
def bucket_sort(arr, bucket_size=5):
		if len(arr) == 0:
				return arr

		# 计算最小值和最大值
		min_value = min(arr)
		max_value = max(arr)

		# 创建桶
		bucket_count = (max_value - min_value) // bucket_size + 1
		buckets = [[] for _ in range(bucket_count)]

		# 将元素放入对应的桶中
		for num in arr:
				index = (num - min_value) // bucket_size
				buckets[index].append(num)

		# 对每个桶进行排序，并合并结果
		sorted_array = []
		for bucket in buckets:
				sorted_array.extend(sorted(bucket))

		return sorted_array
```

时间复杂度的分析为：
$$
T(n) = \Theta(n) + \sum_{i=0}^{n-1}O(n_i^2)
$$
取期望得到：
对于$X_{ij}$为指标随机变量，设为第$j$个元素在第$i$个桶内的元素，$n_i$为桶内元素的个数。
$$
\begin{aligned}
E [T(n)] =& \Theta(n) + \sum_{i=0}^{n-1}O(E[n_i^2]) \\
= & \Theta(n) + \sum_{i=0}^{n-1}O \left( E [(\sum_{j=1}^{n} X_{ij})^2] \right) \\
= & \Theta(n) + \sum_{i=0}^{n-1}O \left( E [\sum_{j=1}^{n} X_{ij}^2]) \right)+ \sum_{i=0}^{n-1}O \left( E [\sum_{k \neq j} X_{ij} X_{ik}] \right) \\
= & \Theta(n) + \sum_{i=0}^{n-1}O \left( \sum_{j=1}^{n}\frac{1}{n} + \sum_{j \neq k} \frac{1}{n^2} \right) \\
= & \Theta(n) + \sum_{i=0}^{n-1}O \left( n \cdot \frac{1}{n} + n(n-1)\cdot \frac{1}{n^2} \right) \\
= & \Theta(n) + \sum_{i=0}^{n-1}O \left( 1 + \frac{n-1}{n} \right) \\
= & \Theta(n) \\
\end{aligned}
$$


### Medians and Order Statistics

#### Selection in expected linear time

使用分治法来求解第$i$个元素的值，使用随机化选择的方法来求解。
```python
def randomized_select(arr, left, right, i):
		if left == right:
				return arr[left]

		pivot_index = partition(arr, left, right)
		k = pivot_index - left + 1

		if i == k:
				return arr[pivot_index]
		elif i < k:
				return randomized_select(arr, left, pivot_index - 1, i)
		else:
				return randomized_select(arr, pivot_index + 1, right, i - k)
```



#### Worst case linear time

> 1. 将输入数组的 $n$ 个元素划分为 $n/5$ 组，每组 5 个元素，且至多只有一组由剩下的 $n \bmod 5$ 个元素组成。
> 2. 寻找这 $n/5$ 组中每一组的中位数：
> 	1. 对每组元素使用插入排序；
> 	2. 然后确定每组排序后的中位数。
> 3. 对第 2 步中找出的 $n/5$ 个中位数，递归调用 `SELECT` 以找出这些中位数的中位数 $x$（若中位数数量为偶数，则取较小者为 $x$）。
> 4. 使用修改版的 `PARTITION`，以中位数的中位数 $x$ 为划分基准，将原数组划分为小于 $x$ 和大于等于 $x$ 的两个部分。
> 	1. 令 $k$ 为划分后低区中元素数量加 1，即 $x$ 是第 $k$ 小的元素；
> 	2. 这样划分的高区就有 $n - k$ 个元素。
> 5. 比较 $i$ 和 $k$：
> 	1. 若 $i = k$，则返回 $x$；
> 	2. 若 $i < k$，在低区递归调用 `SELECT` 查找第 $i$ 小的元素；
> 	3. 若 $i > k$，在高区递归调用 `SELECT` 查找第 $i - k$ 小的元素。

可以在最坏的情况下得到时间复杂度为$O(n)$的选择算法



[^1]: [算法设计与分析复习 - ReadMe 软件学院互助文档](https://ssast-readme.github.io/note/algorithm/%E7%AE%97%E6%B3%95%E6%9C%9F%E6%9C%AB%E5%A4%8D%E4%B9%A0%E7%AC%94%E8%AE%B0_24%E6%98%A5/#3)