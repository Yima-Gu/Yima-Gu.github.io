---
title: Deep Learning Lecture-7
date: 2025-06-21
tags:
  - DeepLearning
math: true
syntax_converted: true

---
## Generative Model



数据分布是生成模型的核心。生成模型的目标是学习数据的分布，然后生成新的数据。目标是**学习数据的分布**，然后生成新的数据。
对生成模型的评价是通过生成的数据的质量来评价的，生成的数据越接近真实数据，生成模型的质量越好。
$$
\theta^* = \arg\max_{\theta} \mathbb{E}_{x \sim p_{data}} \log p_{model}(x|\theta)
$$
![Pasted image 20250330094251](Pasted image 20250330094251.png)

![Pasted image 20250330094404](Pasted image 20250330094404.png)

### GAN: Generative Adversarial Network

对抗机器学习的思想是通过两个网络之间的对抗来学习。生成器和判别器之间的对抗是GAN的核心思想。
使用的博弈问题的思想，使用的是最小化最大的思想。

GAN的思想为：生成器和判别器之间的对抗，生成器生成数据，判别器判断数据的真实性。

目标函数为：
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]
$$
- 当生成器固定时，判别器的目标是最大化判别器的准确率：$\max_D V(D,G)$
- 当判别器固定时，生成器的目标是最小化判别器的准确率：$\min_G V(D,G)$

生成器 $G$ 的实质是将噪声分布 $p_z(z)$（如高斯分布）映射到数据分布 $p_g(x)$。根据概率密度变换定理，若 $G$ 是可逆且光滑的函数，则生成数据的分布为：

$$
p_g(x) = p_z(z) \cdot \left| \det \left( \frac{\partial G^{-1}(x)}{\partial x} \right) \right|
$$

尽管深度神经网络通常不可逆，但通过足够复杂的函数逼近（如多层非线性变换），生成器可以隐式地学习到从 $p_z(z)$ 到 $p_g(x)$ 的映射，覆盖真实分布 $p_{data}(x)$。

**关键点**：  
- 噪声输入的随机性确保了生成数据分布的多样性。  
- 网络的非线性能力允许从简单分布（如高斯分布）逼近复杂分布（如图像像素分布）。

但是GAN的**训练过程是非常困难**的，梯度性质是不好的，因为在比较好的样本中，由于梯度性质的问题会进行较大的更新。

![Pasted image 20250330100434](Pasted image 20250330100434.png)

**训练GAN的技巧**
![Pasted image 20250330100506](Pasted image 20250330100506.png)

#### DCGAN: Deep Convolutional Generative Adversarial Networks

对于生成器和判别器，使用卷积神经网络来进行训练。对于判别器，使用的是较为标准的CNN网络，对于生成器，使用的是转置卷积，先将特征图进行padding，然后进行卷积操作，这样可以获得一个较大的特征图。

- **生成器**：使用转置卷积（反卷积）逐步上采样，生成高分辨率图像。
- **判别器**：使用标准CNN逐步下采样。
- **关键技巧**：批量归一化、Leaky ReLU、全卷积结构。

![Pasted image 20250330102359](Pasted image 20250330102359.png)

证明了泛化定理：在有限的训练样本之下，可以通过训练得到一个泛化的模型。

#### Inception Score

IS 用于衡量生成模型的性能，重点关注两点：  
- **类别明确性**：单个生成样本应明确属于某个类别（对应分类概率尖锐）。  
- **多样性**：生成样本应覆盖多个类别（类别分布均匀）。

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} \left[ \text{KL}\left(p(y|x) \parallel p(y)\right) \right]\right)
$$

- Class Probability Distribution: $p(y|x)$，生成样本 $x$ 属于各个类别的概率，度量的是生成的数据的类别分布和真实数据的类别分布的相似性。
- Marginal Distribution of Generated Data: $p(y)$，度量的是生成的数据的类别分布的多样性，如果生成的数据是单一的称为模式坍塌。

##### KL Divergence

使用KL散度来度量两个分布的相似性：
$$
KL(p||q) =\mathbb{E}_X \left(  \log \frac{p(x)}{q(x)} \right)=\sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}
$$
需要上面的值尽可能偏大

- **KL散度的意义**：  
  - **$p(y|x)$ 尖锐** → 分类概率集中（如某类概率接近 1），此时 $\text{KL}(p(y|x) \parallel p(y))$ 值大。  
  - **$p(y)$ 均匀** → 生成样本覆盖所有类别，$\text{KL}$ 值的期望更大。  
- **取指数的作用**：将对数空间的值转换为正数，放大差异便于比较。

#### FID: Frechet Inception Distance

将生成的数据输入一个网络来提取特征，用得到的特征来拟和两个高斯分布，然后计算两个高斯分布的*Frechet Inception Distance*。这是有显式表达式的。

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$
- **$\mu_r, \mu_g$**：真实数据和生成数据特征的均值向量。  
- **$\Sigma_r, \Sigma_g$**：真实数据和生成数据特征的协方差矩阵。  
- **$\text{Tr}(\cdot)$**：矩阵的迹（对角线元素之和）。


#### Mode Collapse

指生成的数据的多样性不够，类别分布是单一的，这是GAN的一个问题。

原始GAN使用JS散度（$JSD(p_{data} \parallel p_g)$）衡量分布距离，存在：
- **梯度消失**：当 $p_g$ 和 $p_{data}$ 无重叠时，$JSD = \log 2$，梯度为零。
- **模式坍塌**：生成器倾向于生成少数样本。

##### Wasserstein Distance

[沃瑟斯坦度量 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E6%B2%83%E7%91%9F%E6%96%AF%E5%9D%A6%E5%BA%A6%E9%87%8F)

有可能统计距离是不是一个好的距离度量。Wasserstein距离是一个好的距离度量，用推土机距离来度量两个分布的距离。

$$ c(x,y) \mapsto [0, \infty), $$
表示从点$x$运输质量到点$y$的代价。一个从$\mu$到$\nu$的运输方案可以用函数$\gamma(x,y)$来描述，该函数表明从$x$移动到$y$的质量。一个运输方案$\gamma(x,y)$必须满足以下性质：
$$
\begin{aligned}
\int \gamma(x,y) \, \mathrm{d}y = \mu(x), \\
\int \gamma(x,y) \, \mathrm{d}x = \nu(y),
\end{aligned}
$$

前者表示从某一点$x$移到其他所有点的土堆总质量必须等于最初该点$x$上的土堆质量，后者则表示从所有点移到某一点$y$的土堆总质量必须等于最终该点$y$上的土堆质量。
$$
\iint c(x,y) \gamma(x,y) \, \mathrm{d}x \, \mathrm{d}y = \int c(x,y) \, \mathrm{d}\gamma(x,y).
$$

方案$\gamma$并不是唯一的，所有可能的运输方案中代价最低的方案即为最优运输方案。最优运输方案的代价为：

$$
C = \inf_{\gamma \in \Gamma(\mu,\nu)} \int c(x,y) \, \mathrm{d}\gamma(x,y).
$$
##### Wasserstein GAN

![Pasted image 20250330141642](Pasted image 20250330141642.png)

![Pasted image 20250330142308](Pasted image 20250330142308.png)

对于模式坍塌进一步的理解，如果空间上有一个较大的利普希茨系数，那么说明发生了模式坍塌。

![Pasted image 20250330144259](Pasted image 20250330144259.png)

##### Spectral Normalization

![Pasted image 20250330144556](Pasted image 20250330144556.png)

| 指标       | JS散度               | Wasserstein距离       |
|------------|----------------------|-----------------------|
| **连续性**  | 不连续（梯度消失）    | 连续                  |
| **对称性**  | 对称                 | 对称                  |
| **计算复杂度** | 低                  | 高（需约束判别器）     |
#### Conditional GAN

给定一个类来进行生成数据，这样可以生成不同类别的数据，这样可以生成更加多样的数据。并且可以在一定程度上避免模式坍塌。
除了接受高斯噪声还接受一个标签/图像等等作为输入。

#### ACGAN: Auxiliary Classifier GAN

是多任务学习的一种方法，除了生成数据，还可以进行分类。在生成数据的过程中还生成标签，所以可以一定程度上避免模式坍塌。

#### Cycle GAN

![Pasted image 20250330153629](Pasted image 20250330153629.png)

无配对数据下的图像转换（如马→斑马），通过循环一致性损失$Cyc$保证生成的图像在两个方向上的转换是一致的。
### Self-Attention GAN

将*Self-Attention*机制应用到GAN中，这样可以使得生成的数据更加真实。

#### Adaptive Instance Normalization

$$
AdaIN(u ,v) = \sigma(v) \left(\frac{u - \mu(u)}{\sigma(u)}\right) + \mu(v)
$$
本质上为重新着色的操作，将一个图像的风格转移到另一个图像上。

#### StyleGAN

通过控制风格来生成数据，这样可以生成更加多样的数据。

### VAE

#### Encoder

1. **推断潜在变量分布**  
    编码器将输入数据（如图像、文本）映射到潜在空间*latent space*，输出潜在变量$z$的概率分布参数（通常是高斯分布的均值和方差）。这一步称为**变分推断**，目的是找到输入数据在低维潜在空间中的概率表示。
2. **数据压缩与特征提取**  
    编码器将高维输入数据压缩到低维潜在变量$z$，提取数据的关键特征（如形状、颜色等抽象属性），同时去除冗余信息。
3. **引入不确定性**  
    不同于传统自编码器的确定性编码，VAE的编码器输出的是分布的参数，通过随机采样生成 zz，使得潜在空间具有连续性，便于生成新样本。

#### Decoder

1. **数据生成**  
    解码器从潜在变量$z$出发，重构输入数据$x$的分布（如像素值的伯努利分布或高斯分布），生成与原始数据相似的新样本。
2. **潜在空间映射到数据空间**  
    解码器学习如何将低维潜在变量$z$解码为高维数据空间中的样本，捕捉数据生成过程的规律（如像素间的依赖关系）。
3. **生成多样性**  
    由于潜在变量$z$是连续且概率化的，解码器可以在潜在空间中插值或随机采样，生成多样化且合理的新数据。


#### Why Variational

其核心目标是通过学习数据分布$p(x)$，生成与训练数据类似的新样本。具体来说，VAE旨在解决以下问题：
- **生成新数据**：例如生成图像、文本或音频。
- **学习潜在表示**：将高维数据映射到低维潜在空间，同时保持数据的语义特征。
- **概率建模**：显式定义数据的生成过程$p_{\theta} (x|z)$，并引入潜在变量$z$表示数据的隐含因素

在VAE中，我们引入一个潜在变量 $z$，假设数据 $x$ 是由某个先验分布 $p_\theta(z)$ 生成，然后通过条件分布 $p_\theta(x|z)$ 生成可观测数据 $x$。
$$
p_\theta (x) = \int p_\theta(x|z) p_\theta(z) \, \mathrm{d}z
$$
$z$的分布是一个高斯分布，对于$z$采样得到的实例，通过一个网络生成$x$。困难在于上面的反常积分，这是一个高维积分，不可行。可以使用*蒙特卡洛*。

对于后验分布：
$$
p_\theta(z|x) = \frac{p_\theta(x|z) p_\theta(z)}{p_\theta(x)}
$$
在积分中是经常使用的，但是计算是NP-hard的，因此引入**变分推断**，通过优化下界（ELBO）间接逼近。直接求后验往往是不可行的。因此，我们用**变分推断**的方式，去学习一个近似后验分布$q_{\phi}(z∣x)$，并用它来逼近真正的后验分布$p_{\theta}(z∣x)$，从而得到一个变分下界。

变分的意思为变量的替换，在概率中为分布率的替换。
对于这个问题，可以使用一个神经网络来实现，对于这种显式的分布，需要假设分布率，可以指定为高斯分布，用网络来学习均值和协方差，作为*Encoder Network*。
对于条件分布$p_\theta(x|z)$，也可以假定为高斯分布，这样可以计算均值和协方差，作为*Decoder Network*。

![Pasted image 20250330161028](Pasted image 20250330161028.png)

但是VAE是*Intractable*的，这个问题是NP-hard的。对于上面的条件概率，是比较困难的。所以采用一个近似的方法来进行求解。比如假设服从一个高斯分布，之后计算这个分布的均值和协方差矩阵。

#### ELBO: Evidence Lower Bound

训练的目的是使得$\log p_\theta (x_i)$尽可能大，但是很难计算，这里引入“参考系”，也就是引入一个近似的分布$q_\phi(z|x_i)$。

![Pasted image 20250330164549](Pasted image 20250330164549.png)

这里是将最大化似然的过程简化为最大化ELBO的过程。
$$
\log p_\theta (x_i) = \mathbb{E}_{q_\phi(z|x_i)} \left[ \log p_\theta(x_i|z) \right] - KL (q_\phi(z|x_i) || p_\theta(z))
$$

#### Reparameterization Trick

![Pasted image 20250403144444](Pasted image 20250403144444.png)

在VAE的训练过程中，通常需要从潜在变量的分布中采样。直接从分布中采样可能导致梯度无法传播到编码器网络，因此引入了重参数化技巧。
这是的采样变量是$\epsilon$，这样的话采样的过程就在计算图的旁路上，这样就可以进行梯度的传播。
#### VAE Inference

在训练的时候是从$x$中采样得到的，然后由后验分布$q_\phi(z|x)$来进行采样。而在推理过程中，是从潜在变量$z$中采样得到的，然后由条件分布$p_\theta(x|z)$来进行进行推理。
上述过程的合理性在于目标函数ELBO中的第二项：$KL (q_\phi(z|x_i) || p_\theta(z))$在训练中被最小化了。

![Pasted image 20250403152841](Pasted image 20250403152841.png)

#### VQ-VAE
VQ-VAE是对VAE的一个改进，使用了向量量化的方法来进行训练。通过对潜在变量进行离散化来进行训练，这样可以避免模式坍塌的问题。

### Diffusion Probabilistic Models

#### Denoising Diffusion Probabilistic Models

Diffusion模型是通过对数据进行逐步添加噪声来训练模型的。通过对数据进行逐步添加噪声，然后再通过一个网络来进行去噪声的操作。这样可以生成新的数据。

Markovian Process: 逐步添加噪声的过程，通常是一个高斯分布的过程。
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}  x_{t-1}, \beta_t I)
$$
联合分布：
$$
q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})
$$

重参数化表达：
$$
x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon \quad \epsilon \sim \mathcal{N}(0, I)
$$

##### Diffusion Kernel

$$
\begin{aligned}
x_t & = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon \\
&= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t} \epsilon + \sqrt{\alpha_t} \sqrt{1-\alpha_{t-1}} \epsilon' \\
& = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1 -  \alpha_t \alpha_{t-1}} \epsilon \\
& \dots \\
& = \sqrt{\overline{\alpha_t}} x_0 + \sqrt{1 - \overline{\alpha_t}} \epsilon
\end{aligned}
$$
- 其中 $\overline{\alpha_t} = \prod_{s=1}^{t} \alpha_s$

##### Generation by Denoising

- 首先对于$x_T$进行采样，得到一个高斯分布的样本。
- 然后利用后验分布得到$x_{t-1}$，使用Bayes公式：
  $x_{t-1} \sim q(x_{t-1}|x_t) \propto q(x_{t-1}) q(x_t |x_{t-1})$

采用一个变分方法来实现：
$$
p_{\theta}(x_{t-1}|x_t) = \mathcal{N} \left(x_{t-1}; \mu_{\theta}(x_t, t), \sigma_t^2 \mathbf{I} \right) 
$$
$$
p_\theta(x_{0:T}) = p_\theta (x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)
$$

上述过程可以使用一个网络来实现对于均值和方差的输出。

![Pasted image 20250403164408](Pasted image 20250403164408.png)

![Pasted image 20250403165731](Pasted image 20250403165731.png)

[What are Diffusion Models? | Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

![Pasted image 20250405094553](Pasted image 20250405094553.png)

##### Diffusion Parameters

![Pasted image 20250405161801](Pasted image 20250405161801.png)

##### Acceleration Strategies

##### Re-design forward Sampling

- Striding Sampling
	- 在每个时间步长中跳过多个时间步进行采样，从而减少采样次数。
	- 问题在于在相邻的时间戳中反向传播的后验分布可以近似为高斯分布，但是在较大的步长下不一定是这样的。
- DDIM: Denoising Diffusion Implicit Models
	- 通过设计一个新的前向采样过程来加速采样。
	- 通过引入一个新的参数$\alpha_t$来控制前向采样的过程。
	- 通过设计一个新的后验分布来进行采样。
	- 通过设计一个新的损失函数来进行训练。
	- ![Pasted image 20250405163042](Pasted image 20250405163042.png)
- Denosing Diffusion Models
	- ![Pasted image 20250405163350](Pasted image 20250405163350.png)
- Latent Diffusion Models
	- ![Pasted image 20250405163425](Pasted image 20250405163425.png)

##### Conditonal Diffusion Models


![Pasted image 20250405163914](Pasted image 20250405163914.png)

![Pasted image 20250405163941](Pasted image 20250405163941.png)

![Pasted image 20250405164110](Pasted image 20250405164110.png)
