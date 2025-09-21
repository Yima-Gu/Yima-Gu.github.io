---
title: Deep Learning Lecture-8
date: 2025-06-21
tags:
  - DeepLearning
math: true
---

## Transfer Learning

模型在较大规模中的数据集中得到成功，在较小的数据集中进进行学习。**利用先验知识**是实现强人工智能的一个前提条件。

## Pre-Training

预训练的结果是一个模型，然后在小数据集上进行微调。预训练提供的是一个先验知识，然后再进行微调。这已经是深度学习中的共识，在小数据集中做微调是一个非常好的方法。

### Supervised Pre-Training

- BiT: Big Transfer
	- 在大规模数据集上做预训练非常重要
- DAT: Domain Adaptative Transfer
	- 在预训练的数据集上进行筛选对于模型的表现会更好

- 迁移性受到数据间距离的影响
- 在微调的过程中需要对很多层进行调整
- 下游的任务和预训练的任务之间的距离越近，迁移的效果越好

假设空间的解释：
- 预训练的模型能在较小的较好的假设空间上寻找

#### Multi-Task Architecture

- Hard Parameter Sharing
	- 直接共享参数
	- 共享的参数是所有任务都需要的
- Soft Parameter Sharing
	- 每个任务使用一个模型
- TASKONOMY
	- 首先进行训练，之后构建任务的关系图，在训练的过程中找一些更相近的任务进行共享

多任务学习的损失函数：
$$
\min_{\theta} \sum_{i=1}^{T} w_i \mathcal{L}_i(\theta)
$$
其中 $w_i$ 是每个任务的权重
- GradNorm：使得每个任务的梯度相同
- Task uncertainty：每个任务的损失函数的权重是不同的
- Pareto optimal solution：
- Optimize for the worst task
- Regularization：