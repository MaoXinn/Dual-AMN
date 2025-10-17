# Dual-AMN: 基于双重注意力匹配网络和归一化难样本挖掘的高效实体对齐

[![arXiv](https://img.shields.io/badge/arXiv-2103.15452-b31b1b.svg)](https://arxiv.org/pdf/2103.15452.pdf)

本项目是论文 **《Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining》** 的官方Keras实现。

## 简介

实体对齐（Entity Alignment）旨在识别不同知识图谱（Knowledge Graphs, KGs）中表示同一现实世界对象的实体。本项目提出的 **Dual-AMN** 模型，通过一种新颖的**双重注意力匹配网络**和**归一化难样本挖掘**策略，在不牺牲准确率的前提下，将实体对齐的训练速度提升了近10倍，并取得了业界领先（SOTA）的性能。

### 核心思想

*   **双重注意力网络 (Dual Attention Network)**: 传统模型通常混合处理实体的结构和关系信息。Dual-AMN创新地使用两个独立的图注意力编码器，分别对实体邻域结构和关系网络进行建模，然后将两者特征进行融合。这种设计使得模型可以捕获更丰富、更精细的实体表示。

*   **归一化难样本挖掘 (Normalized Hard Sample Mining)**: 为了加速模型的收敛并提升性能，我们设计了一种新颖的损失函数。它能够在训练过程中自动、无监督地识别出最易混淆的“难负样本”，并对其进行归一化加权，从而引导模型专注于学习最具挑战性的任务，极大提升了训练效率。

*   **迭代式半监督训练 (Iterative Semi-supervised Training)**: 模型采用自举（Bootstrapping）策略，在训练数轮后，利用已学到的知识在未标记数据中寻找高可信度的新实体对，并将其加入训练集，通过“学习-预测-再学习”的循环不断自我优化。

## 环境要求

*   Python 3.6
*   Keras 2.2.5
*   Tensorflow 1.14.0
*   Jupyter
*   Scipy
*   Numpy
*   tqdm
*   numba

## 数据集

本项目使用的数据集格式来源于 [GCN-Align](https://github.com/1049451037/GCN-Align), [JAPE](https://github.com/nju-websoft/JAPE), 和 [RSNs](https://github.com/nju-websoft/RSN)。

每个数据集文件夹包含以下文件：

*   `ent_ids_1`: 源知识图谱（KG1）的实体ID。
*   `ent_ids_2`: 目标知识图谱（KG2）的实体ID。
*   `ref_ent_ids`: 已知的跨图谱实体对齐链接（训练集）。
*   `triples_1`: 源知识图谱的三元组（头实体, 关系, 尾实体）。
*   `triples_2`: 目标知识图谱的三元组。

## 如何运行

1.  配置好上述 `环境要求` 中列出的依赖。
2.  在Jupyter环境中打开 `DualA.ipynb` 文件。
3.  在Notebook的第二个代码单元格中，修改 `load_data` 函数的路径以选择不同的数据集，例如 `data/en_fr_15k_V1/`。
4.  按顺序执行Notebook中的所有代码单元格即可开始训练和评估。

## 致谢

本项目的代码实现参考了以下优秀的开源项目，感谢他们的贡献！

*   [keras-gat](https://github.com/danielegrattarola/keras-gat)
*   [GCN-Align](https://github.com/1049451037/GCN-Align)