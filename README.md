# 图机器学习课程作业

基于dgl和pytorch实现同质图节点分类和图分类任务，基于openhgnn实现异质图节点分类任务。

## 实验环境
- system: ubuntu, NVIDIA gpu
- python 3.8
- pytorch 1.10.0
- dgl 0.8.1
## 同质图节点分类

- 数据集：[DGL CoraGraphDataset]( https://docs.dgl.ai/generated/dgl.data.CoraGraphDataset.html)
- 模型：[[GRAND] Graph Random Neural Network for Semi-Supervised Learning on Graphs]( https://arxiv.org/abs/2005.11079)
- 使用代码：[DGL GRAND Example]( https://github.com/dmlc/dgl/tree/master/examples/pytorch/grand)

### 模型介绍
GRAND模型是为了解决GNN的过平滑和鲁棒性问题。为了对图进行数据增强：提出在 GRAND 中进行随机传播，其中每个节点的特征可以部分或全部被随机删除(dropout)，然后受扰动的特征矩阵在图中传播。因此，每个节点都可以对特定的邻域不敏感，从而增加了 GRAND 的健壮性。
此外，随机传播的设计可以自然地分离特征传播和变换，在大多数 gnn 中这两者通常是相互耦合的。这使得 GRAND 能够安全地执行高阶特性传播，而不会增加复杂性，从而降低了 GRAND 的过平滑风险。
更重要的是，随机传播使每个节点能够将消息随机传递到其邻居。在图数据相同的假设下，我们可以随机地为每个节点生成不同的扩展表示。
然后利用一致性正则化(consistency regularization)方法来加强预测模型，例如，一个简单的多层感知模型(MLP) ，对同一个未标记数据的不同增强数据输出相似的预测，改善了 GRAND 在半监督环境下的泛化行为。

### 运行
```bash
cd node_classification/grand
python main.py --dataname cora --gpu 0 --lam 1.0 --tem 0.5 --order 8 --sample 4 --input_droprate 0.5 --hidden_droprate 0.5 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 100 --lr 1e-2  --epochs 2000
```

### 实验结果

- 表现：F1_macro: 0.8402 F1_micro: 0.8520
- 运行日志：[Log]( ./node_classification/grand/log)

## 图分类

- 数据集：[DGL GINDataset MUTAG]( https://docs.dgl.ai/generated/dgl.data.GINDataset.html)
- 模型：[[GIN Graph Isomorphism Network] How Powerful are Graph Neural Networks]( https://arxiv.org/abs/1810.00826)
- 使用代码：[DGL GIN Example]( https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin)

### 模型介绍

图同构网络（Graph Isomorphism Network ，GIN）模型源于一篇原理性论文《How Powerful are Graph Neural Networks？》。该论文分提出了一个能够更好表达图特征的结构 GIN。
图同构网络（GIN）模型对图神经网络提出了一个更高的合理性要求——同构性。即对同构图处理后的图特征应该相同，对非同构图处理后的图特征应该不一样。

图同构网络（GIN）模型是从图神经网络的单射函数特性设计出来的。 GIN模型在图节点邻居特征的每一跳聚合操做以后，又与自身的原始特征混合起来。并在最后使用能够拟合任意规则的全链接网络进行处理，使其具备单射特性。
在特征混合的过程当中，引入了一个可学习参数对自身特征进行调节，并将调节后的特征与聚合后的邻居特征进行相加。

实验中，邻居池化和图池化层都使用了sum。

### 运行
```bash
cd graph_classification/gin/
python main.py --dataset MUTAG --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum --filename MUTAG.txt
```
### 实验结果

- 表现：10-fold accuracy 89%
- 运行日志：[Log]( ./graph_classification/gin/log)

## 异质图节点分类

- 数据集：[OpenHGNN GTNDataset ACM]( https://openhgnn.readthedocs.io/en/latest/_modules/openhgnn/dataset/gtn_dataset.html#ACM4GTNDataset)
- 模型：[[GTN] Graph Transformer Networks](https://arxiv.org/abs/1911.06455)
- 使用代码：[OpenHGNN]( https://github.com/BUPT-GAMMA/OpenHGNN)

### 运行
```bash
git clone https://github.com/BUPT-GAMMA/OpenHGNN
cd OpenHGNN
python main.py -m fastGTN -t node_classification -d acm4GTN --use_best_config -g 0
```

### GTN模型介绍

GTN要点主要有3个：
GTN可生成新的图结构，如识别出原图中无相连但实际有潜在用处的边。
Graph Transformer Layer基于边类型的软选择和组合，可以生成各种meta-paths。
GTN中meta-path自动生成且不依赖领域知识，它比许多预定义meta-path类算法效果好。

### 实验结果

- 表现：F1_macro: 0.9292 F1_micro: 0.9285
- 运行日志：[Log]( ./hetero_node_classification/GTN/log )
