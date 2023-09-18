# 逻辑回归是二元分类，属于多元分类的一种特殊情况。
# 多元分类与二元分类类似，区别在于使用 softmax 函数替代 sigmoid 函数作为激活函数。
# 如果分类的类别数为 n，则 softmax 函数接收 n 个输入，然后输出 n 个概率（概率之和为 1），
# 概率最大的类别就是预测的类别。
# 多元分类问题的损失函数一般也是使用 CrossEntropyLoss(input, target) 交叉熵损失函数。

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#涉及神經網絡相關知識，難度較高

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


class ClassifierNet(nn.Module):
    """
    多元分类神经网络模型
    """

    def __init__(self, *, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        # 隐含层
        self.hidden = nn.Linear(in_features, hidden_features)
        # 输出层
        self.out = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (预测输出)
        """
        # 输入层样本特征 输入 隐含层, 隐含层输出应用 ReLU 激活函数 (增加拟合非线性模型的能力)
        x = F.relu(self.hidden(x))
        # 输出层输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)
        x = F.softmax(self.out(x))
        return x


def generate_samples() -> Tuple[np.ndarray, np.ndarray]:
    """
    生成样本, 样本有 2 个输入特征, 输出 3 种可可能的类别 (分别记作: 0, 1, 2)
    """
    # 一种类别的数量
    cluster_size = 100
    # 每个样本有 2 个输入特征, cluster 的形状为 (cluster_size, 2)
    cluster = torch.ones((cluster_size, 2))

    # 生成具有指定均值和标准差的一组数据 (批量样本的输入特征), data0 的形状为 (cluster_size, 2), 表示 cluster_size 个 二维坐标点 (cluster_size 个样本的输入特征)
    data0 = torch.normal(-4 * cluster, 3)
    # data0 样本批数据对应的输出标签类别为 0, label0 的形状为 (cluster_size, 1), 表示 cluster_size 个样本的输出标签类别
    label0 = torch.zeros((cluster_size, 1))

    # 同样的方法生成第二批样本, 该批样本是输出标签类别为 1
    data1 = torch.normal(4 * cluster, 3)
    label1 = torch.ones((cluster_size, 1))

    # 同样的方法生成第三批样本, 该批样本是输出标签类别为 2
    data2 = torch.normal(12 * cluster, 3)
    label2 = label1 * 2

    # 合并三批样本, 合并后 inputs 的形状为 (3*cluster_size, 2), outputs 的形状为 (3*cluster_size, 1),
    # 表示有 3*cluster_size 个样本, 每个样本有 2 个输入特征, 1 个输出特征(0/1标签类别)
    inputs_2d = torch.cat((data0, data1, data2), dim=0).type(torch.FloatTensor)
    targets_2d = torch.cat((label0, label1, label2), dim=0).type(torch.LongTensor)

    return inputs_2d, targets_2d


def main():
    # 生成样本
    inputs, targets = generate_samples()

    # 绘制样本: 把样本数据绘制为散点图
    x_1d = inputs.data.numpy()[:, 0]        # 第 1 个输入特征作为 X 轴
    y_1d = inputs.data.numpy()[:, 1]        # 第 2 个输入特征作为 Y 轴
    labels_1d = targets.data.numpy()[:, 0]  # 输出标签类别
    # 绘制散点图, 在 (x, y) 坐标处绘制圆点, 样本坐标对应的标签类别用颜色来体现。
    plt.scatter(x_1d, y_1d, s=20, c=labels_1d, cmap="tab10")

    # 创建 网络模型, out_features 表示输出类别可能的数量
    model = ClassifierNet(in_features=inputs.shape[1], hidden_features=20, out_features=3)
    # 创建 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    # 创建 损失函数, 分类问题一般使用 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 总的迭代次数
    epochs = 1000

    # CrossEntropyLoss 交叉熵函数的 目标张量 只支持 0D 或 1D, 转换后的形状为 1D
    targets = targets.reshape(-1)

    # 训练模型
    for epoch in range(epochs):
        # 1. 前向传播 (预测输出)
        outputs = model(inputs)

        # 2. 计算损失值, outputs 的形状为 (samples_count, 3), targets 形状为 (samples_count,)
        loss = criterion(outputs, targets)

        # 3. 梯度清零 (清空 model 参数的梯度值, 即 grad 属性, 不清空会累积)
        optimizer.zero_grad()

        # 4.反向传播 (计算梯度, 计算 model 参数的梯度值)
        loss.backward()

        # 5. 更新模型参数 (根据 model 参数的梯度值 更新 参数值)
        optimizer.step()

        # 输出准确度: 每隔一定次数 或 最后一次 输出准确度
        if (epoch % 100 == 0) or (epoch == epochs - 1):
            # outputs 的形状为 (samples_count, 3), 沿 dim=1 轴计算最大值 (计算每一行的最大值),
            # 返回一个元祖 tuple(max_values_tensor, max_values_indexes_tensor), 元祖元素形状为 (samples_count,)
            # max_values_tensor 和 max_values_indexes_tensor 的每一个元素表示原矩阵每一行的最大值的 值 和 所在行的列索引
            max_tensors = torch.max(outputs, dim=1)

            # output_labels 张量表示原矩阵每一行的最大值的所在行的列索引, 形状为 (samples_count,)
            # 如果第 0 列比较大, 则值为 0, 即这一行样本预测输出的标签类别为0
            # 如果第 1 列比较大, 则值为 1, 即这一行样本预测输出的标签类别为1
            # 如果第 2 列比较大, 则值为 2, 即这一行样本预测输出的标签类别为2
            output_labels = max_tensors[1]

            # 和真实输出标签对比, 计算出预测准确的样本数量
            accurate_count = np.sum(output_labels.data.numpy() == targets.data.numpy())

            # 计算准确率
            accuracy = accurate_count / output_labels.shape[0]
            print(f"Epoch[{epoch:03d}/{epochs - 1}]: loss={loss}, accuracy={accuracy:.3f}")

    plt.show()


if __name__ == "__main__":
    main()
