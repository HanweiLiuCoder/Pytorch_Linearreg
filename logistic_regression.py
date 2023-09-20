# 1. 逻辑回归
# 线性回归 和 非线性回归 的输出都是连续的。
# 逻辑回归 与 线性回归/非线性回归 区别在于 逻辑回归 的输出是 二元离散 的，即输出特征只有两种结果。
# 因此逻辑回归通常用于二元分类问题，即把一堆样本分为两类。为了方便数学表达和计算，
# 一般把逻辑回归分类的两类输出结果分别记做整数 0 和 1。

# 逻辑回归的输出是两种类别，记作 0 和 1，可以用普通网络层组合一个激活函数，
# 让每个样本输出两个 0.0 ~ 1.0 之间的值分别表示两种类别的概率（概率之和为 1），
# 概率大的就作为样本的预测类别。
#
# sigmoid() 表示 S型函数，公式为 sigmoid(x) = 1 / (1 + e^-x)，函数值域为 (0, 1)，
# 即将一组变量映射到 0.0 ~ 1.0 之间的一个数值（一组变量对应的输出的和等于 1）。
# sigmoid() 函数常用于二分类问题。
#
# 分类问题的损失函数一般使用 CrossEntropyLoss(input, target) 交叉熵损失函数，
# 函数的 input 参数一般是二维矩阵，每一行表示预测样本对应的各类别的概率值，
# target 参数是样本的真实类别值（形状必须是 0D 或 1D）。

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

class LogisticRegression(nn.Module):
    """
    逻辑回归神经网络模型
    """

    def __init__(self, *, in_features: int, out_features: int):
        super().__init__()
        # 只使用一层线型模型
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (预测输出)
        """
        # 经过线型模型前向传播
        x = self.linear(inputs)

        # 线型模型的输出应用激活函数, 输出张量的形状为 (inputs.shape[0], out_features), 是一个二维矩阵,
        # 每一行表示一个样本的输出, 每个样本输出 out_features 个值在 (0.0, 1.0) 之间的参数, 分别表示 out_features 个标签类别对应的概率。
        outputs = torch.sigmoid(x)
        return outputs

def main():
    #
    # 生成样本
    #

    # 样本总数的一半
    half_samples = 100
    # 每个样本有 2 个输入特征, cluster 的形状为 (100, 2)
    cluster = torch.ones((half_samples, 2))

    # 生成具有指定均值和标准差的一组数据 (批量样本的输入特征), data0 的形状为 (100, 2), 表示 100 个 二维坐标点 (100 个样本的输入特征)
    data0 = torch.normal(-4 * cluster, 3)
    # data0 样本批数据对应的输出标签类别为 0, label0 的形状为 (100, 1), 表示 100 个样本的输出标签类别
    label0 = torch.zeros((half_samples, 1))

    # 同样的方法生成第二批样本, 该批样本是输出标签类别为 1
    data1 = torch.normal(4 * cluster, 3)
    label1 = torch.ones((half_samples, 1))

    # 合并两批样本, 合并后 inputs 的形状为 (200, 2), targets 的形状为 (200, 1),
    # 表示有 200 个样本, 每个样本有 2 个输入特征, 1 个输出特征(0/1标签类别)
    inputs = torch.cat((data0, data1), dim=0).type(torch.FloatTensor)
    targets = torch.cat((label0, label1), dim=0).type(torch.LongTensor)

    #
    # 绘制样本
    #

    # 把样本数据绘制为散点图
    x = inputs.data.numpy()[:, 0]           # 第 1 个输入特征作为 X 轴, 形状为 (200,)
    y = inputs.data.numpy()[:, 1]           # 第 2 个输入特征作为 Y 轴, 形状为 (200,)
    labels = targets.data.numpy()[:, 0]     # 输出标签类别, 形状为 (200,)

    # 绘制散点图, 在 (x, y) 坐标处绘制圆点, 样本坐标对应的标签类别用颜色来体现。
    # s=16 表示圆点的大小的 2 次方 (s = point_size ** 2)
    # c=labels 表示圆点颜色值, 这里把标签类别(0/1)传给 c (也可以使用固定的 RGB 值)
    # cmap="bwr" 颜色值映射, 颜色值 c 这里传的是标签类别值, cmap 负责把 c (标签类别值) 映射为具体的 RGB 颜色值, 这里 0映射为蓝色, 1映射为红色
    plt.scatter(x, y, s=16, c=labels, cmap="bwr")

    # cmap 可取值参考: https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html

    #
    # 创建/训练网络模型
    #

    # 创建 神经网络模型, 样本输入特征数为 2, 输出特征数为 2 (因为样本有 2 个输出标签类别, 每个标签类别输出为一个概率, 所以输出特征数为 2)
    model = LogisticRegression(in_features=2, out_features=2)
    # 创建 优化器, 使用随机梯度下降法, 学习率为 0.02
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    # 创建 损失函数, 分类问题一般使用 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 总的迭代次数
    epochs = 100

    # CrossEntropyLoss 交叉熵函数的 目标张量 只支持 0D 或 1D, 转换后的形状为 (200,)
    targets = targets.reshape(-1)

    # 训练模型
    for epoch in range(epochs):
        # 1. 前向传播 (预测输出)
        outputs = model(inputs)

        # 2. 计算损失值, outputs 的形状为 (200, 2), targets 形状为 (200,)
        loss = criterion(outputs, targets)

        # 3. 梯度清零 (清空 model 参数的梯度值, 即 grad 属性, 不清空会累积)
        optimizer.zero_grad()

        # 4.反向传播 (计算梯度, 计算 model 参数的梯度值)
        loss.backward()

        # 5. 更新模型参数 (根据 model 参数的梯度值 更新 参数值)
        optimizer.step()

        #
        # 输出准确度
        #

        # 每隔 10 次 或 最后一次 输出准确度
        if (epoch % 10 == 0) or (epoch == epochs - 1):
            # outputs 的形状为 (200, 2), 沿 dim=1 轴计算最大值 (计算每一行的最大值),
            # 返回一个元祖 tuple(max_values_tensor, max_values_indexes_tensor), 元祖元素形状为 (200,)
            # max_values_tensor 和 max_values_indexes_tensor 的每一个元素表示原矩阵每一行的最大值的 值 和 所在行的列索引
            max_tensors = torch.max(outputs, dim=1)

            # output_labels 张量表示原矩阵每一行的最大值的所在行的列索引, 形状为 (200,)
            # 如果第 0 列比较大, 则值为 0, 即这一行样本预测输出的标签类别为0
            # 如果第 1 列比较大, 则值为 1, 即这一行样本预测输出的标签类别为1
            output_labels = max_tensors[1]

            # 和真实输出标签对比, 计算出预测准确的样本数量。
            # 下面 == 两边的值均为相同形状的 ndarray, == 的计算结果为 bool 类型的矩阵, 结果为 True 表示预测准确, False 表示预测错误,
            # 用于数值计算时 True == 1, False == 0, bool矩阵的元素值累加和即为 True 的数量。
            accurate_count = np.sum(output_labels.data.numpy() == targets.data.numpy())

            # 计算准确率
            accuracy = accurate_count / (2 * half_samples)
            print(f"Epoch[{epoch:02d}/{epochs - 1}]: loss={loss}, accuracy={accuracy}")

    plt.show()


if __name__ == "__main__":
    main()
