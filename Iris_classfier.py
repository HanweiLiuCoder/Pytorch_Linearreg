
import torch
import torch.nn.functional as F
from sklearn import datasets
from sklearn import preprocessing
from torch import nn
from torch import optim
import matplotlib.pyplot as plt


class IrisClassifierNet(nn.Module):
    """
    神经网络模型
    """

    def __init__(self, *, in_features: int, hidden_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        # 隐含层
        self.hidden = nn.Linear(in_features=in_features, out_features=hidden_features, dtype=dtype)
        # 输出层
        self.out = nn.Linear(in_features=hidden_features, out_features=out_features, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (预测输出)
        """
        x = self.hidden(inputs)             # 数据传输到隐含层
        outputs = self.out(F.relu(x))       # 应用 ReLU 激活函数, 增加非线性拟合能力, 然后传输到输出层
        return F.softmax(outputs, dim=1)    # 把输出应用 softmax 激活函数 (把各类别输出值映射为对应的概率)


def main():
    # 加载鸢尾花数据集, 获取样本的 输入 和 输出
    iris_dataset = datasets.load_iris()
    x_2d, y_1d = iris_dataset.data, iris_dataset.target
    print(x_2d.shape)               # (150, 4)      一共150个样本, 每个样本4个特征
    print(y_1d.shape)               # (150,)        样本对应的输出类别, 元素值 0、1 或 2

    # 对输入特种数据做标准化处理(缩放为 单位方差, 0均值 的标准数据)
    scaler = preprocessing.StandardScaler()
    x_2d = scaler.fit_transform(x_2d)

    # 样本数据转换为 Tensor
    inputs = torch.tensor(x_2d, dtype=torch.float32)
    # 交叉熵损失函数的第二个参数(即输出的目标类别标签targets)的类型必须是 int64 类型, 且形状必须为 0D 或 1D
    targets = torch.tensor(y_1d, dtype=torch.int64)

    print(inputs.dtype, inputs.shape)       # torch.float32 torch.Size([150, 4])
    print(targets.dtype, targets.shape)     # torch.int64 torch.Size([150])

    # 创建 网络模型, 4个输入特征, 3个输出类别 (模型的 dtype 必须和输入输出数据的 dtype 相同)
    model = IrisClassifierNet(in_features=4, hidden_features=3, out_features=3, dtype=torch.float32)
    # 创建 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    # 创建 损失函数
    criterion = nn.CrossEntropyLoss()

    epochs = 2000

    for epoch in range(epochs):
        # 1. 前向传播, 预测输出
        outputs = model(inputs)

        # 2. 计算损失值
        loss = criterion(outputs, targets)

        # 3. 梯度清零
        optimizer.zero_grad()

        # 4. 误差反向传播, 计算梯度并累加
        loss.backward()

        # 5. 更新模型参数
        optimizer.step()

        # 每隔一定次数输出准确率
        if (epoch % 200 == 0) or (epoch == epochs - 1):
            # 预测的输出标签类别 (每行取概率最大的索引为标签类别值)
            output_labels = torch.max(outputs, dim=1)[1]

            # 预测准确的数量 (相同位置的类别值相等则为True, 不能则为False, True是1, False是0, 全部结果相加就是正确预测的数量)
            accurate_count = torch.sum(output_labels == targets)

            # 计算准确率
            accuracy = accurate_count / output_labels.shape[0]
            print(f"Epoch[{epoch:04d}/{epochs - 1}]: loss={loss}, accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
