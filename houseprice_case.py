import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn

#我们需要定我们的模型。在PyTorch中，我们可以通过继承torch.nn.Module类来定义我们的模型，并实现forward方法来定义前向传播。
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出的维度都是1

    def forward(self, x):
        out = self.linear(x)
        return out

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #创建模型实例
    model = LinearRegressionModel()

    #设置损失函数：均方误差（最标准的误差函数）
    criterion = nn.MSELoss()

    #使用随机梯度下降作为优化器。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #开始训练
    # 转换为 PyTorch 张量

    #准备训练数据 方法1
    # 房屋面积
    areas = np.array([20,30,40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,150,160,170,180,190], dtype=float)
    # 房价
    prices = np.array([120,145,160,180, 300, 360, 420, 460, 490, 550, 660, 720, 780, 840,880,890,950,1000], dtype=float)


    #准备训练数据， 方法2： 利用随机函数生成一些线性回归的训练数据
    # 设置随机种子，以便结果可复现
    np.random.seed(0)
    # 设置线性回归模型的参数
    true_slope = 2.5
    true_intercept = 10
    # 生成随机的x值
    areas = np.linspace(0, 10, 1000)
    # 生成随机的噪声
    noise = np.random.normal(loc=0, scale=3, size=1000)
    # 根据线性回归模型生成对应的y值
    prices = true_slope * areas + true_intercept + noise

    #print(areas.size, prices.size)
    #stop()

    # 数据规范化
    areas = (areas - np.mean(areas)) / np.std(areas)
    prices = (prices - np.mean(prices)) / np.std(prices)

    inputs = torch.from_numpy(areas)
    targets = torch.from_numpy(prices)

    #显示数据
    #plt.scatter(inputs,targets)
    #plt.show()

    #数据类型统一，否则报错： mat1 and mat2 must have the same dtype
    inputs = inputs.to(torch.float32)
    targets = targets.to(torch.float32)

    # 转换为二维张量
    inputs = inputs.view(-1, 1)
    targets = targets.view(-1, 1)

    # 进行 60 轮训练
    for epoch in range(100):
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 60, loss.item()))

    #模型评估
    #计算平均损失
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 不需要计算梯度
        predictions = model(inputs)
        loss = criterion(predictions, targets)
    print('Final Loss:', loss.item())


    #进行预测计算：
    # 预测一个 100 平方米的房子的价格
    area = torch.tensor([100.0])
    area = (area - torch.mean(inputs)) / torch.std(inputs)  # 需要进行同样的数据规范化
    price = model(area)
    print('面积100的房间价格 Predicted price:', price.item())

    #可视化显示
    model.eval()
    predict = model(Variable(inputs))
    predict = predict.data.numpy()
    plt.plot(inputs.numpy(), targets.numpy(), 'ro', label='Original data 原始数据')
    plt.plot(inputs.numpy(), predict, label='Fitting Line 回归线')
    plt.show()