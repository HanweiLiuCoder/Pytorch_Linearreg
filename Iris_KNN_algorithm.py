#虽然PyTorch本身没有提供K近邻算法的实现，但可以使用PyTorch的张量操作和计算图特性来实现K近邻算法。
# https://zhuanlan.zhihu.com/p/341572059
#https://github.com/ZhiHuDaShiJie/Fundamentals-of-Machine-Learning/blob/main/README.md

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch的张量
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()

# 计算测试集样本与训练集样本的欧氏距离
distances = torch.cdist(X_test, X_train)

# 预测测试集样本的类别
k = 3  # 设置K值
_, indices = torch.topk(distances, k, largest=False, sorted=True)
y_pred = torch.mode(y_train[indices], dim=1).values

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
