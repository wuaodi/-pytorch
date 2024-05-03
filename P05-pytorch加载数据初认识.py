# pytorch加载数据主要涉及两个类

# 1、Dataset：提供一种方式来组织数据和label
## 应该具备以下功能：
## 如何获取每个数据和label
## 告诉我们一共有多少数据

# 2、Dataloader：为后面的网络提供不同的数据形式

from torch.utils.data import Dataset
help(Dataset)

# 所有的数据集构建要继承这个类，所有的子类需要重写__getitem__类，用于获取一个样本，也可以重写__len__
