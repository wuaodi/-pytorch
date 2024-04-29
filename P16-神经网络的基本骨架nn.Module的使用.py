# Containers 神经网络的骨架，里面最常用的是Module的模块，所有的神经网络都要继承这个类
# 卷积层，池化层等等都是往骨架里填充的东西

# 定义模型，继承nn.Module，需要重写__init__和forward

import torch.nn as nn
import torch


# 写一个自己的神经网络模型， output = input + 1

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x + 1
        return y


model = MyModel()
a = torch.tensor(1.0)
output = model(a)
print(output)
