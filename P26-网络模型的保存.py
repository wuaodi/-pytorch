"""模型的两种保存方式： 保存模型和权重；只保存权重"""

import torchvision
import torch
from torch import nn

# 加载vgg16
vgg16 = torchvision.models.vgg16(pretrained=True)

# 方式1： torch.save(模型，路径) 保存模型的结构和权重文件
torch.save(vgg16, 'method1.pth')

# 方式2： torch.save(模型的状态字典，路径) 保存的是模型的权重文件
# 官方推荐的，比较小（对于小网络实际差别不大）
torch.save(vgg16.state_dict(), 'method2.pth')


# 方式1的陷阱
# 如果是自己定义的模型，需要在加载的时候保证代码中有定义的模型的类，不需要实现
class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.layer1(x)
        return x


my_model = My_model()
torch.save(my_model, 'my_model.pth')
