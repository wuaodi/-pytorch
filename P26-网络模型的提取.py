"""加载保存好的torch模型"""
import torch
import torchvision
from torch import nn

# 方式1保存的提取: torch.load('.pth文件')
vgg16 = torch.load('method1.pth')
# print(vgg16)

# 方式2保存的提取: 加载权重字典，构建网络，网络加载权重字典
weights = torch.load('method2.pth')
# print(weights)  # 打印出来的就是各层的weights和bias字典
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(weights)
# print(vgg16)


# 方式1的陷阱
# 没办法直接加载，需要导入定义模型结构的类，注意代码文件名称只能英文
# 或者直接复制过来，不需要实现
class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.layer1(x)
        return x


my_model = torch.load('my_model.pth')
print(my_model)
