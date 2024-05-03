"""torch.nn中的Conv2d的使用"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR10测试集，batch_size为64
test_set = torchvision.datasets.CIFAR10('dataset', train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_dataloader = DataLoader(test_set, 64, drop_last=True)


# 搭建网络，使用nn.Conv2d，单层，输出通道为6
class Mymodel(nn.Module):
    def __init__(self):
        """这里写网络层"""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        """这里写层间的计算"""
        y = self.conv1(x)
        return y


# 初始化网络，对数据进行前向推理，把图片和结果显示一下
model = Mymodel()
print('model: ', model)
writer = SummaryWriter('logs')
for step, data in enumerate(test_dataloader):
    imgs, labels = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images('input', imgs, step)
    output = torch.reshape(output, [-1, 3, 30, 30])
    writer.add_images('output reshape', output, step)
writer.close()
