"""torchvision.models中的vgg16的使用"""
import torchvision
from torch import nn

# 加载vgg16网络，这里相当于实例化，可以看实现也是继承的nn.Module
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)

# 打印一下网络架构看看
print(vgg16)

# 加载CIFAR10数据
train_data = torchvision.datasets.CIFAR10('dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

# 使用.add_module，增加一层，输入为1000 输出为10
vgg16.classifier.add_module('add_layer', nn.Linear(in_features=1000, out_features=10))
print(vgg16)

# 修改最后一层，直接引用替换
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
