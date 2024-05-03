"""torch.nn中的Linear全连接层以及其他层的使用"""
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

test_dataset = torchvision.datasets.CIFAR10('dataset', train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# 写线性层
class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 10)

    def forward(self, x):
        output = self.linear1(x)
        return output


model = My_model()
for step, data in enumerate(test_dataloader):
    imgs, labels = data
    print('原始维度BCHW：', imgs.shape)
    # torch.flatten 的写法
    imgs_flatten = torch.flatten(imgs, start_dim=1, end_dim=-1)
    print('flatten后维度：', imgs_flatten.shape)
    # torch.nn.Flatten 的写法，这个默认start_dim=1所以不用管batch_size
    faltten_layer = torch.nn.Flatten()
    imgs_nnflatten = faltten_layer(imgs)
    print('nn flatten后维度：', imgs_nnflatten.shape)
    # reshape的写法
    imgs_reshape = torch.reshape(imgs, [4, -1])
    print('reshape后维度：', imgs_reshape.shape)
    # 经过线性层
    result = model(imgs_nnflatten)
    print('经过线性层后维度：', result.shape)
    break
