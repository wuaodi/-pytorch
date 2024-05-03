"""torch.nn中的非线性激活层的使用 ReLU Sigmoid"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 建一个二维的tensor数据，reshape为（batch_size channel h w）
img_simu = torch.tensor([[1, -3],
                         [-5, 6]], dtype=torch.float32)
img_simu = torch.reshape(img_simu, [-1, 1, 2, 2])

# 输入CIFAR10的图片来看看
test_dataset = torchvision.datasets.CIFAR10('dataset', train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# 建一个ReLU的非线性激活层看一下数据改变
class My_relu(nn.Module):
    def __init__(self):
        super().__init__()
        # inplace参数指定是否对变量原位替换，默认为false
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, x):
        output = self.relu1(x)
        return output

# 建一个Sigmoid的非线性激活层看一下图片改变
class My_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output


# 实例化看一下ReLU
model_relu = My_relu()
print(model_relu(img_simu))

# 实例化看一下Sigmoid
model_sigmoid = My_sigmoid()
writer = SummaryWriter('logs')
for step, data in enumerate(test_dataloader):
    imgs, labels = data
    result = model_sigmoid(imgs)
    writer.add_images('imgs', imgs, step)
    writer.add_images('result', result, step)

writer.close()
