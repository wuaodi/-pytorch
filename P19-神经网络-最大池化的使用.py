"""torch.nn中的MaxPool2d的使用"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 写一个5*5的tensor，模拟单通道图像，转为4维，注意类型为float32
img_simu = torch.tensor([[1, 2, 0, 3, 1],
                         [0, 1, 2, 3, 1],
                         [1, 2, 1, 0, 0],
                         [5, 2, 3, 1, 1],
                         [2, 1, 0, 1, 1]], dtype=torch.float32)
img_simu = torch.reshape(img_simu, [1, 1, 5, 5])


# 输入CIFAR10的图片来看看
test_dataset = torchvision.datasets.CIFAR10('dataset', train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# 写模型，最大池化层
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        # ceil=True允许有出界部分，默认为False，stride默认为kernel_size
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        y = self.maxpool1(x)
        return y


# 实例化，看输出
model = Mymodel()
result = model(img_simu)
print(result)
print(result.shape)

writer = SummaryWriter('logs')
for step, data in enumerate(test_dataloader):
    imgs, labels = data
    result = model(imgs)
    writer.add_images('imgs', imgs, step)
    writer.add_images('result', result, step)

writer.close()


