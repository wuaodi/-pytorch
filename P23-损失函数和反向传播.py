"""torch.nn中的Loss Function的使用"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
labels = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape BCHW
inputs = torch.reshape(inputs, [1, 1, 1, 3])
labels = torch.reshape(labels, [1, 1, 1, 3])

# L1Loss / MSELoss / 计算返回结果
l1loss = nn.L1Loss()
print('l1损失: ', l1loss(inputs, labels))
mseloss = nn.MSELoss()
print('mse损失: ', mseloss(inputs, labels))
# CrossEntropyLoss 假设x有三类
crossloss = nn.CrossEntropyLoss()
inputs = torch.tensor([0.2, 0.5, 0.8])
inputs = torch.reshape(inputs, [1, 3])  # shape: batch_size=1, class=3
labels = torch.tensor([2])  # shape: batch_size
print('交叉熵损失: ', crossloss(inputs, labels))

# 在神经网络中使用loss function
# backward是计算权重的梯度，可以使用debug模式来看一下
# 下一节课讲的优化器是根据梯度来更新权重
class My_model(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64,10))

    def forward(self, x):
        output = self.model(x)
        return output

# 加载数据
test_dataset = torchvision.datasets.CIFAR10('dataset', train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

model = My_model()
for data in test_dataloader:
    imgs, labels = data
    result = model(imgs)
    print(result.shape)
    print(labels.shape)
    loss = crossloss(result, labels)
    print('loss: ', loss)
    # 反向传播后可以在debug模式下 model-model-Protected Attributes-_modules-'0'-weight-grad 看到梯度值
    # 如果不反向传播则为None
    loss.backward()
    break
