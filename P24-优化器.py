"""torch.optim中的优化器的使用"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


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

# 实例化模型
model = My_model()
# 损失函数
crossloss = nn.CrossEntropyLoss()
# 优化器 传入要更新的参数，学习率等
optim = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    loss_sum = 0
    for data in test_dataloader:
        imgs, labels = data
        result = model(imgs)
        # 计算损失，梯度清零，反向传播，优化参数
        loss = crossloss(result, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_sum = loss_sum + loss
    # 打印结果
    print('loss of epoch{} is {}'.format(epoch, loss_sum))
