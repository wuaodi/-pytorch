"""利用GPU训练方法1"""
# 网络模型、数据（输入、标注）、损失函数 对这三个调用.cuda()

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# 准备数据集，训练和测试
train_data = torchvision.datasets.CIFAR10('dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10('dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_num = len(train_data)
test_num = len(test_data)
print('训练集图像数量为：{}'.format(train_num))
print('测试集图像数量为：{}'.format(test_num))

# 利用DataLoader加载数据集, batch64
train_loader = DataLoader(train_data, 64, shuffle=True)
test_loader = DataLoader(test_data, 64, shuffle=False)


# 搭建神经网路
class My_model(nn.Module):
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
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 实例化神经网络
model = My_model()
if torch.cuda.is_available():
    model = model.cuda()  # 将模型放到GPU

# 损失函数定义
crossloss = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    crossloss = crossloss.cuda()  # 将损失函数放到GPU

# 优化器定义
learning_rate = 0.01
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 初始化tensorboard日志
writer = SummaryWriter('logs')

# 循环训练与测试
# 网络训练
for epoch in range(10):
    model.train()
    for step, data in enumerate(train_loader):
        imgs, labels = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()  # 将数据放到GPU
            labels = labels.cuda()  # 将数据放到GPU
        results = model(imgs)
        loss = crossloss(results, labels)
        model.zero_grad()
        loss.backward()
        optim.step()
        if step % 100 == 0:
            print('第{}轮 {}/{}迭代 loss: {:.6f}'.format(epoch, step, int(train_num / 64), loss.item()))
    writer.add_scalar('training loss', loss.item(), epoch)

    # 网络测试
    model.eval()
    test_loss = 0.0
    acc_num = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()  # 将数据放到GPU
                labels = labels.cuda()  # 将数据放到GPU
            results = model(imgs)
            loss = crossloss(results, labels)
            test_loss += loss
            acc_num += (results.argmax(1) == labels).sum()
        print('----第{}轮结束 精度:{:.4f} 测试loss:{:.6f}----'.format(epoch, acc_num.item()/len(test_data), test_loss))
        writer.add_scalar('testing loss', loss.item(), epoch)
        writer.add_scalar('testing acc', acc_num.item()/len(test_data), epoch)

    # 保存每一轮的结果，保存模型和参数
    torch.save(model, 'runs/epoch{}.pth'.format(epoch))
