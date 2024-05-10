"""完整的模型训练套路写法"""
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from P27_model import My_model

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

# 搭建神经网路，单独当到一个文件中，可以自己验证一下尺寸
# 实例化神经网络
model = My_model()

# 损失函数定义
crossloss = nn.CrossEntropyLoss()

# 优化器定义
learning_rate = 0.01
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 初始化tensorboard日志
writer = SummaryWriter('logs')

# 循环训练与测试
# 网络训练
# model.train()，只对特定的一些层有作用，比如drop out，训练和测试时不一样
# 每隔100迭代记录训练次数的为多少的时候，训练的loss是多少，loss.item()
for epoch in range(10):
    model.train()
    for step, data in enumerate(train_loader):
        imgs, labels = data
        results = model(imgs)
        loss = crossloss(results, labels)
        model.zero_grad()
        loss.backward()
        optim.step()
        if step % 100 == 0:
            print('第{}轮 {}/{}迭代 loss: {:.6f}'.format(epoch, step, int(train_num / 64), loss.item()))
    writer.add_scalar('training loss', loss.item(), epoch)

    # 网络测试
    # model.eval()，只对特定的一些层有作用，比如drop out，训练和测试时不一样
    # with torch.no_grad()，不需要用到梯度，也不需要优化梯度
    # 每一轮结束，把测试集loss打印出来
    # 计算每一轮的准确率
    # outputs.argmax(从哪个维度看)
    # (preds == targets).sum()
    model.eval()
    test_loss = 0.0
    acc_num = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            results = model(imgs)
            loss = crossloss(results, labels)
            test_loss += loss
            acc_num += (results.argmax(1) == labels).sum()
        print('----第{}轮结束 精度:{:.4f} 测试loss:{:.6f}----'.format(epoch, acc_num.item()/len(test_data), test_loss))
        writer.add_scalar('testing loss', loss.item(), epoch)
        writer.add_scalar('testing acc', acc_num.item()/len(test_data), epoch)

    # 保存每一轮的结果，保存模型和参数
    torch.save(model, 'runs/epoch{}.pth'.format(epoch))
