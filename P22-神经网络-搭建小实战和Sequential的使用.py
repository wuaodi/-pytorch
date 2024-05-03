"""使用Containers中的 nn.Sequential 模块 搭建一个对CIFAR10分类的网络 """

import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 加载数据
test_dataset = torchvision.datasets.CIFAR10('dataset', train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)


# 搭建模型
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


# 实例化，并输入输出数据
model = My_model()
print(model)
for step, data in enumerate(test_dataloader):
    imgs, label = data
    result = model(imgs)
    print('输入的shape：', imgs.shape)
    print('输出的shape：', result.shape)
    # 调试的时候可以把后面的层删除掉看中间层的输出的shape
    break

# 使用tensorboard可视化模型 add_graph
writer = SummaryWriter('logs')
writer.add_graph(model, imgs)
writer.close()
