"""Dataset就是一幅扑克牌，DataLoader就是从扑克牌中取牌"""
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# 看官方文档，可以在Docs里面搜索一下

# 加载CIFAR 10的测试集
test_data = torchvision.datasets.CIFAR10('dataset', train=False, transform=torchvision.transforms.ToTensor())
# 创建test_loader对象
# dataloader(batch_size=64), 会分别对dataset的64组数据的图片和标签分别打包，返回一个高维的图片张量和一个高维的标签张量
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, drop_last=True)

# 使用tensorboard来展示
writer = SummaryWriter('logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        img, label = data
        # 注意由于是一个batch的图片不是单张图片，所以用add_images
        writer.add_images('epoch{}'.format(epoch), img, step)
        step += 1

writer.close()
