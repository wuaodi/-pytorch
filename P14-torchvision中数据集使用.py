"""如何把数据集与transforms结合起来使用"""
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 去官网看 pytorch.org-Docs-torchvision
# torchvision.datasets CIFAR10数据集 内部实现是继承自Dataset的类
# 5万张训练集，1万张测试集，10类，图片大小为32*32，100多M大小
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# 构建train_set 和 test_set
train_set = torchvision.datasets.CIFAR10('dataset', train=True, transform=trans, download=True)
test_set = torchvision.datasets.CIFAR10('dataset', train=False, transform=trans, download=True)
# print(type(train_set))
# print(train_set.class_to_idx)
# print(train_set[0])
# img, label = train_set[0]
# img.show()
# print('label_idx: ', label)

# 和transform进行联动，转为tensor类型，使用tensorboard显示前10张
writer = SummaryWriter('logs')
for i in range(10):
    img, label = train_set[i]
    writer.add_image('img', img, i)

writer.close()
