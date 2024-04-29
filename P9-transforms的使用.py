# 从torchvision中导入transforms

# transforms.ToTensor 去看两个问题
# 1、transforms如何使用：输入图片，创建自己的工具进行一系列变换，输出变换后的图片
# 2、为什么需要tensor的数据类型

# ctrl+p 可以看一个函数需要什么参数

from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# PIL.JpegImagePlugin.JpegImageFile类型 RGB
img = Image.open('练手数据集/train/ants_image/0013035.jpg')
# Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
trans = transforms.ToTensor()
# tensor包含了梯度、设备、反向传播等一系列的参数
# C H W 3 512 768
img_trans = trans(img)
print(img_trans)
print(img_trans.shape)

# 写tensorboard日志
writer = SummaryWriter('logs')
writer.add_image('tensor_img', img_trans, 1)
writer.close()

