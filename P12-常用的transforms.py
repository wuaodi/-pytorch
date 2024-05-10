# 看官方文档会更准确
# 注意看输入输出类型，关注方法需要什么参数
# 如果输出没写，可以print(type(变量))，可以run in console，可以断点调试
# 本代码关于 ToTensor Normalize Resize RandomCrop Compose

# File | Settings | Editor | General | Code Completion -> 取消Match case前面的对勾可以不区分大小写匹配

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 使用Image读入图片，初始化tensorboard
img = Image.open('练手数据集/train/ants_image/0013035.jpg')
writer = SummaryWriter('logs')

# ToTensor，使用tensorboard显示
""" Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('tensor img', img_tensor, 0)

# Normalize
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.4, 0.5, 0.5))
img_normal = trans_normalize(img_tensor)
writer.add_image('normal img', img_normal, 0)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img_tensor)
print(type(img_resize))
writer.add_image('resize512 img', img_resize, 0)

# RandomCrop, 裁剪十个都显示
trans_randomcrop = transforms.RandomCrop((256, 256))
for i in range(10):
    img_crop = trans_randomcrop(img_tensor)
    writer.add_image('random crop img', img_crop, i)

# Compose
trans_compose = transforms.Compose([trans_totensor, trans_normalize, trans_randomcrop])
for i in range(10):
    img_compose = trans_compose(img)
    writer.add_image('trans_compose img', img_compose, i)

writer.close()


