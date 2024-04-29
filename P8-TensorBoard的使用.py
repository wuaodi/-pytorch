# 对于探究模型的不同阶段是如何输出的很游泳

from torch.utils.tensorboard import SummaryWriter
import cv2

# 创建实例，传入存储的文件夹路径
writer = SummaryWriter("logs")

# 添加标量, 添加图片
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)

image_path = '练手数据集/train/ants_image/0013035.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)  # H W C
writer.add_image('test', img, 1, dataformats='HWC')

# 关闭
writer.close()

# 打开日志命令
# tensorboard --logdir=logs --port=6007
