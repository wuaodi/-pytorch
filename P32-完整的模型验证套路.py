"""利用训练好的模型进行测试"""
import torch
from torch import nn
from PIL import Image
import cv2
import torchvision

device = torch.device('cuda:0')

# 使用PIL读取一张图片
# 使用image = image.convert('RGB')可以实现4通道png图像只保留三个颜色通道，jpg图像不变
img = Image.open('dog.png')
img = img.convert('RGB')
img_cv = cv2.imread('dog.png')
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# 对图像进行变换，ToTensor，Resize，四个维度
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize([32, 32])])
img = trans(img)
img = torch.reshape(img, [1, 3, 32, 32])
img = img.to(device)
img_cv = trans(img_cv)
img_cv = torch.reshape(img_cv, [1, 3, 32, 32])
img_cv = img_cv.to(device)

print(img.shape)
print(img_cv.shape)


# 加载网络模型
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


model = torch.load('runs/epoch25.pth')
model = model.to(device)

# 测试，eval模式，不要梯度
model.eval()
with torch.no_grad():
    result = model(img)
    result = result.to('cpu').numpy()
    print('-------------------------')
    print(type(result))
    print(result)

    result_cv = model(img_cv)
    print('-------------------------')
    print(result_cv)
    print(result_cv.argmax(1).item())
