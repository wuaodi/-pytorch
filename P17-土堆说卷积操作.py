"""使用代码来演示一下torch的卷积操作"""
import torch
# torch.nn.functional相当于是一个个齿轮，
# torch.nn是封装成了变速箱方向盘，所以后面用torch.nn就可以了
import torch.nn.functional as F

# 写一个5*5的tensor，模拟单通道图像
img = torch.tensor([[1, 2, 0, 3, 1],
                    [0, 1, 2, 3, 1],
                    [1, 2, 1, 0, 0],
                    [5, 2, 3, 1, 1],
                    [2, 1, 0, 1, 1]])

# 写一个3*3的tensor，模拟卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 使用torch.nn.functional.conv2d
# 查看其文档是需要4个维度的数据的输入
# 使用torch.reshape变换尺寸
img = torch.reshape(img, [1, 1, 5, 5])
kernel = torch.reshape(kernel, [1, 1, 3, 3])
# 1、stride = 1
result = F.conv2d(img, kernel, stride=1)
print("result of stride=1:\n", result)

# 2、stride = 2
result = F.conv2d(img, kernel, stride=2)
print("result of stride=2:\n", result)

# 3、stride = 1, padding = 1（在外面填充一层0）
result = F.conv2d(img, kernel, stride=1, padding=1)
print("result of stride=1 and padding=1:\n", result)
