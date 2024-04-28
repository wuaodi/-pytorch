# torch 是一个package，一个工具箱，可能有小的分割区，每个里面有不同的工具

# dir() 打开，看见工具箱里有什么
# help() 说明书

import torch
torch.cuda.is_available()

print(dir(torch))
print(dir(torch.cuda))
print(dir(torch.cuda.is_available))
# 返回的是各种各样的魔法函数，说明这个是个工具了，而不是一个工具箱分割区，可以用help看看这个工具怎么用

print(dir(torch.cuda.is_available))
# 返回的是各种各样的魔法函数，说明这个是个工具了，而不是一个工具箱分割区，可以用help看看这个工具怎么用

print(help(torch.cuda.is_available))