from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root = root_dir
        self.label_dir = label_dir
        self.image_dir = os.path.join(self.root, self.label_dir)
        self.image_list = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        """重写[]方法的调用"""
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        img = Image.open(img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        """重写len()方法的调用"""
        return len(self.image_list)


ants_data = MyData("hymenoptera_data/train", "ants")
bees_data = MyData("hymenoptera_data/train", "bees")
# 两个小的数据集可以直接叠加，父类Dataset中有实现
dataset = ants_data + bees_data
image, label = dataset[125]
image.show()
print(len(dataset))
