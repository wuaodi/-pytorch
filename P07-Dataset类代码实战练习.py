from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir):
        self.root = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __getitem__(self, idx):
        """重写[]方法的调用"""
        image_list = os.listdir(os.path.join(self.root, self.image_dir))
        image_path = os.path.join(self.root, self.image_dir, image_list[idx])
        label_path = os.path.join(self.root, self.label_dir, image_list[idx][0:-4] + '.txt')
        img = Image.open(image_path)
        with open(label_path) as f:
            label = f.readline().strip()  # 读取第一行，去除后面的前后的空格、换行符等
        return img, label

    def __len__(self):
        """重写len()方法的调用"""
        return len(os.listdir(os.path.join(self.root, self.image_dir)))


ants_data = MyData("练手数据集", "train/ants_image", "train/ants_label")
image, label = ants_data[0]
image.show()
print('label: ', label)
print('数据集长度: ', len(ants_data))
