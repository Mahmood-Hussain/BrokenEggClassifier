import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class EggDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        print(f'Found Classes: {self.classes}')
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                sample = (img_path, self.class_to_idx[cls_name])
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label

# test dataset
if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    data_dir = '/media/hdd/mahmood/datasets/EggCompetetion/train'
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),

        # color transforms
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        transforms.ToTensor()
    ])
    dataset = EggDataset(data_dir, t)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for imgs, labels in dataloader:
        print(imgs.shape)
        print(labels)
        for img, label in zip(imgs, labels):
            plt.imshow(img.permute(1, 2, 0))
            plt.title(dataset.classes[label])
            plt.show()
        break