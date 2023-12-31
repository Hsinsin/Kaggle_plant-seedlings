import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from tools.augmentor import RandomErasing

class Test_set(Dataset):
    def __init__(self, root, transform):
        self.root = root 
        self.transform = transform
        self.img_list = os.listdir(root)

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        img_path = os.path.join(self.root,filename)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return(len(self.img_list))

def get_train_set(batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201]),
        RandomErasing()
    ])

    train_set = ImageFolder('data/train', train_transform)
    valid_set = ImageFolder('data/valid', train_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, valid_loader
    
def get_test_set(batch_size):
    test_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])
    test_set = Test_set('data/test',test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    train_set = ImageFolder('data/train', None)
    idx_to_class = dict((j, i) for i, j in train_set.class_to_idx.items())

    return test_loader, idx_to_class