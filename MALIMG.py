import os
import glob
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import torch

class MalimgDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,evalt=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.evalt = evalt #to diff with eval()


        if self.train:
            data_path = os.path.join(self.root, 'train')
            print("Use train data set")
        else:
            if self.evalt:
                data_path = os.path.join(self.root, 'val')
                print("Use eval data set")
            else:    
                data_path = os.path.join(self.root, 'test')
                print("Use test data set")
        self.data = datasets.ImageFolder(data_path, transform=self.transform)


    @property
    def class_to_idx(self):
        return self.data.class_to_idx

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
