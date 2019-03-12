import numpy as np
import torch
from torchvision import datasets,transforms
import torch.utils.data as pData
import os
from PIL import Image

class MyDataLoader(pData.Dataset):
    def __init__(self,picNames):
        self.paths = picNames
        self.transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])])


    def __getitem__(self,index):
        if 'well' in self.paths[index]:
            label=1
        elif 'ill' in self.paths[index]:
            label=0
        img = Image.open(self.paths[index])
        img = self.transform(img)
        return img,label

    def __len__(self):
        return self.paths.shape[0]
