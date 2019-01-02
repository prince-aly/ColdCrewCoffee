# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:11:55 2019

@author: Yahia
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
#os.listdir(os.getcwd())    #List everyting in working directory
#os.chdir('/home/prince_aly/whales')
data = pd.read_csv('train.csv')
n = 42
img_name = data.iloc[n,0]
img_class = data.iloc[n,1]
print(img_name)
print(img_class)

class whaleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return 


whale_dataset = whaleDataset(csv_file= 'train.csv', root_dir = 'train/')

fig = plt.figure()

# Apply each of the above transforms on sample.
data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transformed_dataset = whaleDataset(csv_file='train.csv',
                                           root_dir='dtrain/',
                                           transform=data_transforms)
dataloaders = torch.utils.data.DataLoader(whale_dataset, batch_size=4, shuffle=True, num_workers=4)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(dataloaders)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
