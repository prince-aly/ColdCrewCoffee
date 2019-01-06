# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:11:55 2019

@author: Yahia
"""

from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torchvision
from torch.autograd import Variable
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of cuda devices: ",torch.cuda.device_count())

plt.ion()   # interactive mode
#os.listdir(os.getcwd())    #List everyting in working directory
#os.chdir('/home/prince_aly/whales')
data = pd.read_csv('train.csv')
n = 42
img_name = data.iloc[n,0]
img_class = data.iloc[n,1]
print(img_name)
print(img_class)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, root_dir, transform = None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.root_dir = root_dir
        self.scale = transforms.Resize((64,64))
        self.to_tensor = transforms.ToTensor()
#        self.center_crop = transforms.CenterCrop(224)
        self.transform = transform
        self.classes = pd.Series(self.label_arr).unique()
    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = os.path.join(self.root_dir, self.image_arr[index])
        img = Image.open(img_name)
    
#        img_as_tensor = self.to_tensor(img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        if self.transform:
            img = self.transform(img)
        else:
            img = self.scale(img)
#            img = self.center_crop(img)
            img = self.to_tensor(img)
        return (img, single_image_label)
    def __len__(self):
        return self.data_len



fig = plt.figure()

# Apply each of the above transforms on sample.
data_transforms = transforms.Compose([transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transformed_dataset = CustomDatasetFromImages(csv_path='train.csv',
                                           root_dir='train/')
trainloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=3, shuffle=True, num_workers = 0)
classes = transformed_dataset.classes
import matplotlib.pyplot as plt
## functions to show an image
#
#
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(classes))

model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#dataiter = iter(testloader)
#images, labels = dataiter.next()
#
## print images
#imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
#outputs = model_ft(images)
#_, predicted = torch.max(outputs, 1)
#
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))
#
#correct = 0
#total = 0
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        outputs = model_ft(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#print('Accuracy of the network on the 10000 test images: %d %%' % (
#    100 * correct / total))