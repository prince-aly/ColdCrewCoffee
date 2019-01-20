# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:11:55 2019

@author: Team Cold Crew Coffee
"""
#Import Necessary Libraries
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
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Use the gpu if its available so code does not to be changed based on machine running it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print("Number of cuda devices: ",torch.cuda.device_count())

plt.ion()   # interactive mode
#List everyting in working directory
#os.listdir(os.getcwd())    

#Personal Directories

#os.chdir('/home/prince_aly/whales')
#dataDir = "C:\\Users\\Yahia\\Desktop\\FunProjects\\Whales\\train.csv"
#rootDir = "C:\\Users\\Yahia\\Desktop\\FunProjects\\Whales\\train"
#valDir = "C:\\Users\\Yahia\\Desktop\\FunProjects\\Whales\\val"
#Machine Directories
#Location of where the csv file is 
dataDir = "/home/prince_aly/whales/train.csv"
#Location to folder with all the images
rootDir = "/home/prince_aly/whales/train"
valDir =  "/home/prince_aly/whales/val"
#Read the csv file with the image names and labels then match each label to a number
data = pd.read_csv(dataDir, header = 0)  #Image names
labelsArr = np.asarray(data.iloc[:, 1])  #labels for each image
classes = pd.Series(labelsArr).unique()   #list of all the labels
classes.sort()   #sort the vector
class_to_idx = {classes[i]: i for i in range(len(classes))}  #Maps each label to a number

#Print a sample image and its label to make sure everything is running correctly
n = 42   #Arbitrary number
img_name = data.iloc[n,0]
img_class = data.iloc[n,1]
print(img_name)
print(img_class)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, root_dir, transform = None):
        """
        Args:
            csv_path (string): path to csv file
            root_dir (string): path to the folder where images are
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
        #Reisze the image to 128 pixels to standarize all images
        self.scale = transforms.Resize((128,128))
        #Transform to Tensor --- data type for pytorch to track gradients 
        self.to_tensor = transforms.ToTensor()
        #Crop size is large than actual image to not, 224 is chosen based on the transfer learning model used (resnet18)-----Change accordingly
        self.center_crop = transforms.CenterCrop(224)    
        #Normalize the mean and SD for each image
        self.normalize = transforms.Normalize([0.5, 0.5 , 0.5], [0.5, 0.5 , 0.5])   
         #If the transform is specified in the input, otherwise it remains empty
        self.transform = transform  
    #list of all the labels (species)
        self.classes = pd.Series(self.label_arr).unique()
        #sort the vector
        self.classes.sort()    
        #Maps each label to a number
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}   
    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = os.path.join(self.root_dir, self.image_arr[index])
        img = Image.open(img_name).convert('RGB')
        # Get label(class) of the image based on the cropped pandas column
        label = self.label_arr[index]
        
        #apply transformations if inputted, otherwise use standarized transforms specified in _init_ 
        if self.transform:
            img = self.transform(img)
        else:
            #Apply the transformations 
            img = self.scale(img)
            img = self.center_crop(img)
            img = self.to_tensor(img)
            img = self.normalize(img)
         #Returns the transformed image, and the number corresponding to that label
        return (img, self.class_to_idx[label]) 
    def __len__(self):
        #Return the total number of images---used in determining number of batches based on batch size
        return self.data_len



fig = plt.figure()

#Read all the images and apply different transformations if necessary
transformed_dataset = CustomDatasetFromImages(csv_path= dataDir, root_dir= rootDir)   
#Load the dataset into batches, shuffle to decrease overfitting, and specify computational use
trainloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=5, shuffle=True, num_workers = 4)  


## functions to show an image
# Make sure images are showing approporiately based on batch size
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


###### Transfer learning model ##########
model_ft = models.resnet18(pretrained=True)
#Obtrain the number of features in the last layer of the model
num_ftrs = model_ft.fc.in_features
#Change the number of features to the number of classes in the dataset
model_ft.fc = nn.Linear(num_ftrs, len(classes))
#Run the model on the GPU if available, and run on parallel if multiple GPU's are available
model_ft = torch.nn.DataParallel(model_ft.to(device))

#Set the loss function: Cross Entropy Loss
criterion = nn.CrossEntropyLoss()
#Set the optimizer algorithm- stochasitc gradient descent
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the +
        inputs, labels = data     #Inputs are the images
        #Set the inputs and labels as Variables (standard data type for Pytorch)
        #Send the images and  labels to GPU if available
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))

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

class TestDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.image_arr = os.listdir(root_dir)
        # Calculate len
        self.data_len = len(self.image_arr)
        self.root_dir = root_dir
        #Reisze the image to 128 pixels to standarize all images
        self.scale = transforms.Resize((128,128))
        #Transform to Tensor --- data type for pytorch to track gradients 
        self.to_tensor = transforms.ToTensor()
        #Crop size is large than actual image to not, 224 is chosen based on the transfer learning model used (resnet18)-----Change accordingly
        self.center_crop = transforms.CenterCrop(224)    
        #Normalize the mean and SD for each image
        self.normalize = transforms.Normalize([0.5, 0.5 , 0.5], [0.5, 0.5 , 0.5])   
         #If the transform is specified in the input, otherwise it remains empty
        self.transform = transform   
    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = os.path.join(self.root_dir, self.image_arr[index])
        img = Image.open(img_name).convert('RGB')
        # Get label(class) of the image based on the cropped pandas column
        #apply transformations if inputted, otherwise use standarized transforms specified in _init_ 
        if self.transform:
            img = self.transform(img)
        else:
            #Apply the transformations 
            img = self.scale(img)
            img = self.center_crop(img)
            img = self.to_tensor(img)
            img = self.normalize(img)
         #Returns the transformed image, and the number corresponding to that label
        return (img, self.image_arr[index]) 
    def __len__(self):
        #Return the total number of images---used in determining number of batches based on batch size
        return self.data_len


######## Getting output for test data
test_dataset = TestDataset(root_dir= valDir)   
#Load the dataset into batches, shuffle to decrease overfitting, and specify computational use

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers = 4)  
model_ft = model_ft.eval()
outputss = []
img_namess=  []
for i, data in enumerate(testloader, 0):
    inputs, img_names = data     #Inputs are the images
    inputs = Variable(inputs.to(device), volatile = True)
    outputs = model_ft(inputs)
    outputss.append(outputs)
    img_namess.append(img_names)

imgLabels = pd.DataFrame({classes[i] for i in outputss})
img_namess = pd.DataFrame(img_namess)
imgLabels.to_csv('labels.csv', index = False)
img_namess.to_csv('imgName.csv', index = False)
print("Testing Complete")
