# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:54:46 2018

@author: braxt
"""

import torch
import numpy as np
from PIL import Image
from matplotlib.image import imread
import glob
import cv2

i = 0
image_list = []
red_mean1 = []
blue_mean1 = []
green_mean1 = []
red_std1 = []
blue_std1 = []
green_std1 = []
#for filename in glob.glob('C:/Users/braxt.000/Documents/Python Playground/data/faces/*.jpg'): #assuming gif
for filename in glob.glob('C:/Users/braxt.000/Documents/Python Playground/Whales/all/train/*.jpg'): 
    im=cv2.imread(filename)
    image_list.append(im)
    working_image = image_list[i]
    (means, stds) = cv2.meanStdDev(working_image)
    stats = np.concatenate([means, stds]).flatten()
    red_mean1 = red_mean1 + [stats[2]]
    green_mean1 = green_mean1 + [stats[1]]
    blue_mean1 = blue_mean1 + [stats[0]]
    red_std1 = red_std1 + [stats[5]]
    green_std1 = green_std1 + [stats[4]]
    blue_std1 = blue_std1 + [stats[3]]
    i = i + 1
    if i > 8100:
        break
    
red_mean = (np.mean(red_mean1))/255
green_mean = (np.mean(green_mean1))/255
blue_mean = (np.mean(blue_mean1))/255

red_std = (np.mean(red_std1))/255
green_std = (np.mean(green_std1))/255
blue_std = (np.mean(blue_std1))/255