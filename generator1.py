# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:06:14 2020

@author: zubair
"""
import os
from PIL import Image, ImageOps
import numpy as np
def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def data_gen(path, desired_size, no_channels, batch_size, shuffle):
    
    classes_list = os.listdir(path)
    images_path = []
    classes = []
    for i in range(len(classes_list)):
        j = os.listdir(path+ "/" + classes_list[i])
        for images in j:
            pa = path + "/" + classes_list[i] + "/" + images
            images_path.append(pa)
            classes.append(i)
    if shuffle == True:
        import random
        c = list(zip(images_path, classes))
        random.shuffle(c)
        images_path, classes = zip(*c)
        
        
    x = np.zeros((0, desired_size[0], desired_size[1], no_channels))
    y = np.zeros((0))
    batch  = 0
    for img in range(len(images_path)):
        xx = []
        yy = []
        img1 = Image.open(images_path[img])
        #print(img)
        img1 = resize_with_padding(img1, (desired_size[0], desired_size[1]))
        #print(img1.size)
        img1 = np.array(img1)
        img1 = img1/255.
        img1 = img1.reshape(1,desired_size[0], desired_size[1], no_channels)
        x = np.append(x, img1, axis=0)
        #x.append(img1)
        y = np.append(y,classes[img])
        #y.append(classes[img]) 
        batch = batch + 1
        try:
            if batch == batch_size:
                #print(y)
                y = y.astype(int)
                n_values = np.max(classes) + 1
                batch = 0
                y = np.eye(n_values)[y]
                xx.append(x)
                yy.append(y)
                
                y = np.zeros((0))
                x = np.zeros((0,desired_size[0], desired_size[1], no_channels))
                xx = np.array(xx)
                xx = xx.astype(np.float32)
                
                yy = np.array(yy)
                yy = yy.astype(np.float32)
                yy = yy.reshape(yy.shape[1], yy.shape[2])
                xx = xx.reshape(xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4])
                yield(xx, yy)
        except:
            continue
       
#
#def valid_data_gen(path, desired_size, no_channels, batch_size, shuffle):   
#    classes_list = os.listdir(path)
#    images_path = []
#    classes = []
#    for i in range(len(classes_list)):
#        j = os.listdir(path+ "/" + classes_list[i])
#        for images in j:
#            pa = path + "/" + classes_list[i] + "/" + images
#            images_path.append(pa)
#            classes.append(i)
#    if shuffle == True:
#        import random
#        c = list(zip(images_path, classes))
#        random.shuffle(c)
#        images_path, classes = zip(*c)
#     
#    x = np.zeros((0, desired_size[0], desired_size[1], no_channels))
#    y = np.zeros((0))
#    batch  = 0
#    for img in range(len(images_path)):
#        xx = []
#        yy = []
#        img1 = Image.open(images_path[img])
#        #print(img)
#        img1 = resize_with_padding(img1, (desired_size[0], desired_size[1]))
#        #print(img1.size)
#        img1 = np.array(img1)
#        img1 = img1/255.
#        img1 = img1.reshape(1,desired_size[0], desired_size[1], no_channels)
#        x = np.append(x, img1, axis=0)
#        #x.append(img1)
#        y = np.append(y,classes[img])
#        #y.append(classes[img]) 
#        batch = batch + 1
#        try:
#            
#            if batch == batch_size:
#                #print(y)
#                y = y.astype(int)
#                n_values = np.max(classes) + 1
#                batch = 0
#                y = np.eye(n_values)[y]
#                xx.append(x)
#                yy.append(y)
#                
#                y = np.zeros((0))
#                x = np.zeros((0,desired_size[0], desired_size[1], no_channels))
#                xx = np.array(xx)
#                xx = xx.astype(np.float32)
#                
#                yy = np.array(yy)
#                yy = yy.astype(np.float32)
#                yy = yy.reshape(yy.shape[1], yy.shape[2])
#                xx = xx.reshape(xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4])
#                yield(xx, yy)
#        except:
#            
#            continue
