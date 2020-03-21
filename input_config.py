# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:20:54 2020

@author: zubair
"""

epochs = 2
trian_path = 'dataset/train/'
valid_path = 'dataset/valid'
test_path = 'dataset/test'
n_classes = 7

##input image size
desired_size = [256, 256]


batch_size = 3

shuffle = True

#Donot change it now it is working for RGB images
no_channels = 3