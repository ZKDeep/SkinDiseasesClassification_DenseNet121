# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:15:46 2020

@author: zubair
"""
import os
from os import walk
import numpy as np
import model
import input_config
import generator1

test_path = input_config.test_path
desired_size = input_config.desired_size
batch_size = input_config.batch_size
shuffle = input_config.shuffle
no_channels = input_config.no_channels
model = model.model()

model.load_weights('weights.h5', by_name = True)



v = []
for (dirpath, dirnames, filenames) in walk(test_path):
    v.extend(filenames)
test_files = len(v)


valid_batches = generator1.data_gen(test_path, desired_size, no_channels, batch_size, shuffle)
evl = model.evaluate_generator(valid_batches , steps= np.floor(test_files/batch_size), verbose=1)

print("Loss on Test ====>" + str(evl[0]) + "    "  + "Accuracy on Test ===>"  + str(evl[1]))