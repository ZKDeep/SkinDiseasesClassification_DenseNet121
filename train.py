# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:07:28 2020

@author: zubair
"""


import numpy as np
import model
import input_config
import generator1
from os import walk


epochs = input_config.epochs
train_path = input_config.trian_path
valid_path = input_config.valid_path
desired_size = input_config.desired_size
batch_size = input_config.batch_size
shuffle = input_config.shuffle
no_channels = input_config.no_channels
model = model.model()
los = 99999

t = []
for (dirpath, dirnames, filenames) in walk(train_path):
    t.extend(filenames)
    

v = []
for (dirpath, dirnames, filenames) in walk(valid_path):
    v.extend(filenames)
    

t_files = len(t)
v_files = len(v)

try:
    model.load_weights("weights.h5", by_name=True)
    print("Loading Exiting weights........ please wait")
except:
    print("No weights found...... model loaded with image_net weights...")
    pass



for epoch in range(epochs):   
    try:
        print(".....")
        print(".....")
        print(".....")
        print(".....")
        print("epoch No=====>" + str(epoch))
        train_batches = generator1.data_gen(train_path, desired_size, no_channels, batch_size, shuffle)
        valid_batches = generator1.data_gen(valid_path, desired_size, no_channels, batch_size, shuffle)
        
        hist = model.fit_generator(train_batches, steps_per_epoch=np.floor(t_files/batch_size),
                           epochs =1, verbose=1, shuffle =True)
        print("validating====>")
        evl = model.evaluate_generator(valid_batches, steps=np.floor(v_files/batch_size), verbose=1 )
        
        new_los = evl[0]
        print("validation loss====>" + str(evl[0]))
        
        if new_los < los:
            print("Saving best weights")
            model.save_weights("weights.h5")
            los = new_los
    except:
        continue
