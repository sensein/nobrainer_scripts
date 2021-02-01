#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:00:18 2021

@author: hoda
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nobrainer.metrics import dice

root_path="training_files/"
model_name="kwyk_check_21-01-26_19-06"
file_name="output-kwyk_check_21-01-26_19-06.json"

path = os.path.join(root_path,model_name,file_name)
with open(path) as f:
    data = json.load(f)
    
labels = data["labels"]
preds = data["predictions"]

# take the first batch
lbl_0 = labels[0]
prd_0 = preds [0]
# chaning the list to 3D array
lbl_0 = np.array(lbl_0, ndmin=3)
prd_0 = np.array(prd_0, ndmin=3)
# take one 3d array
lbl_0_0 = lbl_0[0,:,:,:]
prd_0_0 = prd_0[0,:,:,:]
# plot the mid slice
slc_l = lbl_0_0[64,:,:]
slc_p = prd_0_0[64,:,:]
plt.imshow(slc_l,cmap="jet")
plt.title("label")
plt.imshow(slc_p,cmap="jet")
plt.title("prediction")

# calculate the dice score for the first batch
n_classes = 115
lbl_0_h = tf.one_hot(lbl_0,depth = n_classes)
prd_0_h = tf.one_hot(prd_0, depth = n_classes)
dice_score = tf.reduce_mean(dice(lbl_0_h,prd_0_h, axis=(1,2,3)))

 