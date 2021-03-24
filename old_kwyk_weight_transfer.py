#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer weights to the model with new structure with 115 classes and save the 
model weights
Created on Thu Mar 18 07:03:19 2021

@author: hoda
"""
# Imports
#import nobrainer
import tensorflow as tf
from baysian_meshnet import variational_meshnet


n_classes =115
volume_shape = (256, 256, 256)
block_shape = (32,32,32)
       
EPOCHS = 100
lr = 1e-06
BATCH_SIZE = 2
num_training_brains = 1
num_examples = int(((volume_shape[0]/block_shape[0])**3)*num_training_brains)
#num_examples=1
one_hot_label=True

initial_epoch = 0 ; scaling_start_epoch=5; scaling_increase_per_epoch = 0.3
warmup_factor=0

# create model and loading weights
old_model = variational_meshnet(
        n_classes = 50,
        input_shape = block_shape+(1,),
        filters=96,
        scale_factor = num_examples,
        dropout="concrete",
        receptive_field=37,
        #batch_size= BATCH_SIZE,
        warmup_factor = warmup_factor,
        )
# download weights
weights_path = tf.keras.utils.get_file(
    fname="nobrainer_spikeslab_32iso_weights.h5",
    origin="https://dl.dropbox.com/s/rojjoio9jyyfejy/nobrainer_spikeslab_32iso_weights.h5")

old_model.load_weights(weights_path)

# new model
new_model = variational_meshnet(
        n_classes = n_classes,
        input_shape = block_shape+(1,),
        filters=96,
        scale_factor = num_examples,
        dropout="concrete",
        receptive_field=37,
        #batch_size= BATCH_SIZE,
        warmup_factor = warmup_factor,
        )

# loading weights to the new model
for i, layer in enumerate(old_model.layers[:-2]):
    new_model.layers[i].set_weights(layer.get_weights())
    
# save the new model weights.
saved_weight_path="./training_files/old_kwyk_weights/kwyk_b32_cl115_weights.hd5/"
new_model.save_weights(saved_weight_path)    





