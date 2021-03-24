#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 01:06:27 2021

@author: hoda
"""
import numpy as np
import tensorflow as tf
from nobrainer.volume import from_blocks_numpy
from nobrainer.metrics import generalized_dice
import matplotlib.pyplot as plt
import json

def plot_output(path):
    outfile= np.load(path)
    label = outfile['label']
    result = outfile['result']
    plt.figure(1); plt.imshow(label[:,128,:])
    plt.figure(2); plt.imshow(result[:,128,:])
    
def save_parameters(file_name, model_name, **kwargs):
    vars = {
        "Model name":model_name,
        **kwargs
        }
    with open(file_name,'w') as fp:
        json.dump(vars, fp, indent=4)
    
def save_output(output_prefix, model, data, volume_shape, block_shape, one_hot_label=False):
    '''volume_shape and block_shape are tuple of 3'''    
    num_blocks = int((volume_shape[0]/block_shape[0])**3) 
    labels = np.empty(shape = (num_blocks,*block_shape))
    results = np.empty(shape = (num_blocks,*block_shape))
    data = data.unbatch().batch(1)
    for batch, (feat, label) in enumerate(data.take(num_blocks)):
        pred = model(feat)
        pred = np.argmax(pred, -1)
        if one_hot_label:
            label = tf.math.argmax(label, axis=-1)
        labels[batch,:,:,:] = label.numpy()
        results[batch,:,:,:] = pred
    
    labels = from_blocks_numpy(labels, volume_shape)
    results = from_blocks_numpy(results, volume_shape)    
    np.savez(output_prefix,label=labels,result= results)
    
def calcualte_dice(label, pred, n_classes, axis=(1,2,3), one_hot_label=False):
    """ pred is the output probabilities of the network"""
    #pred = np.argmax(pred, -1)
    #pred = tf.one_hot(pred, depth = n_classes)
    if not one_hot_label:
        label = tf.one_hot(label, depth= n_classes)
    return generalized_dice(label, pred, axis=axis)
    
    
    


