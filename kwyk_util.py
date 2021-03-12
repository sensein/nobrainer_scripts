#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 01:06:27 2021

@author: hoda
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_output(path):
    outfile= np.load(path)
    label = outfile['label']
    result = outfile['result']
    ax = plt.imshow(label[:,128,:])
    ax = plt.imshow(result[:,128,:])
    
    
    
    


