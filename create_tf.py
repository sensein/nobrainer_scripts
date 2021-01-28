import os
import numpy
import glob
import nibabel
import sys
sys.path.append('.')
import nobrainer.dataset 
import nobrainer.transform
import nobrainer.tfrecord
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import csv

def _read_csv(filepath, skip_header=True, delimiter=","):
    """Return list of tuples from a CSV, where each tuple contains the items
    in a row.
    """
    with open(filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        return [tuple(row) for row in reader]

volume_filepaths = _read_csv('kwyk_paths.csv')

nobrainer.tfrecord.write(
    features_labels=volume_filepaths,
    filename_template='/om/user/abizeul/nobrainer_expe/data/data_shard-{shard:03d}.tfrec',
    examples_per_shard=200,
    multi_resolution=True,
    affine=False,
    bias_field=False,
    resolutions=[8,16,32,64,128,256],
)

