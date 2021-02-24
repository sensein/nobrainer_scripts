#!/usr/bin/env python3
# -*- coding: utf-8 -*- >>> tf.__version__
#'2.1.3'
#import tensorflow_probability as tfp 
#>>> tfp.__version__
#'0.8.0-rc0'
"""
Created on Tue Feb 23 18:34:40 2021

@author: aakanksha
"""
# Imports
import nobrainer
import tensorflow as tf
import sys
import json
import glob
import numpy as np
import pandas as pd
import os
import warnings
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # pylint: disable=g-import-not-at-top
import tensorflow_probability as tfp
tfd = tfp.distributions

# params 

root_path = "/home/aakanksha/Documents/tfrecords/training/"
train_pattern = root_path+'data-evaluate_shard-*.tfrec'
eval_pattern = root_path + 'data-evaluate_shard-000.tfrec'

#root_path = '/nobackup/users/abizeul/kwyk/tfrecords/'
#train_pattern = root_path +'data-train_shard-*.tfrec'
#eval_pattern = root_path + "data-evaluate_shard-*.tfrec"
  
block_shape = (64,64,64) 
n_classes = 115
volume_shape = (256, 256, 256)
EPOCHS = 50
batch_Size = 1
learning_rate = 0.0001
model_name = "kwyk_check1_{}.ckpt"
batch_size = 1
max_steps = 6000
model_dir = '/home/aakanksha/Documents/wazeer_imp/nobrainer-add-progressivegan/models/'
viz_steps = 4000
NUM_TRAIN_EXAMPLES = 10
NUM_HELDOUT_EXAMPLES = 10000
num_monte_carlo = 10
checkpoint_dir = os.path.join("training_files",'kwyk',"training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#from tensorflow.contrib.learn.python.learn.datasets import mnist
def _to_blocks(x, y,block_shape):
    """Separate `x` into blocks and repeat `y` by number of blocks."""
    print(x.shape)
    x = nobrainer.volume.to_blocks(x, block_shape)
    y = nobrainer.volume.to_blocks(y, block_shape)
    return (x, y)

def get_dict(n_classes):
    print('Conversion into {} segmentation classes from freesurfer labels to 0-{}'.format(n_classes,n_classes-1))
    if n_classes == 50: 
        tmp = pd.read_csv('50-class-mapping.csv', header=0,usecols=[1,2],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        return mydict
    elif n_classes == 115:
        tmp = pd.read_csv('115-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        del tmp
        return mydict
    else: raise(NotImplementedError)

def process_dataset(dset,batch_size,block_shape,n_classes,train= True):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), nobrainer.volume.replace(y,get_dict(n_classes))))
    # Separate features into blocks.
    dset = dset.map(lambda x, y:_to_blocks(x,y,block_shape))
    # This step is necessary because separating into blocks adds a dimension.
    dset = dset.unbatch()
    #dset = dset.shuffle(buffer_size=100)
    # Only shuffle the dset for training
    if train:
        dset = dset.shuffle(buffer_size=100)
    else:
        pass
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    return dset

def get_dataset(pattern,volume_shape,batch,block_shape,n_classes,train=True):
    dataset = nobrainer.dataset.tfrecord_dataset(
        file_pattern=glob.glob(pattern),
        volume_shape=volume_shape,
        shuffle=False,
        scalar_label=False,
        compressed=True)
    dataset = process_dataset(dataset,batch,block_shape,n_classes,train=train)
    return dataset

def create_model():
  """Creates a Keras model using the LeNet-5 architecture.
  Returns:
      model: Compiled Keras model.
  """
  # KL divergence weighted by the number of training samples, using
  # lambda function to pass as input to the kernel_divergence_fn on
  # flipout layers.
  kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

  neural_net = tf.keras.Sequential([
        tf.keras.Input(shape=block_shape+(1,), batch_size= batch_size),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (1,1,1),
                                        padding="SAME",
                                        activation=tf.nn.relu,
                                        kernel_divergence_fn=kl_divergence_function),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (2,2,2),
                                        padding="SAME",
                                        activation=tf.nn.relu,
                                        kernel_divergence_fn=kl_divergence_function),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (4,4,4),
                                        padding="SAME",
                                        activation=tf.nn.relu,
                                        kernel_divergence_fn=kl_divergence_function),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (8,8,8),
                                        padding="SAME",
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (16,16,16),
                                        padding="SAME",
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (32,32,32),
                                        padding="SAME",
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(96,
                                        kernel_size=3,
                                        dilation_rate= (1,1,1),
                                        padding="SAME",
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tfp.layers.Convolution3DFlipout(n_classes, 
                                kernel_size = 1, 
                                dilation_rate= (1,1,1),
                                padding = 'SAME',
                                activation=tf.nn.softmax)])

  optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=neural_net)
  neural_net.compile(optimizer, loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'], experimental_run_tf_function=False)
  return neural_net, checkpoint

model, checkpoint = create_model()
model.build(input_shape=block_shape+(1,))
dataset_train = get_dataset(train_pattern,volume_shape,batch_size,block_shape,n_classes)
dataset_eval = get_dataset(eval_pattern,volume_shape,batch_size,block_shape,n_classes, train= False)
#val_img = nib.load('/home/aakanksha/Documents/wazeer_imp/nobrainer-add-progressivegan/Sample_T1.nii.gz')
for epoch in range(EPOCHS):
    print('Epoch number ',epoch)
    i = 0; epoch_accuracy, epoch_loss = [], []
    for steps, (batch_x,batch_y) in enumerate(dataset_train.take(100)):
        batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
        epoch_accuracy.append(batch_accuracy)
        epoch_loss.append(batch_loss)
                
        if steps % 400 == 0:
            print('Epoch: {}, Batch index: {}, '
              'Loss: {:.3f}, Accuracy: {:.3f}'.format(
                  epoch, steps,tf.reduce_mean(epoch_loss),
                  tf.reduce_mean(epoch_accuracy)))
    if epoch % 10 == 0:
        checkpoint.save(checkpoint_prefix.format(epoch=epoch))
            
saved_model_path=os.path.join("./training_files",model_name,"saved_model/")
model.save(saved_model_path, save_format='tf')
#            out = validations(val_img, model, block_shape= block_shape, epoch= epoch, batch_size= 1)
#        if (step+1) % viz_steps == 0:
#            print(' ... Running monte carlo inference')
#            probs = tf.stack([model.predict(dataset_eval.take(10)), verbose=1)
#                          for _ in range(num_monte_carlo)], axis=0)
#            mean_probs = tf.reduce_mean(probs, axis=0)
#            heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
#            print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))    