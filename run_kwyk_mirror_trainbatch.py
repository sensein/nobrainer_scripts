import nobrainer
import tensorflow as tf
import sys
import json
import glob
import numpy as np
import os
import pandas as pd
from nobrainer.models.bayesian import variational_meshnet
import losses
from losses import *

def _to_blocks(x, y,block_shape):
    """Separate `x` into blocks and repeat `y` by number of blocks."""
    print(x.shape)
    x = nobrainer.volume.to_blocks(x, block_shape)
    y = nobrainer.volume.to_blocks(y, block_shape)
    return (x, y)

def get_dict(n_classes):
    print('Conversion into {} segmentation classes from freesurfer labels to 0-{}'.format(n_classes,n_classes-1))
    if n_classes == 50: 
        tmp = pd.read_csv('50-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        return mydict
    elif n_classes == 115:
        tmp = pd.read_csv('115-class-mapping.csv', header=0,usecols=[0,1],dtype=np.int32)
        mydict = dict(tuple(zip(tmp['original'],tmp['new'])))
        del tmp
        return mydict
    else: raise(NotImplementedError)

def process_dataset(dset,batch_size,block_shape,n_classes):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), nobrainer.volume.replace(y,get_dict(n_classes))))
    # Separate features into blocks.
    dset = dset.map(lambda x, y:_to_blocks(x,y,block_shape))
    # This step is necessary because separating into blocks adds a dimension.
    dset = dset.unbatch()
    dset = dset.shuffle(buffer_size=100)
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    return dset

def get_dataset(pattern,volume_shape,batch,block_shape,n_classes):

    dataset = nobrainer.dataset.tfrecord_dataset(
        file_pattern=glob.glob(pattern),
        volume_shape=volume_shape,
        shuffle=False,
        scalar_label=False,
        compressed=True)
    dataset = process_dataset(dataset,batch,block_shape,n_classes)
    return dataset

def run(block_shape):
    
    # Constants
    root_path = '/om/user/satra/kwyk/tfrecords/'
    train_pattern = root_path+'data-train_shard-*.tfrec'

    n_classes =115
    volume_shape = (256, 256, 256)      
    EPOCHS = 1
    BATCH_SIZE_PER_REPLICA = 1

    #Setting up the multi gpu strategy
    strategy = tf.distribute.MirroredStrategy()
    print("Number of replicas {}".format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # Create a `tf.data.Dataset` instance.
    dataset_train = get_dataset(train_pattern,volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes)

    # Distribute dataset.
    train_dist_dataset = strategy.experimental_distribute_dataset(dataset_train)
 
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(1e-03)
        model = variational_meshnet(n_classes=n_classes,input_shape=block_shape+(1,), filters=96,dropout="concrete",is_monte_carlo=True,receptive_field=129) 
        loss_fn = losses.ELBO(model=model, num_examples=np.prod(block_shape),reduction=tf.keras.losses.Reduction.NONE)
        model.compile(loss=loss_fn,optimizer=optimizer,experimental_run_tf_function=False)
      
        for epoch in range(EPOCHS):
            print('Epoch number ',epoch)
            i = 0
            for data in dataset_train:
                i += 1
                error = model.train_on_batch(data)
                print('Batch {}, error : {}'.format(i,error))

if __name__ == '__main__':
   
    block_shape = (int(sys.argv[1]),int(sys.argv[1]),int(sys.argv[1]))
    run(block_shape)
