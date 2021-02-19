import tensorflow as tf
import nobrainer
import glob
import numpy as np
import pandas as pd


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
    
def process_dataset(dset,batch_size,block_shape,n_classes,training= True):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), nobrainer.volume.replace(y,get_dict(n_classes))))
    # Separate features into blocks.
    dset = dset.map(lambda x, y:_to_blocks(x,y,block_shape))
    # This step is necessary because separating into blocks adds a dimension.
    dset = dset.unbatch()
    if training:
        dset = dset.shuffle(buffer_size=100)
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    return dset

def get_dataset(pattern,volume_shape,batch,block_shape,n_classes, training = True):

    dataset = nobrainer.dataset.tfrecord_dataset(
        file_pattern=glob.glob(pattern),
        volume_shape=volume_shape,
        shuffle=False,
        scalar_label=False,
        compressed=True)
    dataset = process_dataset(dataset,batch,block_shape,n_classes, training = training)
    return dataset





