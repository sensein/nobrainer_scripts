import nobrainer
import tensorflow as tf
import sys
import json
import glob
import datetime
import numpy as np
import os
import pandas as pd
from nobrainer.models.bayesian import variational_meshnet
from nobrainer.metrics import dice
import losses
from losses import *
from time import time


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

def run(block_shape, dropout_typ,model_name):
    
    # Constants
    #root_path = '/om/user/satra/kwyk/tfrecords/'
    # to run the code on Satori
    root_path = "/nobackup/users/abizeul/kwyk/tfrecords/"

    train_pattern = root_path+'data-train_shard-*.tfrec'
    eval_pattern = root_path + "data-evaluate_shard-*.tfrec"

    

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
    dataset_eval = get_dataset(eval_pattern,volume_shape,GLOBAL_BATCH_SIZE,block_shape,n_classes, train= False)

    # Distribute dataset.
    #train_dist_dataset = strategy.experimental_distribute_dataset(dataset_train)

    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = os.path.join("training_files",model_name,"training_checkpoints")
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
 
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(1e-03)
        model = variational_meshnet(n_classes=n_classes,input_shape=block_shape+(1,), filters=96,dropout=dropout_typ,is_monte_carlo=True,receptive_field=129) 
        loss_fn = losses.ELBO(model=model, num_examples=np.prod(block_shape),reduction=tf.keras.losses.Reduction.NONE)
        #dice_metric = generalized_dice()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        model.compile(loss=loss_fn,optimizer=optimizer,experimental_run_tf_function=False)
      
        # training loop
        train_loss=[]
        start=time()
        for epoch in range(EPOCHS):
            print('Epoch number ',epoch)
            i = 0
            for data in dataset_train:
                i += 1
                error = model.train_on_batch(data)
                train_loss.append(error)
                print('Batch {}, error : {}'.format(i, error))

            checkpoint.save(checkpoint_prefix.format(epoch=epoch))
        training_time=time()-start
            

        # evaluating loop
        print("---------- evaluating ----------")
        i=0
        eval_loss=[]
        dice_scores=[]
        preds=[]
        labels=[]
        #class_dice= np.zeros_like(n_classes)
        for data in dataset_eval.take(10):
            i += 1
            eval_error = model.test_on_batch(data)
            eval_loss.append(eval_error)
            # calculate dice
            result = model.predict_on_batch(data)
            preds.append(np.argmax(result,-1).tolist())
            (feat, label) = data
            labels.append(label.numpy().tolist())
            label = tf.one_hot(label, depth= n_classes)
            dice_score = tf.reduce_mean(dice(label,result,axis=(1,2,3))).numpy()
            dice_scores.append(dice_score.tolist())
            print('Batch {}, eval_loss : {}, dice_score: {}'.format(i, eval_error, dice_score))


        # Save model and variables

        #model_name="kwyk_128_full.h5"
        #saved_model_path=os.path.join("./training_files",model_name,"saved_model/{}.h5".format(model_name))
        #model.save(saved_model_path, save_format='h5')
        saved_model_path=os.path.join("./training_files",model_name,"saved_model/")
        model.save(saved_model_path, save_format='tf')

        variables={
            "train_loss":train_loss,
            "eval_loss":eval_loss,
            "eval_dice":dice_scores
        }
        file_path = os.path.join("training_files",model_name,"data-{}.json".format(model_name))
        with open(file_path, 'w') as fp:
            json.dump(variables, fp, indent=4)

        outputs={
            "labels":labels,
            "predictions":preds
        }
        file_path = os.path.join("training_files",model_name,"output-{}.json".format(model_name))
        with open(file_path, 'w') as fp:
            json.dump(outputs, fp, indent=4)

        
    return training_time



if __name__ == '__main__':

    start=time()
    model_name="kwyk_check_{}".format(datetime.datetime.now().strftime("%y-%m-%d_%H-%M"))
    print("----------------- model name: {} -----------------".format(model_name))
    os.mkdir(os.path.join("training_files",model_name))
    os.mkdir(os.path.join("training_files",model_name,"saved_model"))
    os.mkdir(os.path.join("training_files",model_name,"training_checkpoints"))

    block_shape = (int(sys.argv[1]),int(sys.argv[1]),int(sys.argv[1]))
    dropout=sys.argv[2]
    if dropout == "None":
        dropout = None
    training_time=run(block_shape,dropout,model_name)
    end=time()-start
    print("training loop takes: {} & whole code takes: {}".format(training_time, end))
