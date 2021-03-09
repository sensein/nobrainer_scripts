#!/usr/bin/env python3

# Imports
#import nobrainer
import tensorflow as tf
import os
import numpy as np
from kwyk_data import get_dataset
from nobrainer.volume import from_blocks_numpy
from nobrainer.metrics import generalized_dice
from baysian_meshnet import variational_meshnet


# constants
#root_path = '/om/user/satra/kwyk/tfrecords/'
#root_path = '/om2/user/hodaraja/kwyk/nobrainer_scripts/'
#root_path = "data/"
# to run the code on Satori
root_path = "/nobackup/users/abizeul/kwyk/tfrecords/"

#train_pattern = root_path+"single_volume-000.tfrec"
#eval_pattern = root_path + "single_volume-000.tfrec"

train_pattern = root_path+"data-train_shard-000.tfrec"
eval_pattern = root_path + "data-evaluate_shard-000.tfrec"

n_classes =50
volume_shape = (256, 256, 256)
block_shape = (64,64,64)
       
EPOCHS = 5 
lr = 1e-04
BATCH_SIZE = 1
scale = 64

model_name = "kwyk_kl_b{}_cl{}".format(block_shape[0], n_classes)
checkpoint_dir = os.path.join("training_files",model_name,"training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

print("--------- Loading data -------------")
dataset_train = get_dataset(train_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes)
dataset_eval = get_dataset(eval_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes, training= False)
print("_________ data loaded with block_size: {}, batch_size: {}___________".format(block_shape[0], BATCH_SIZE)) 

# create the model
model = variational_meshnet(
    n_classes = n_classes,
    input_shape = block_shape+(1,),
    receptive_field=129,
    filters=96,
    scale_factor = scale,
    dropout=None,
    batch_size= BATCH_SIZE,
    )
breakpoint()
optimizer = tf.keras.optimizers.Adam(lr=lr)
#loss_fn = tf.keras.losses.sparse_categorical_crossentropy() 
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
model.compile(optimizer, loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'], experimental_run_tf_function=False)

#training loop
train_accuracy, train_loss = [], []
valid_accuracy, valid_loss = [], []
for epoch in range(EPOCHS):
    print('Epoch number ',epoch)
    epoch_accuracy, epoch_loss, epoch_dice = [], [], []
    for steps, (batch_x,batch_y) in enumerate(dataset_train.take(1)):
        batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
        epoch_accuracy.append(batch_accuracy)
        epoch_loss.append(batch_loss)
        # calculate dice
        result = model.predict_on_batch(batch_x)
        result = np.argmax(result, -1)
        result = tf.one_hot(result, depth = n_classes)
        batch_y = tf.one_hot(batch_y, depth= n_classes)
        dice_score = generalized_dice(batch_y, result, axis=(1,2,3))
        epoch_dice.append(dice_score)
    # save checkpoint every 10 epoch      
    if epoch % 10 == 0:
        checkpoint.save(checkpoint_prefix.format(epoch=epoch))    
    print("loss:{}, accuracy:{}, dice:{}".format(tf.reduce_mean(epoch_loss),
                                    tf.reduce_mean(epoch_accuracy),
                                    tf.reduce_mean(epoch_dice)))
    
    #evaluation
    epoch_val_accuracy = [] 
    epoch_val_loss = []
    eval_dice = []
    for eval_x, eval_y in dataset_eval.take(1):
        batch_val_loss, batch_val_accuracy = model.test_on_batch(eval_x, eval_y)
        epoch_val_loss.append(batch_val_loss)
        epoch_val_accuracy.append(batch_val_accuracy)
        # calculate dice
        result = model.predict_on_batch(eval_x)
        result = np.argmax(result, -1)
        result = tf.one_hot(result, depth = n_classes)
        eval_y = tf.one_hot(eval_y, depth= n_classes)
        dice_score = generalized_dice(eval_y, result, axis=(1,2,3))
        eval_dice.append(dice_score)
    print("Eval_loss: {}, Eval_accuracy: {}, Eval_dice: {}".format(tf.reduce_mean(epoch_val_loss),
                                                    tf.reduce_mean(epoch_val_accuracy),
                                                    tf.reduce_mean(eval_dice)))   
    
# save model
saved_model_path=os.path.join("./training_files",model_name,"saved_model/")
model.save(saved_model_path, save_format='tf')
  
# test and save output
print("------------ test--------------")
test_dataset = get_dataset(train_pattern, volume_shape, BATCH_SIZE, block_shape, n_classes, training= False)
#import pdb; pdb.set_trace()
num_blocks = int((volume_shape[0]/block_shape[0])**3) 
labels = np.empty(shape = (num_blocks,*block_shape))
results = np.empty(shape = (num_blocks,*block_shape))

for batch, (feat, label) in enumerate(test_dataset.take(num_blocks)):
    pred = model(feat)
    pred = np.argmax(pred, -1)
    labels[batch,:,:,:] = label.numpy()
    results[batch,:,:,:] = pred
    
labels = from_blocks_numpy(labels, volume_shape)
results = from_blocks_numpy(results, volume_shape)
output_file = os.path.join("training_files",model_name,"output_b{}_cl{}".format(block_shape[0],n_classes))     
np.savez(output_file,label=labels,result= results)
        