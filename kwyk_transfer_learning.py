#!/usr/bin/env python3

# Imports
#import nobrainer
import tensorflow as tf
import os
from kwyk_data import get_dataset
from baysian_meshnet import variational_meshnet
from kwyk_losses import MaskedCategoricalCrossEntropy#,Dice_Cce,DiceLoss
from kwyk_utils import save_parameters, save_output, calcualte_dice

# constants
#root_path = '/om/user/satra/kwyk/tfrecords/'
#root_path = '/om2/user/hodaraja/kwyk/nobrainer_scripts/'
root_path = "data/"
# to run the code on Satori
#root_path = "/nobackup/users/abizeul/kwyk/tfrecords/"

train_pattern = root_path+"single_volume-000.tfrec"
eval_pattern = root_path + "single_volume-000.tfrec"

#train_pattern = root_path+"data-train_shard-*.tfrec"
#eval_pattern = root_path + "data-evaluate_shard-*.tfrec"

n_classes =115
volume_shape = (256, 256, 256)
block_shape = (32,32,32)
       
EPOCHS = 200
lr = 1e-06
BATCH_SIZE = 2
num_training_brains = 1
num_examples = int(((volume_shape[0]/block_shape[0])**3)* num_training_brains/BATCH_SIZE)
#num_examples = int(((volume_shape[0]/block_shape[0])**3)*num_training_brains)
#num_examples=1
one_hot_label=True

initial_epoch = 0 ; scaling_start_epoch=5 ; scaling_increase_per_epoch = 1
#scaling_end_epoch = scaling_start_epoch + np.ceil(1/scaling_increase_per_epoch)
warmup_factor=0

model_name = "kwyk_transfer_nfzlyr_Mcce_kl_b{}_cl{}".format(block_shape[0], n_classes)
checkpoint_dir = os.path.join("training_files",model_name,"training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

print("--------- Loading data -------------")
dataset_train = get_dataset(train_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes,
                            one_hot_label=one_hot_label,
                            filter_background=True)
dataset_eval = get_dataset(eval_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes, 
                           training= False,
                           one_hot_label=one_hot_label)
print("_________ data loaded with block_size: {}, batch_size: {}___________".format(block_shape[0], BATCH_SIZE)) 

if initial_epoch >= scaling_start_epoch:
    warmup_factor = tf.convert_to_tensor(min(1., warmup_factor + (initial_epoch - scaling_start_epoch) * scaling_increase_per_epoch))
kl_beta=tf.Variable(warmup_factor, dtype=tf.float32)

# instanciate the model
model = variational_meshnet(
        n_classes = n_classes,
        input_shape = block_shape+(1,),
        receptive_field=37,
        filters=96,
        scale_factor = num_examples,
        dropout= "concrete",
        batch_size= BATCH_SIZE,
        warmup_factor=kl_beta,
        )
# load weights
weights_path = "./training_files/old_kwyk_weights/kwyk_b32_cl115_weights.hd5/"
model.load_weights(weights_path) 

# freeze trained layers
# for layer in model.layers[:-2]:
#     layer.trainable=False
    
optimizer = tf.keras.optimizers.Adam(lr=lr) 
#loss_fn = DiceLoss(axis=(1,2,3))
loss_fn = MaskedCategoricalCrossEntropy()
#loss_fn = Dice_Cce(axis =(1,2,3), ignore_background = True)
#loss_fn = tf.keras.losses.CategoricalCrossentropy()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
model.compile(optimizer, loss=loss_fn,
                 metrics=['categorical_accuracy'],
                 experimental_run_tf_function=False)

#training loop
train_accuracy, train_loss = [], []
valid_accuracy, valid_loss = [], []
for epoch in range(EPOCHS):
    print('Epoch number ',epoch)
    epoch_accuracy, epoch_loss, epoch_dice = [], [], []
    for steps, (batch_x,batch_y) in enumerate(dataset_train.take(num_examples)):
        batch_loss, batch_accuracy = model.train_on_batch(batch_x, batch_y)
        epoch_accuracy.append(batch_accuracy)
        epoch_loss.append(batch_loss)
        # calculate dice
        result = model.predict_on_batch(batch_x)            
        dice_score = calcualte_dice(batch_y,result,n_classes,axis=(1,2,3),one_hot_label=one_hot_label)
        epoch_dice.append(dice_score)
        
    # save checkpoint and output every 10 epoch      
    if epoch % 10 == 0:
        checkpoint.save(checkpoint_prefix.format(epoch=epoch))
        output_path = "./training_files/" + model_name + "/out_epoch-{}".format(epoch)
        save_output(output_path, model, dataset_eval, volume_shape, block_shape, one_hot_label=one_hot_label)
        save_parameters(output_path+"_prm.out",model_name,loss=tf.reduce_mean(epoch_loss).numpy().tolist(), 
                        accuracy=tf.reduce_mean(epoch_accuracy).numpy().tolist(),
                        dice=tf.reduce_mean(epoch_dice).numpy().tolist())
        
    print("loss:{}, accuracy:{}, dice:{}".format(tf.reduce_mean(epoch_loss),
                                    tf.reduce_mean(epoch_accuracy),
                                    tf.reduce_mean(epoch_dice)))
    
    #adjusting the warmup factor
    if epoch >= scaling_start_epoch:
            new_warmup_factor = tf.convert_to_tensor(min(1., warmup_factor + (epoch - scaling_start_epoch) * scaling_increase_per_epoch), dtype=tf.float32)
            kl_beta.assign(new_warmup_factor)
            print("epoch {}, new kl_factor {}".format(epoch, kl_beta.numpy()))
            
    #evaluation
    epoch_val_accuracy = [] 
    epoch_val_loss = []
    eval_dice = []
    for eval_x, eval_y in dataset_eval.take(num_examples):
        batch_val_loss, batch_val_accuracy = model.test_on_batch(eval_x, eval_y)
        epoch_val_loss.append(batch_val_loss)
        epoch_val_accuracy.append(batch_val_accuracy)
        # calculate dice
        result = model.predict_on_batch(eval_x)
        dice_score = calcualte_dice(eval_y, result, n_classes, axis=(1,2,3), one_hot_label=one_hot_label)
        eval_dice.append(dice_score)
    print("Eval_loss: {}, Eval_accuracy: {}, Eval_dice: {}".format(tf.reduce_mean(epoch_val_loss),
                                                    tf.reduce_mean(epoch_val_accuracy),
                                                    tf.reduce_mean(eval_dice))) 
    
# save model
#saved_model_path=os.path.join("./training_files",model_name,"saved_model/")
#model.save(saved_model_path, save_format='tf')
saved_weight_path=os.path.join("./training_files",model_name,"model_weights.hd5/")
model.save_weights(saved_weight_path)
saved_param_path = os.path.join("./training_files",model_name,"model_parameters.json")
save_parameters(saved_param_path,model_name,
                block_shape = block_shape,
                batch_size = BATCH_SIZE,
                n_classes = n_classes,
                lr = lr,
                n_epochs = EPOCHS,
                num_training_brains = num_training_brains,
                loss_fn = loss_fn.name,
                kl_start_epoch = scaling_start_epoch,
                one_hot_label = one_hot_label
                )
  
# test and save output
print("------------ test--------------")
test_dataset = get_dataset(train_pattern, volume_shape, BATCH_SIZE, block_shape, n_classes, training= False)
output_file = os.path.join("training_files",model_name,"output_test_b{}_cl{}".format(block_shape[0],n_classes))
save_output(output_file,model,test_dataset,volume_shape,block_shape)
