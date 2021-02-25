import tensorflow as tf
from nobrainer.models.bayesian import variational_meshnet
from nobrainer.volume import from_blocks_numpy
from nobrainer.metrics import generalized_dice
from kwyk_data import get_dataset
import numpy as np
import time

# constants
#root_path = '/om/user/satra/kwyk/tfrecords/'
#root_path = '/om2/user/hodaraja/kwyk/nobrainer_scripts/'
#root_path = "data/"
# for satori run
root_path = "/nobackup/users/abizeul/kwyk/tfrecords/"

train_pattern = root_path +'data-train_shard-*.tfrec'
eval_pattern = root_path +"data-evaluate_shard-*.tfrec"

n_classes =115
volume_shape = (256, 256, 256)
block_shape = (32,32,32)
BATCH_SIZE = 1
n_examples = BATCH_SIZE
    
dropout_typ = 'concrete'
BATCH_SIZE = 1     
EPOCHS = 20
lr = 1e-04

print("--------- Loading data -------------")
dataset_train = get_dataset(train_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes)
dataset_eval = get_dataset(eval_pattern,volume_shape, BATCH_SIZE, block_shape, n_classes)

# create model 
model = variational_meshnet(n_classes = n_classes,
                            input_shape = (*block_shape,1),
                            receptive_field = 129,
                            filters = 96,
                            is_monte_carlo=True,
                            dropout = dropout_typ,
                            )

print("===================> model created")

# optimizer
optimizer = tf.keras.optimizers.Adam(lr)

def elbo(y_true, y_pred, num_examples, from_logits=False):
    """Labels should be integers in `[0, n)`."""
    scc_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    neg_log_likelihood = -scc_fn(y_true, y_pred)
    kl = sum(model.losses) / num_examples
    elbo_loss = neg_log_likelihood + kl
    return elbo_loss

def accuracy(preds, labels):
    return np.mean((np.argmax(preds,-1) == labels.numpy()))

def calculate_dice(labels, preds):
    labels= tf.one_hot(labels)
    preds = tf.one_hot(np.argmax(preds,-1))
    return generalized_dice(labels,preds, axis=(1,2,3))

# define train step
@tf.function
def train_step(feats, labels):
    with tf.GradientTape() as tape:
        predictions = model(feats, training=True)
        loss = elbo(labels, predictions, n_examples, model)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss

@tf.function
def eval_step(feats, labels):
    
    predictions = model(feats, training=False)
    e_loss = elbo(labels, predictions, n_examples, model)
    
    return predictions, e_loss


times = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
dices = []
tic=time.time()
# training loop
print("-------- training----------")
for epoch in range(EPOCHS):
    print("Epoch: {}".format(epoch))
    # reset metrics
    epoch_train_loss = []
    epoch_train_accuracy = []
    epoch_eval_loss = []
    epoch_eval_accuracy = []
    epoch_dice =[]
    
    for step, (feats, labels) in enumerate(dataset_train):
        preds, loss = train_step(feats,labels)
        
        epoch_train_loss.append(loss.numpy())
        acc = accuracy(preds, labels)
        epoch_train_accuracy.append(acc)
        epoch_dice = calculate_dice(labels, preds)
        
    for val_img, val_label in dataset_eval:
        val_preds, val_loss= eval_step(val_img,val_label)
        
        epoch_eval_loss.append(val_loss.numpy())
        val_acc= accuracy(val_preds, val_label)
        epoch_eval_accuracy.append(val_acc)
   
    train_losses.append(np.mean(epoch_train_loss))
    train_accs.append(np.mean(epoch_train_accuracy))
    val_losses.append(np.mean(epoch_eval_loss))
    val_accs.append(np.mean(epoch_eval_accuracy))
    
    print("Loss: {:.3f}, Accuracy: {:.3}, Eval-loss: {:.3}, Eval-accuracy: {:.3f}".format(np.mean(epoch_train_loss),
                                                                                         np.mean(epoch_train_accuracy),
                                                                                         np.mean(epoch_eval_loss),
                                                                                         np.mean(epoch_eval_accuracy)
                                                                                         ))

# test the model
print("------------ test--------------")

test_dataset = get_dataset(train_pattern, volume_shape, BATCH_SIZE, block_shape, n_classes, training= False)
#import pdb; pdb.set_trace()
num_blocks = int((volume_shape[0]/block_shape[0])**3) 
labels = np.empty(shape = (num_blocks,*block_shape))
results = np.empty(shape = (num_blocks,*block_shape))

for batch, (feat, label) in enumerate(test_dataset):
    pred = model(feat)
    pred = np.argmax(pred, -1)
    labels[batch,:,:,:] = label.numpy()
    results[batch,:,:,:] = pred
    
labels = from_blocks_numpy(labels, volume_shape)
results = from_blocks_numpy(results, volume_shape)      
#import pdb; pdb.set_trace()
np.savez("output_var_b32",label=labels,result= results)