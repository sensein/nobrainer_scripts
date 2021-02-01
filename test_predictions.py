import tensorflow as tf
from run_kwyk_mirror_trainbatch import get_dataset
from nobrainer.losses import ELBO
from nobrainer.metrics import dice
import numpy as np



if __name__ == "__main__":

    # Constants
    root_path='/om/user/satra/kwyk/tfrecords/'
    eval_pattern=root_path+'data-evaluate_shard-*.tfrec'
    #model_path="saved_model/"
    model_path="/om2/user/hodaraja/kwyk/nobrainer_scripts/training_files/kwyk_4gpu_21-01-03_01-18/saved_model"

    n_classes =115
    volume_shape = (256, 256, 256)
    batch_size=1
    block_shape=(128,128,128)

    # prepare the evaluation dataset
    dataset_evaluate=get_dataset(pattern=eval_pattern, 
                                volume_shape=volume_shape, 
                                batch=batch_size, 
                                block_shape=block_shape, 
                                n_classes= n_classes, 
                                train=False)
    # Load the saved model
    model=tf.keras.models.load_model(model_path, compile=False)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = ELBO(model=model, num_examples=np.prod(block_shape),reduction=tf.keras.losses.Reduction.NONE)
    model.compile(loss=loss_fn,optimizer=optimizer,experimental_run_tf_function=False)
    
    #dice_scores=[]
    i=0
    for data in dataset_evaluate.take(10):
        i +=1
        #result=predict(data,model_path, block_shape=block_shape, n_samples=1)
        result = model.predict_on_batch(data)
        eval_error = model.test_on_batch(data)
        #print("batch {}, predicted value {}".format(i,result))
        (feat, label) = data
        label = tf.one_hot(label, depth= n_classes)
        dice_score = tf.reduce_mean(dice(label,result,axis=(1,2,3)))
        #print("batch {}, actual label {}".format(i,result))
        print("batch {}, eval loss {},dice score {}".format(i, eval_error, dice_score))


