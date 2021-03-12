#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:48:48 2021

@author: hoda and Aakanksha 
"""

import tensorflow as tf
import nobrainer
from nobrainer.metrics import generalized_dice
from tensorflow.python.keras.utils.losses_utils import ReductionV2
from tensorflow.python.keras.losses import LossFunctionWrapper

# def nll_l2(y_true, y_pred):
#     scc_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
#     nll = scc_fn(y_true, y_pred)
#     l2 = tf.nn.l2_loss()
    
def dice_loss(y_true, y_pred, axis=(1,2,3,4)):
    return 1-generalized_dice(y_true, y_pred, axis=(1,2,3,4))

class DiceLoss(LossFunctionWrapper):
    def __init__(self, axis=(1, 2, 3, 4), reduction=ReductionV2.AUTO, name="dice"):
        super().__init__(dice_loss, axis=axis, reduction=reduction, name=name)
        

def diceandmse(y_true, y_pred,axis=(1, 2, 3, 4)):
    d = 1.0 - nobrainer.metrics.dice(y_true=y_true, y_pred=y_pred, axis=axis)
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return d + mse


class DiceandMse(LossFunctionWrapper):
    """Computes one minus the Dice and MSE similarity between labels and predictions.
    ```
    """

    def __init__(self, axis=(1, 2, 3, 4), reduction=ReductionV2.AUTO, name="dice"):
        super().__init__(diceandmse, axis=axis, reduction=reduction, name=name)


def diceandbce(y_true, y_pred,axis=(1, 2, 3, 4)):
    d = 1.0 - nobrainer.metrics.dice(y_true=y_true, y_pred=y_pred, axis=axis)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 10*(d + bce)


class DiceandBce(LossFunctionWrapper):
    """Computes one minus the Dice similarity between labels and predictions.
    """

    def __init__(self, axis=(1, 2, 3, 4), reduction=ReductionV2.AUTO, name="dice"):
        super().__init__(diceandbce, axis=axis, reduction=reduction, name=name)


###### noberainer implementation of elbo loss
def elbo(y_true, y_pred, model, num_examples, from_logits=False):
    """Labels should be integers in `[0, n)`."""
    scc_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    neg_log_likelihood = -scc_fn(y_true, y_pred)
    kl = sum(model.losses) / num_examples
    elbo_loss = neg_log_likelihood + kl
    return elbo_loss

class ELBO(LossFunctionWrapper):
    """Loss to minimize Evidence Lower Bound (ELBO).
    Use this loss for multiclass variational segmentation.
    Labels should not be one-hot encoded.
    """

    def __init__(
        self,
        model,
        num_examples,
        from_logits=False,
        reduction=ReductionV2.AUTO,
        name="ELBO",
    ):
        super().__init__(
            elbo,
            model=model,
            num_examples=num_examples,
            from_logits=from_logits,
            name=name,
            reduction=reduction,
        )
        



