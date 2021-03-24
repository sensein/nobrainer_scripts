#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:38:56 2021

@author: hoda
"""

import tensorflow as tf
import tensorflow_probability as tfp


def normal_prior(prior_mu, prior_std):
    """Defines distributions prior for Bayesian neural network.
       Simply set tf.zeros(shape, dtype) with a new mu value for any new distribution  as needed
       I tried Normal, He, Xavier. 
    """
    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return prior_fn

def xavier_prior(prior_mu, prior_std):
    """Defines distributions prior for Bayesian neural network.
       Simply set tf.zeros(shape, dtype) with a new mu value for any new distribution  as needed
       I tried Normal, He, Xavier. 
    """
    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        xavier = tf.keras.initializers.GlorotNormal()
        dist = tfd.Normal(loc=xavier(shape=shape),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return prior_fn

def He_prior(prior_mu, prior_std):
    """Defines distributions prior for Bayesian neural network.
       Simply set tf.zeros(shape, dtype) with a new mu value for any new distribution  as needed
       I tried Normal, He, Xavier. 
    """
    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        he = tf.keras.initializers.HeNormal()
        dist = tfd.Normal(loc=he(shape=shape),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return prior_fn


def get_sample_weight(y_true, n_classes):
    # check if the input is one_hot_encoded
    if y_true.shape[-1] == 1:
        y_true = tf.one_hot(y_true, depth = n_classes)
    mask = 1- y_true[:,:,:,:,0]
    return mask

