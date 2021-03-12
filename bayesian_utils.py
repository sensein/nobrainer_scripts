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
