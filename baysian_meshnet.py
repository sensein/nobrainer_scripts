"""
Created on Thu Feb 20 10:08:29 2021

@author: aakanksha (Version Adapted from the previous Nobrainer-Variational_Meshnet Model)
"""

"""Implementations of Bayesian neural networks."""

import tensorflow as tf
import tensorflow_probability as tfp

from nobrainer.layers.dropout import BernoulliDropout
from nobrainer.layers.dropout import ConcreteDropout
from bayesian_utils import normal_prior

tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers
weightnorm = tfp.layers.weight_norm.WeightNorm # This does not work with TFP ConvVariational layers 
tfd = tfp.distributions
def variational_meshnet(
    n_classes,
    input_shape,
    receptive_field=67,
    filters=71,
    scale_factor = scale
    is_monte_carlo=False,
    dropout=None,
    activation=tf.nn.relu,
    batch_size=None,
    name="variational_meshnet",
):
    """Instantiate variational MeshNet model.

    Please see https://arxiv.org/abs/1805.10863 for more information.

    Parameters
    ----------
    n_classes: int, number of classes to classify. For binary applications, use
        a value of 1.
    input_shape: list or tuple of four ints, the shape of the input data. Omit
        the batch dimension, and include the number of channels.
    receptive_field: {37, 67, 129}, the receptive field of the model. According
        to the MeshNet manuscript, the receptive field should be similar to your
        input shape. The actual receptive field is the cube of the value provided.
    filters: int, number of filters per volumetric convolution. The original
        MeshNet manuscript uses 21 filters for a binary segmentation task
        (i.e., brain extraction) and 71 filters for a multi-class segmentation task.
    activation: str or optimizer object, the non-linearity to use.
    scale_factor: A tf float 32 variable to scale up the KLD loss.  
    is_monte_carlo: bool, only Related to dropout version! 
    dropout: string, type of dropout layer.
    batch_size: int, number of samples in each batch. This must be set when
        training on TPUs.
    name: str, name to give to the resulting model object.

    Returns
    -------
    Model object.

    Raises
    ------
    ValueError if receptive field is not an allowable value.
    """

    if receptive_field not in {37, 67, 129}:
        raise ValueError("unknown receptive field. Legal values are 37, 67, and 129.")
    
    def one_layer(x, layer_num, scale_factor,dilation_rate=(1, 1, 1)):
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(scale_factor, dtype=tf.float32))
        x = tfpl.Convolution3DFlipout(filters,
            kernel_size=3, padding="same",dilation_rate=dilation_rate,
            kernel_prior_fn=prior_fn,
            activation=activation, 
            kernel_divergence_fn=kl_divergence_function, 
            name="layer{}/vwnconv3d".format(layer_num),)(x)
        if dropout is None:
            pass
        elif dropout == "bernoulli":
            x = BernoulliDropout(
                rate=0.5,
                is_monte_carlo=is_monte_carlo,
                scale_during_training=False,
                name="layer{}/bernoulli_dropout".format(layer_num),
            )(x)
        elif dropout == "concrete":
            x = ConcreteDropout(
                is_monte_carlo=is_monte_carlo,
                temperature=0.02,
                use_expectation=is_monte_carlo,
                name="layer{}/concrete_dropout".format(layer_num),
            )(x)
        else:
            raise ValueError("unknown dropout layer, {}".format(dropout))
        x = tfkl.Activation(activation, name="layer{}/activation".format(layer_num))(x) # This activation makes no sense if Dropout is NONE! 
        return x

    inputs = tfkl.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    prior_fn = normal_prior(prior_mu = 0, prior_std = 1.0)
    
    if receptive_field == 37:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3)
        x = one_layer(x, 4, dilation_rate=(2, 2, 2))
        x = one_layer(x, 5, dilation_rate=(4, 4, 4))
        x = one_layer(x, 6, dilation_rate=(8, 8, 8))
        x = one_layer(x, 7)
    elif receptive_field == 67:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2)
        x = one_layer(x, 3, dilation_rate=(2, 2, 2))
        x = one_layer(x, 4, dilation_rate=(4, 4, 4))
        x = one_layer(x, 5, dilation_rate=(8, 8, 8))
        x = one_layer(x, 6, dilation_rate=(16, 16, 16))
        x = one_layer(x, 7)
    elif receptive_field == 129:
        x = one_layer(inputs, 1)
        x = one_layer(x, 2, dilation_rate=(2, 2, 2))
        x = one_layer(x, 3, dilation_rate=(4, 4, 4))
        x = one_layer(x, 4, dilation_rate=(8, 8, 8))
        x = one_layer(x, 5, dilation_rate=(16, 16, 16))
        x = one_layer(x, 6, dilation_rate=(32, 32, 32))
        x = one_layer(x, 7)

    x = tfpl.Convolution3DFlipout(
        filters=n_classes,
        kernel_size=1, 
	kernel_prior_fn=prior_fn,
        padding="same",
        name="classification/vwnconv3d",
    )(x)

    final_activation = "sigmoid" if n_classes == 1 else "softmax"
    x = tfkl.Activation(final_activation, name="classification/activation")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
