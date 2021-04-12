import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras


class SamplingLayer(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample zs, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        mu, sigma = inputs
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return dist.sample(name="sample")


class ActionEmbedLayer(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.action_shape = config.action_shape

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.int32)
        one_hot = tf.one_hot(inputs, self.action_shape)
        shape = tf.concat([tf.shape(one_hot)[:-2], [3 * self.action_shape]], axis=0)
        one_hot = tf.reshape(one_hot, shape)
        return one_hot
