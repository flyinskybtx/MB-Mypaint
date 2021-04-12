import tensorflow as tf
from tensorflow import keras

from Model import make_layer
from Model.custom_layers import SamplingLayer


class ObsEncoder(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_size = config.latent_size
        self.dist_z = config.dist_z

        self.functional_layers = []
        for i, layer_config in enumerate(config.encoder_layers, start=1):
            layer_config.number = i
            self.functional_layers.append(make_layer(layer_config))

        if self.dist_z:
            self.mu = keras.layers.Dense(self.latent_size, name='z_mu')
            self.log_sigma = keras.layers.Dense(self.latent_size, name='z_log_sigma')
            self.sigma = keras.layers.Lambda(lambda x: 0.1 + 0.9 * tf.nn.softplus(x), name='z_sigma')
            self.sampling = SamplingLayer(name='sampling')
        else:
            self.z = keras.layers.Dense(self.latent_size, name='z')

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for layer in self.functional_layers:
            x = layer(x)

        if self.dist_z:
            mu = self.mu(x)
            log_sigma = self.log_sigma(x)
            sigma = self.sigma(log_sigma)
            z = self.sampling((mu, sigma))

            if training:
                return mu, sigma, z
            else:
                return mu
        else:
            z = self.z(x)
            return z

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
