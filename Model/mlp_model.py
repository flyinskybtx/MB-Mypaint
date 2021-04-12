import tensorflow as tf
from tensorflow import keras

from Model import make_layer


class MLP(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.concat = keras.layers.Concatenate(axis=-1, name='latent_z')
        self.encoder_layers = []
        for i, layer_config in enumerate(config.dynamics_layers['obs_encoder'], start=1):
            layer_config.number = i
            self.encoder_layers.append(make_layer(layer_config))
        self.decoder_layers = []
        for i, layer_config in enumerate(config.dynamics_layers['decoder'],
                                         start=len(config.dynamics_layers['obs_encoder']) + 1):
            layer_config.number = i
            self.decoder_layers.append(make_layer(layer_config))

        self.mu = keras.layers.Dense(config.latent_size, activation='linear', name='z_mu')

    def call(self, inputs, training=False, **kwargs):
        x = self.concat(inputs)
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.mu(x)
        return x

    def build_graph(self, input_shape):
        self.build(input_shape)
        inputs = [tf.keras.Input(shape=shape[1:]) for shape in input_shape]

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
