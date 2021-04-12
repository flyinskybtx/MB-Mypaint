import tensorflow as tf
from tensorflow import keras

from Model import make_layer


class ObsDecoder(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_output = config.decoder_binary_output
        self.functional_layers = []
        for i, layer_config in enumerate(config.decoder_layers, start=1):
            layer_config.number = i
            self.functional_layers.append(make_layer(layer_config))

        if self.binary_output:
            self.bw_layer = keras.layers.Lambda(lambda x: round_through(x), name='Bw_output')

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for layer in self.functional_layers:
            x = layer(x)

        if training:
            return x
        else:
            if self.binary_output:
                return self.bw_layer(x)
            else:
                return x

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)
