import os

from tensorflow import keras

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError


class LayerConfig(AttrDict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.setdefault(k, v)


def make_layer(layer_config: LayerConfig):
    if layer_config.type == 'conv':  # N=(W−F+2P)/S+1
        layer = keras.layers.Conv2D(name=f'Conv{layer_config.number}',
                                    filters=layer_config.filters,
                                    kernel_size=layer_config.kernel_size,
                                    strides=layer_config.setdefault('strides', 1),
                                    padding=layer_config.setdefault('padding', 'same'),
                                    activation=layer_config.setdefault('activation', None))
    elif layer_config.type == 'deconv':
        layer = keras.layers.Conv2DTranspose(name=f'Deconv{layer_config.number}',
                                             filters=layer_config.filters,
                                             kernel_size=layer_config.kernel_size,
                                             strides=layer_config.setdefault('strides', 1),
                                             padding=layer_config.setdefault('padding', 'same'),
                                             activation=layer_config.setdefault('activation', None))
    elif layer_config.type == 'dense':
        layer = keras.layers.Dense(name=f'Dense{layer_config.number}',
                                   activation=layer_config.setdefault('activation', None),
                                   units=layer_config.units)
    elif layer_config.type == 'pool':
        layer = keras.layers.MaxPooling2D(name=f'Pool{layer_config.number}',
                                          pool_size=layer_config.pool_size,
                                          strides=layer_config.setdefault('strides', 1),
                                          padding=layer_config.setdefault('padding', 'same'))
    elif layer_config.type == 'upsampling':
        layer = keras.layers.UpSampling2D(name=f'Upsample{layer_config.number}', size=layer_config.size)
    elif layer_config.type == 'reshape':
        layer = keras.layers.Reshape(name=f'Reshape{layer_config.number}', target_shape=layer_config.target_shape)
    elif layer_config.type == 'flatten':
        layer = keras.layers.Flatten(name=f'Flatten{layer_config.number}')
    elif layer_config.type == 'dropout':
        layer = keras.layers.Dropout(name=f'Dropout{layer_config.number}',
                                     rate=layer_config.rate)
    elif layer_config.type == 'batch_norm':
        layer = keras.layers.BatchNormalization(name=f'BN{layer_config.number}')
    elif layer_config.type == 'activation':
        layer = keras.layers.Activation(activation=layer_config.activation,
                                        name=f'{layer_config.activation}_{layer_config.number}')

    else:
        raise ValueError(f"Layer type of {layer_config.type} is not allowed")

    return layer
