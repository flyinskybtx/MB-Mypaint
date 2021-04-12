# Model
import random
import string

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from tensorflow import keras


class LayerConfig:
    def __init__(self, **kwargs):
        self.conv = kwargs.setdefault('conv', None)
        self.batch_norm = kwargs.setdefault('conv', False)
        self.activation = kwargs.setdefault('activation', None)
        self.pool = kwargs.setdefault('pool', None)
        self.dropout = kwargs.setdefault('dropout', False)
        self.fc = kwargs.setdefault('fc', False)
        self.flatten = kwargs.setdefault('flatten', False)
        self.padding = kwargs.setdefault('padding', 'valid')


def build_block(config: LayerConfig, number):
    layers = []
    if config.conv:
        layers.append(keras.layers.Conv2D(filters=config.conv[0], kernel_size=config.conv[1], padding=config.padding,
                                          strides=config.conv[
                                              2], name=f'Conv_{number}'))

        if config.batch_norm:
            layers.append(keras.layers.BatchNormalization(name=f'BN_{number}'))
        if config.activation:
            layers.append(keras.layers.Activation(activation=config.activation, name=f'{config.activation}_{number}'))
        if config.pool:
            layers.append(keras.layers.MaxPooling2D(pool_size=config.pool[0], strides=config.pool[1],
                                                    padding=config.pool[2], name=f'Pool_{number}'))
        if config.dropout:
            layers.append(keras.layers.Dropout(config.dropout, name=f'Drop_{number}'))

    if config.fc:
        layers.append(keras.layers.Dense(config.fc, activation=config.activation, name=f'Dense_{number}'))

        if config.dropout:
            layers.append(keras.layers.Dropout(config.dropout, name=f'Drop_{number}'))

    if config.flatten:
        layers.append(keras.layers.Flatten(name=f'Flatten_{number}'))

    return layers


class CnnModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        image_input = keras.layers.Input(shape=(obs_space.shape), name="image_input")
        x = image_input
        blocks_configs = model_config['custom_model_config']['blocks']
        for i, bc in enumerate(blocks_configs):
            block = build_block(bc, i)
            for layer in block:
                x = layer(x)

        logits = keras.layers.Dense(self.num_outputs, activation='linear', name="logits")(x)
        values = keras.layers.Dense(1, activation=None, name="values")(x)

        ## Create Model
        self.base_model = keras.Model(inputs=image_input,
                                      outputs=[logits, values])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, value_out = self.base_model(input_dict['obs'])
        self._value_out = value_out
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, import_file):
        # Override this to define custom weight loading behavior from h5 files.
        self.base_model.load_weights(import_file)


if __name__ == '__main__':
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, CnnModel)

    model_config = {
        'custom_model': model_name,
        "custom_model_config": {
            'blocks': [
                LayerConfig(conv=[16, [5, 5], 3], batch_norm=True, activation='relu', pool=[2, 2, 'same'], dropout=0.3),
                LayerConfig(conv=[32, [3, 3], 1], activation='relu', pool=[2, 2, 'same']),
                LayerConfig(flatten=True),
                LayerConfig(fc=64, activation='linear', dropout=0.2),
            ]
        },
    }

    observation_space = gym.spaces.Box(low=0,
                                       high=1,
                                       shape=(192, 192, 1),
                                       dtype=np.float)
    action_space = gym.spaces.MultiDiscrete([28, 28, 10] * 6)
    num_outputs = 18
    name = 'test_custom_cnn'

    model = CnnModel(observation_space, action_space, num_outputs, model_config, name)
