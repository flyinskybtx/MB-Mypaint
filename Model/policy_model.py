import tensorflow as tf
from ray.rllib.models.tf import TFModelV2
from tensorflow import keras

from Model import make_layer


class CustomPolicyModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, layers):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        image_input = keras.layers.Input(shape=obs_space.shape, name="image_input")
        x = image_input

        for i, layer_config in enumerate(layers, start=1):
            layer_config.number = i
            x = make_layer(layer_config)(x)

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


