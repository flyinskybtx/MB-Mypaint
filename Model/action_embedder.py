from tensorflow import keras

from Model.custom_layers import ActionEmbedLayer


class ActionEmbedder(keras.Model):
    def get_config(self):
        pass

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_shape = config.action_shape
        self.embedder = ActionEmbedLayer(config)

    def call(self, inputs, **kwargs):
        return self.embedder(inputs)


