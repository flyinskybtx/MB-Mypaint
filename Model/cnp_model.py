import tensorflow as tf
from tensorflow import keras

from Model import make_layer


class CNP(keras.Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.context_concat = keras.layers.Concatenate(name='context_concat', axis=-1)
        self.encoder_layers = []
        for i, layer_config in enumerate(config.dynamics_layers['obs_encoder'], start=1):
            layer_config.number = i
            self.encoder_layers.append(make_layer(layer_config))
        self.aggregate_layer = keras.layers.Lambda(lambda logit: tf.reduce_mean(logit, axis=1, keepdims=False),
                                                   name='avg_aggregates')
        self.query_concat = keras.layers.Concatenate(name='query_concat', axis=-1)
        self.decoder_layers = []
        for i, layer_config in enumerate(config.dynamics_layers['decoder'],
                                         start=len(config.dynamics_layers['obs_encoder']) + 1):
            layer_config.number = i
            self.decoder_layers.append(make_layer(layer_config))

        self.mu = keras.layers.Dense(config.latent_size, name='z_mu')
        self.log_sigma = keras.layers.Dense(config.latent_size, name='z_log_sigma')
        self.sigma = keras.layers.Lambda(lambda x: 0.1 + 0.9 * tf.nn.softplus(x), name='z_sigma')
        self.dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)

    def call(self, inputs, training=False, **kwargs):
        context_x, context_y, query_x = inputs
        encodes = self.context_concat([context_x, context_y])
        for layer in self.encoder_layers:
            encodes = layer(encodes)
        logits = self.aggregate_layer(encodes)
        decodes = self.query_concat([logits, query_x])
        for layer in self.decoder_layers:
            decodes = layer(decodes)

        mu = self.mu(decodes)
        log_sigma = self.log_sigma(decodes)
        sigma = self.sigma(log_sigma)
        dist_concat = self.dist_concat([mu, sigma])

        if training:
            return dist_concat

        else:
            return mu

    def train_step(self, data, **kwargs):
        X, Y = data
        with tf.GradientTape() as tape:
            y_pred = self.call(X, training=True)
            loss = self.compiled_loss(Y, y_pred, regularization_losses=self.losses, )

        grads = tape.gradient(loss, self.trainable_weights, )
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(Y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data, **kwargs):
        X, Y = data
        y_pred = self.call(X, training=True)
        loss = self.compiled_loss(Y, y_pred)
        self.compiled_metrics.update_state(Y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def build_graph(self, input_shape):
        """

        Args:
            input_shape: (list) context_x, contexy_y, query_x

        Returns:

        """
        self.build(input_shape)
        inputs = [tf.keras.Input(shape=shape[1:]) for shape in input_shape]

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
