from os import path as osp

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from Model import AttrDict, MODEL_DIR
from Model.basics import make_layer


def dist_logp(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)
    log_p = dist.log_prob(y_true)
    loss = -tf.reduce_mean(log_p)
    return loss


def dist_mse(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    mse = tf.reduce_mean(tf.square(tf.reduce_mean(mu - y_true)))
    return mse


def stats(y_true, y_pred):
    return tf.reduce_mean(y_pred, keepdims=False) - tf.reduce_mean(y_true, keepdims=False)


class ActionEmbedder:
    def __init__(self, config):
        self.action_shape = config.action_shape

    def _embedding(self, action):
        embedding = np.zeros(3 * self.action_shape)
        for i, act in enumerate(action):
            embedding[i * self.action_shape + int(act)] = 1
        return embedding

    def transform(self, actions):
        return np.stack(self._embedding(action) for action in actions)


class CNPModel:
    def __init__(self, config: AttrDict):
        self.config = config
        self.state_dims = config.latent_size
        self.action_dims = 3 * self.config.action_shape
        self.embedder = None
        self.repr_model = None

        self.build_model()

    def build_model(self):
        context_x = keras.layers.Input(name='context_x',
                                       shape=(None, self.state_dims + self.action_dims))
        context_y = keras.layers.Input(name='context_y',
                                       shape=(None, self.state_dims))
        query_x = keras.layers.Input(name='query_x', shape=(self.state_dims + self.action_dims,))

        # ------------------ encoder --------------------- #
        encodes = keras.layers.Concatenate(name='context_concat', axis=-1)([context_x, context_y])
        for i, layer_config in enumerate(self.config.dynamics_layers['encoder'], start=1):
            layer_config.number = i
            layer = make_layer(layer_config)
            encodes = layer(encodes)
        # ------------------ aggregator ------------------ #
        aggregates = keras.layers.Lambda(lambda logit: tf.reduce_mean(logit, axis=1, keepdims=False),
                                         name='avg_aggregates')(encodes)
        # ------------------ decoder ------------------ #
        decodes = keras.layers.Concatenate(name='query_concat', axis=-1)([aggregates, query_x])
        for i, layer_config in enumerate(self.config.dynamics_layers['decoder'],
                                         start=len(self.config.dynamics_layers['encoder']) + 1):
            layer_config.number = i
            layer = make_layer(layer_config)
            decodes = layer(decodes)
        mu = keras.layers.Dense(name='mu', units=self.state_dims, activation='linear')(decodes)
        log_sigma = keras.layers.Dense(name=f'log_sigma', units=self.state_dims)(decodes)
        sigma = keras.layers.Lambda(lambda logit: 0.1 + 0.9 * tf.nn.softplus(logit),
                                    name='sigma')(log_sigma)  # bound variance
        dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)([mu, sigma])

        # ------------------ model ----------------------- #
        self.model = keras.models.Model(inputs={'context_x': context_x,
                                                'context_y': context_y,
                                                'query_x': query_x},
                                        outputs={'mu': mu,
                                                 'sigma': sigma,
                                                 'dist_concat': dist_concat},
                                        name='Dynamics_cnp')
        print(f"Created model {self.model.name}")

    def train_model(self, train_generator, vali_generator, epochs=100, ):
        self.model.compile(
            # run_eagerly=True,
            optimizer=keras.optimizers.Adam(5e-5),
            loss={'dist_concat': dist_logp},
            metrics={'dist_concat': dist_mse, 'mu': stats},
        )

        train_data = train_generator
        train_data.train_model = True
        vali_data = vali_generator
        vali_data.train_model = False

        self.model.fit(
            train_data, epochs=epochs,
            # steps_per_epoch=100,
            validation_data=vali_data, validation_steps=20,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_dist_concat_loss',
                    patience=5, mode='auto',
                    restore_best_weights=True),
            ]
        )

    def save_model(self):
        self.model.save_weights(osp.join(MODEL_DIR, f'checkpoints/{self.model.name}.h5'))
        print(f'Saved dynamics model to checkpoints/{self.model.name}')

    def load_model(self):
        self.model.load_weights(osp.join(MODEL_DIR, f'checkpoints/{self.model.name}.h5'),
                                # custom_objects={'dist_logp': dist_logp,
                                #                 'dist_mse': dist_mse,
                                #                 'stats': stats}
                                )
        print(f'Loaded CNP model from checkpoints/{self.model.name}')

    def set_context(self, context_x, context_y):
        print(f"Set {context_x.shape[0]} context points for dynamics")
        # check shape
        assert context_x.shape[0] == context_y.shape[0], "Number of contexts not match"
        assert context_y.shape[-1] == self.state_dims
        assert context_x.shape[-1] == self.state_dims + self.action_dims

        self.context_x = np.expand_dims(context_x, axis=0)
        self.context_y = np.expand_dims(context_y, axis=0)

    def set_repr(self, repr_model, embedder):
        self.repr_model = repr_model
        self.embedder = embedder

    def predict(self, query_x):
        batch_size = query_x.shape[0]
        if batch_size > 1:
            query = {
                'context_x': np.repeat(self.context_x, batch_size, axis=0),
                'context_y': np.repeat(self.context_y, batch_size, axis=0),
                'query_x': query_x
            }
        else:
            query = {
                'context_x': self.context_x,
                'context_y': self.context_y,
                'query_x': query_x
            }
        target_y = self.model.predict(query)
        # target_y = {name: pred for name, pred in zip(self.model.output_names, target_y)}
        return target_y
