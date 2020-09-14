import numpy as np
import tensorflow as tf

tf.executing_eagerly = True
import tensorflow_probability as tfp
from tensorflow import keras

from Data.Deprecated.core_config import experimental_config
from Model.cnn_model import build_block, LayerConfig

config = {
    'image_size': experimental_config.image_size
}


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


def build_cnp_model(config):
    context_x = keras.layers.Input(name='context_x',
                                   shape=(None, config['state_dims'] + config['action_dims']))
    context_y = keras.layers.Input(name='context_y',
                                   shape=(None, config['state_dims']))
    query_x = keras.layers.Input(name='query_x', shape=(config['state_dims'] + config['action_dims'],))

    # ------------------ latent_encoder --------------------- #
    x = keras.layers.Concatenate(name='context_concat', axis=-1)([context_x, context_y])
    for i, ec in enumerate(config['latent_encoder'], start=1):
        block = build_block(ec, i)
        for layer in block:
            x = layer(x)
    encodes = keras.layers.Dense(name='encodes', units=config['logits_dims'], activation='linear')(x)

    # ------------------ aggregator ------------------ #
    aggregates = keras.layers.Lambda(lambda logit: tf.reduce_mean(logit, axis=1, keepdims=False),
                                     name='avg_aggregates')(encodes)

    # ------------------ latent_decoder ------------------ #
    x = keras.layers.Concatenate(name='query_concat', axis=-1)([aggregates, query_x])
    for i, ec in enumerate(config['latent_decoder'], start=len(config['latent_encoder']) + 1):
        block = build_block(ec, i)
        for layer in block:
            x = layer(x)
    mu = keras.layers.Dense(name='mu', units=config['state_dims'])(x)
    log_sigma = keras.layers.Dense(name=f'log_sigma', units=config['state_dims'])(x)
    sigma = keras.layers.Lambda(lambda logit: 0.1 + 0.9 * tf.nn.softplus(logit),
                                name='sigma')(log_sigma)  # bound variance
    dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)([mu, sigma])

    # ------------------ model ----------------------- #
    cnp_model = keras.models.Model(inputs={'context_x': context_x,
                                           'context_y': context_y,
                                           'query_x': query_x},
                                   outputs={'mu': mu,
                                            'sigma': sigma,
                                            'dist_concat': dist_concat},
                                   name='cnp_model')
    return cnp_model


def test_cnp_train():
    config = {
        'state_dims': 5,
        'action_dims': 2,
        'logits_dims': 8,
        'latent_encoder': {LayerConfig(fc=8, activation='relu')},
        'latent_decoder': {LayerConfig(fc=8, activation='relu')},
    }

    cnp_model = build_cnp_model(config)
    cnp_model.compile(
        # run_eagerly=True,
        optimizer=keras.optimizers.Adam(1e-2),
        loss={'dist_concat': dist_logp},
        metrics={'dist_concat': dist_logp, 'mu': stats},
    )

    print(cnp_model.summary())
    keras.utils.plot_model(cnp_model, show_shapes=True, to_file=f'../Model/{cnp_model.name}.png')

    num_context = 20
    Xs = {'context_x': np.random.rand(20000, num_context, config['state_dims'] + config['action_dims']),
          'context_y': np.ones((20000, num_context, config['state_dims'])),
          'query_x': np.random.rand(20000, config['state_dims'] + config['action_dims'])}
    Ys = {'dist_concat': np.ones((20000, config['state_dims'])),
          'mu': np.ones((20000, config['state_dims']))}

    cnp_model.fit(
        Xs, Ys,
        epochs=1000,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5)
        ]
    )


if __name__ == '__main__':
    test_cnp_train()
