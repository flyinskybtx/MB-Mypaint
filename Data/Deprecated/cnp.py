import numpy as np
import tensorflow_probability as tfp
from ray.rllib.utils import try_import_tf
from tensorflow import keras

tf = try_import_tf()


def build_cnp(x_shape, y_shape, encoder_hiddens, decoder_hiddens):
    # --- Encoder ---
    context_x = keras.layers.Input(shape=(None, x_shape), name='context_x')
    context_y = keras.layers.Input(shape=(None, y_shape), name='context_y')
    encoder_input = keras.layers.concatenate(inputs=[context_x, context_y], axis=-1)
    y = encoder_input

    for i, size in enumerate(encoder_hiddens[:-1], 1):
        y = keras.layers.Dense(size, name=f'encoder_{i}', activation='relu')(y)  # MLP
    y = keras.layers.Dense(encoder_hiddens[-1], name=f'dense_{len(encoder_hiddens)}', activation='linear')(y)

    # --- cnp_aggregator ---
    y = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=False),
                            name='avg_aggregator')(y)

    # --- Decoder ---
    query_x = keras.layers.Input(shape=(x_shape,), name='query_x')
    y = keras.layers.concatenate(inputs=[y, query_x], axis=-1)
    # ys = query_x

    for i, size in enumerate(decoder_hiddens, 1):
        y = keras.layers.Dense(size, name=f'decoder_{i}', activation='relu')(y)  # MLP

    mu = keras.layers.Dense(y_shape, name=f'mu')(y)
    log_sigma = keras.layers.Dense(y_shape, name=f'log_sigma')(y)
    sigma = keras.layers.Lambda(lambda x: 0.1 + 0.9 * tf.nn.softplus(x),
                                name='sigma')(log_sigma)  # bound variance

    dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)([mu, sigma])

    model = keras.models.Model(inputs=[context_x, context_y, query_x], outputs=[mu, sigma, dist_concat])
    #
    return model


def dist_loss(y_true, y_pred):
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


if __name__ == '__main__':
    # cnp_model = CNPModel([16], [8], 3)
    # cnp_model.build(input_shape=[(10, 7), (10, 5), (1, 7)])
    # cnp_model.summary()
    cnp_model = build_cnp(7, 5, [8], [8])
    cnp_model.compile(loss={'dist_concat': dist_loss, },
                      metrics=[dist_mse]
                      )
    cnp_model.summary()

    Xs = {
        'context_x': np.random.rand(20000, 11, 7),
        'context_y': np.random.rand(20000, 11, 5),
        'query_x': np.random.rand(20000, 7),
    }
    Ys = {
        'dist_concat': np.random.rand(20000, 5),
    }

    cnp_model.fit(Xs, Ys, batch_size=10, epochs=100)

    seed = 0
    test_X = {
        'context_x': Xs['context_x'][0:1, :, :],
        'context_y': Xs['context_y'][0:1, :, :],
        'query_x': Xs['query_x'][0:1, :],
    }
    test_Y = {
        'dist_concat': Ys['dist_concat'][0:1, :]
    }
    test_y_ = cnp_model.predict(test_X)
    print(f'y_true: {test_Y}')
    print(f'y_pred: {test_y_}')
