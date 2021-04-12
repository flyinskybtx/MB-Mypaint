import tensorflow as tf
from tensorflow import keras

from Model.custom_layers import SamplingLayer


def kl_divergence(mu1, logvar1, mu2, logvar2):
    pass


def kl_divergence_to_normal(mu1, logvar1):
    pass


class MlpApp(keras.Model):
    def __init__(self, z_size):
        super().__init__()
        self.z_size = z_size

        self.concat = keras.layers.Concatenate(axis=-1)
        self.hidden_layers = [
            keras.layers.Dense(256),
            keras.layers.Dense(128),
        ]

        self.z = keras.layers.Dense(self.z_size, name='z_mu')
        self.log_sigma = keras.layers.Dense(self.z_size, name='z_log_sigma')
        self.sigma = keras.layers.Lambda(lambda x: 0.1 + 0.9 * tf.nn.softplus(x), name='z_sigma')
        self.sampling = SamplingLayer(name='sampling')

    def __call__(self, inputs, training=False, **kwargs):
        x1, x2 = inputs
        x = self.concat([x1, x2])
        for layer in self.hidden_layers:
            x = layer(x)

        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = self.sigma(log_sigma)
        z = self.sampling((mu, sigma))

        return mu, sigma, z

    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)


class Z_Lstm(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat = keras.layers.Concatenate(axis=-1)
        self.hidden_layers = [
            keras.layers.LSTM(256, return_sequences=True, ),
            keras.layers.LSTM(256, return_sequences=True, ),
            keras.layers.Dense(128, activation='tanh')
        ]

    def call(self, inputs, **kwargs):
        latent, z = inputs
        x = self.concat([latent, z])
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class SvgfpModel(keras.Model):
    def __init__(self, encoder, decoder, frame_predictor, prior, post_prior, **kwargs):
        super(SvgfpModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.frame_predictor = frame_predictor
        self.prior = prior
        self.posterior = post_prior
        self.h_target = keras.layers.Lambda(lambda x: x[:, 1:, :, :, :])
        self.h = keras.layers.Lambda(lambda x: x[:, :-1, :, :, :])

    def call(self, inputs, **kwargs):
        # Inputs: x[i:i+seq_len], (None, None, obs_size, obs_size)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        x = tf.reshape(inputs, (batch_size * seq_len, tf.shape(inputs)[2:]))
        h_seq = self.encoder(x)
        h_seq = tf.reshape(h_seq, (batch_size, seq_len, -1))

        # 把h_seq拆成t-1个输入和t-1个输出两部分
        h_target = self.h_target(h_seq)
        h = self.h(h_seq)

        z_t, mu, logvar = self.posterior(h_target)
        _, mu_p, logvar_p = self.prior(h)
        h_pred = self.frame_predictor(tf.concat([h, z_t], axis=1))
        x_pred = self.decoder(h_pred)

        recon_loss = tf.losses.mse(x_pred, x)
        kld = kl_divergence(mu, logvar, mu_p, logvar_p)





