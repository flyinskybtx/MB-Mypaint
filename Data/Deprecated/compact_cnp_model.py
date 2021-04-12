import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from Model.custom_layers import ActionEmbedLayer
from Model.basics import make_layer


def dist_logp(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)
    log_p = dist.log_prob(y_true)
    loss = -tf.reduce_mean(log_p)
    return loss


def build_cnp_model(config, obs_encoder):
    context_obs = keras.layers.Input(shape=(None, config.obs_size, config.obs_size, 4),
                                     name='context_obs')  # b, n, 64, 64, 4
    context_new_obs = keras.layers.Input(shape=(None, config.obs_size, config.obs_size, 4),
                                         name='context_new_obs')  # b, n, 64, 64, 4
    context_actions = keras.layers.Input(shape=(None, 3),
                                         name='context_actions')  # b, n, 3,
    query_obs = keras.layers.Input(shape=(config.obs_size, config.obs_size, 4),
                                   name='query_obs')  # b, 64, 64, 4
    query_actions = keras.layers.Input(shape=(3,), name='query_actions')  # b, 3

    mu0, _ = obs_encoder(context_obs)
    mu1, _ = obs_encoder(context_new_obs)
    mu = keras.layers.Subtract(name='Delta_z')([mu1, mu0])
    onehots = ActionEmbedLayer(config, name='Context_action_embedder')(context_actions)
    encodes = keras.layers.Concatenate(axis=-1, name='Context_logits')([mu, onehots])

    # Context Encoder
    for i, layer_config in enumerate(config.dynamics_layers['obs_encoder'], start=1):
        layer_config.number = i
        layer = make_layer(layer_config)
        encodes = layer(encodes)

    # Context Aggregator
    logits = keras.layers.Lambda(lambda logit: tf.reduce_mean(logit, axis=1, keepdims=False),
                                 name='avg_aggregates')(encodes)

    # CNP Decoder
    query_mu, query_sigma = obs_encoder(query_obs)
    onehots = ActionEmbedLayer(config, name='Query_action_embedder')(query_actions)
    decodes = keras.layers.Concatenate(axis=-1, name='query_logits')([logits, query_mu, onehots])
    for i, layer_config in enumerate(config.dynamics_layers['decoder'],
                                     start=len(config.dynamics_layers['obs_encoder']) + 1):
        layer_config.number = i
        layer = make_layer(layer_config)
        decodes = layer(decodes)
    mu = keras.layers.Dense(name='mu', units=config.latent_size, activation='linear')(decodes)
    log_sigma = keras.layers.Dense(name=f'log_sigma', units=config.latent_size)(decodes)
    sigma = keras.layers.Lambda(lambda logit: 0.1 + 0.9 * tf.nn.softplus(logit),
                                name='sigma')(log_sigma)
    dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)([mu, sigma])

    model = keras.models.Model(inputs={'context_obs': context_obs,
                                       'context_new_obs': context_new_obs,
                                       'context_actions': context_actions,
                                       'query_obs': query_obs,
                                       'query_actions': query_actions, },
                               outputs=[mu, sigma, dist_concat], name='cnp_model')

    return model


class CNP(keras.Model):
    def __init__(self, config, obs_encoder, obs_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.obs_size = config.obs_size

        self.obs_encoder = obs_encoder
        self.obs_decoder = obs_decoder

        self.obs_encoder.trainable = False
        self.obs_decoder.trainable = False

        self.model = build_cnp_model(config, self.obs_encoder)
        self.model.summary()

        self.context_obs = None
        self.context_new_obs = None
        self.context_actions = None

    def set_context(self, context_obs=None, context_new_obs=None, context_actions=None):
        self.context_obs = context_obs
        self.context_new_obs = context_new_obs
        self.context_actions = context_actions

    def call(self, inputs: dict, training=None, mask=None):
        if all([self.context_obs is not None, self.context_new_obs is not None, self.context_actions is not None]):
            inputs.setdefault('context_obs', self.context_obs)
            inputs.setdefault('context_new_obs', self.context_new_obs)
            inputs.setdefault('context_actions', self.context_actions)

        mu, sigma, _ = self.model(inputs)
        return mu, sigma

    def reconstruct(self, inputs, ):
        inputs.setdefault('context_obs', self.context_obs)
        inputs.setdefault('context_new_obs', self.context_new_obs)
        inputs.setdefault('context_actions', self.context_actions)
        delta, _, _ = self.model(inputs)
        latent, _ = self.obs_encoder(inputs['query_obs'])
        new_latent = latent + delta
        # new_latent = latent
        # TODO: use sigma and sampler here

        reconstruction = self.obs_decoder(new_latent)  # b, 64, 64, 1
        return reconstruction

    def train_step(self, data, **kwargs):
        x = data[0]
        y = data[1]

        with tf.GradientTape() as tape:
            mu, sigma, dist_concat = self.model(x)

            mu0, _ = self.obs_encoder(x['query_obs'])
            mu1, _ = self.obs_encoder(y['target_new_obs'])
            # TODO: use sigma
            delta = mu1 - mu0

            # dist_loss = dist_logp(deltas, dist_concat)
            dist = tfp.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma)
            log_p = dist.log_prob(delta)
            dist_loss = -tf.reduce_mean(log_p)
            mu_loss = tf.reduce_mean(tf.losses.mse(mu, delta))
            reconstruction = self.obs_decoder(mu + mu0)  # b, 64, 64
            gt = tf.expand_dims(y['target_new_obs'][:, :, :, 0] - y['target_new_obs'][:, :, :, 1], axis=-1)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(gt, reconstruction)
            )
            reconstruction_loss *= self.obs_size * self.obs_size
            # total_loss = loss_mu + loss_sigma + reconstruction_loss
            total_loss = dist_loss + mu_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "dist_loss": dist_loss,
            "mu_loss": mu_loss,
            "recons_loss": reconstruction_loss,
        }

    def test_step(self, data, **kwargs):
        x = data[0]
        y = data[1]

        mu, sigma, dist_concat = self.model(x)

        mu0, _ = self.obs_encoder(x['query_obs'])
        mu1, _ = self.obs_encoder(y['target_new_obs'])
        # TODO: use sigma
        delta = mu1 - mu0

        # dist_loss = dist_logp(deltas, dist_concat)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=sigma)
        log_p = dist.log_prob(delta)
        dist_loss = -tf.reduce_mean(log_p)

        mu_loss = tf.reduce_mean(tf.losses.mse(mu, delta))
        reconstruction = self.obs_decoder(mu + mu0)  # b, 64, 64
        gt = tf.expand_dims(y['target_new_obs'][:, :, :, 0] - y['target_new_obs'][:, :, :, 1], axis=-1)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(gt, reconstruction)
        )
        reconstruction_loss *= self.obs_size * self.obs_size
        total_loss = dist_loss + mu_loss

        return {
            "loss": total_loss,
            "dist_loss": dist_loss,
            "mu_loss": mu_loss,
            "recons_loss": reconstruction_loss,
        }


class CNPVisualiztionCallback(keras.callbacks.Callback):
    def __init__(self, data_generator, frequency):
        super().__init__()
        self.generator = data_generator
        self.frequency = frequency

    def on_epoch_begin(self, epoch, logs=None, totol_count=10):
        if epoch % self.frequency != 0:
            return

        X, Y = self.generator.next()
        mu0, _ = self.model.obs_encoder(X['query_obs'])
        mu1, _ = self.model.obs_encoder(Y['target_new_obs'])
        delta, _, = self.model(X)
        delta_gt = mu1 - mu0

        _gts = X['query_obs'][:, :, :, 0] - X['query_obs'][:, :, :, 1]
        gts = Y['target_new_obs'][:, :, :, 0] - Y['target_new_obs'][:, :, :, 1]
        preds = self.model.reconstruct(X)
        preds = np.squeeze(preds, axis=-1)

        count = 0
        for pred, gt, dl, dl_gt, _gt, lt, lt_gt in zip(preds, gts, delta, delta_gt, _gts, mu1, mu0+delta):
            if np.max(gt) > 0:
                frame = np.concatenate([pred, _gt, gt], axis=1)  # concat along ys axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                error = np.mean((pred - gt) ** 2)
                fig.suptitle(f'Delta: {np.round(dl, 3)},\n GT: {np.round(dl_gt, 3)}'
                             f'\nLatent:{np.round(lt, 3)}, \n Latent_GT:{np.round(lt_gt, 3)} '
                             )
                plt.imshow(frame)
                plt.show()
                plt.close(fig)
                count += 1
            if count > totol_count:
                break
