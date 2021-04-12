import os
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import numpy

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples

from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import os

# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()

LOG_DIR = os.path.abspath(os.path.dirname(__file__))


class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight, kl_start=20, anneal_time=40):
        super().__init__()
        self.kl_start = kl_start
        self.anneal_time = anneal_time
        self.weight = weight

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.kl_start:
            new_weight = min(keras.backend.get_value(self.weight) + (1. / self.anneal_time), 1.)
            keras.backend.set_value(self.weight, new_weight)
        print("Current KL Weight is " + str(keras.backend.get_value(self.weight)))


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample zs, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        mu, sigma = inputs
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return dist.sample(name="sample")
        # z_mean, z_log_var = inputs
        # batch = tf.shape(z_mean)[0]
        # dim = tf.shape(z_mean)[1]
        # epsilon = tf.random.normal(shape=(batch, dim))
        # return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # return z_mean


class VisCallback(keras.callbacks.Callback):
    def __init__(self, data, frequency, total_count=10):
        super().__init__()
        self.data = data
        self.frequency = frequency
        self.total_count = total_count

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
        X = np.copy(self.data)
        np.random.shuffle(X)
        Y_ = self.model(X)
        # Y_ = self.model(X).mean()
        # Y_ = Y_.numpy()

        count = 0
        for y_pred, y_true in zip(Y_, X):
            if np.max(y_true) > 0:
                frame = np.concatenate([y_pred, y_true], axis=1)  # concat along ys axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                error = np.mean((Y_ - X) ** 2)
                fig.suptitle(f'MSE: {error}')
                plt.imshow(frame)
                plt.show()
                plt.close(fig)
                count += 1
            if count >= self.total_count:
                break


prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(7), scale=1),
                                      reinterpreted_batch_ndims=1)

inputs = keras.layers.Input(shape=(64, 64, 1), name='encoder_input')
x = keras.layers.BatchNormalization(name='encoder_bn1')(inputs)
x = keras.layers.Conv2D(32, (2, 2), 1, padding='same', activation='relu', name='encoder_conv1')(x)
x = keras.layers.MaxPool2D((2, 2), 2, padding='same', name='encoder_pool1')(x)
x = keras.layers.BatchNormalization(name='encoder_bn2')(x)
x = keras.layers.Conv2D(16, (2, 2), 1, padding='same', activation='relu', name='encoder_conv2')(x)
x = keras.layers.MaxPool2D((2, 2), 2, padding='same', name='encoder_pool2')(x)
x = keras.layers.BatchNormalization(name='encoder_bn3')(x)
x = keras.layers.Conv2D(16, (2, 2), 1, padding='same', activation='relu', name='encoder_conv3')(x)
x = keras.layers.MaxPool2D((2, 2), 2, padding='same', name='encoder_pool3')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu', name='encoder_dense1')(x)
x = keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(7), activation=None)(x)
x = tfp.layers.MultivariateNormalTriL(7,
                                      activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.1),
                                      )(x)

outputs = x
encoder = keras.Model(inputs, outputs, name='encoder')

latent_inputs = keras.layers.Input(shape=(7,), name='decoder_input')
x = keras.layers.Dense(256, activation='relu', name='decoder_dense1')(latent_inputs)
x = keras.layers.Dense(1024, activation='relu', name='decoder_dense2')(x)
x = keras.layers.Reshape((8, 8, 16), name='decoder_reshape')(x)
x = keras.layers.UpSampling2D((2, 2), name='decoder_up1')(x)
x = keras.layers.BatchNormalization(name='decoder_bn1')(x)
x = keras.layers.Conv2DTranspose(16, (2, 2), 1, 'same', activation='relu', name='decoder_deconv1')(x)
x = keras.layers.UpSampling2D((2, 2), name='decoder_up2')(x)
x = keras.layers.BatchNormalization(name='decoder_bn2')(x)
x = keras.layers.Conv2DTranspose(32, (2, 2), 1, 'same', activation='relu', name='decoder_deconv2')(x)
x = keras.layers.UpSampling2D((2, 2), name='decoder_up3')(x)
x = keras.layers.Conv2DTranspose(1, (2, 2), 1, 'same', activation='sigmoid', name='output_image')(x)
# xs = keras.layers.Flatten()(xs)
# xs = tfp.layers.IndependentBernoulli((64, 64, 1), tfp.distributions.Bernoulli.logits)(xs)
# negative_log_likelihood = lambda xs, rv_x: -rv_x.log_prob(xs)

outputs = x
decoder = keras.Model(latent_inputs, outputs)

vae = keras.Model(encoder.inputs, outputs=decoder(encoder.outputs[0]), name='vae')

if __name__ == '__main__':
    encoder.summary()
    decoder.summary()
    vae.summary()

    vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                # loss=negative_log_likelihood,
                loss=keras.losses.mse,
                # run_eagerly=True,
                )

    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:2]
    data = get_all_vae_samples(engines, train_decoder=False)['obs']
    X = numpy.expand_dims(data[:, :, :, 0] - data[:, :, :, 1], axis=-1)
    X = X[:1000]
    Y = numpy.copy(X)

    vae.fit(X, Y, batch_size=32, epochs=500,
            validation_split=0.2, shuffle=True,
            callbacks=[
                # keras.callbacks.EarlyStopping(
                #     monitor='loss', patience=10, mode='auto', restore_best_weights=True),
                # GradExtendedTensorBoard(
                #     val_data=(X, Y),
                #     log_dir=os.path.join(LOG_DIR, 'logs/vae/' + datetime.now().strftime("%m%d-%H%M")),
                #     histogram_freq=1,
                #     update_freq=1,
                #     write_grads=True),
                VisCallback(X, 5, total_count=3),
            ]
            )

    # def vae_loss(weight):
    #     @tf.function
    #     def loss(y_true, y_pred):
    #         bce_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    #         mu = vae.get_layer('encoder').get_layer('z_mu').output
    #         sigma = vae.get_layer('encoder').get_layer('z_sigma').output
    #         kl_loss = -0.5 * tf.reduce_mean(tf.math.log(tf.square(sigma))
    #                                         - tf.square(sigma)
    #                                         - tf.square(mu)
    #                                         + 1)
    #         total_loss = bce_loss + weight * kl_loss
    #         return total_loss
    #
    #     return loss
    #
    #
    # kl_weight = tf.Variable(0.)
    # vae.compile(optimizer=keras.optimizers.Adam(1e-3),
    #             loss=vae_loss(kl_weight),
    #             experimental_run_tf_function=False,
    #             # run_eagerly=True,
    #             )
    #
    # sims = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
    #                  key=lambda xs: int(xs.split('Sim')[-1]))[:2]
    #
    # data_loader = get_all_vae_samples(sims, train_decoder=False)['obs']
    # X = numpy.expand_dims(data_loader[:, :, :, 0] - data_loader[:, :, :, 1], axis=-1)
    # Y = numpy.copy(X)
    #
    # X = X[:1000]
    # Y = numpy.copy(X)
    #
    # vae.fit(X, Y, batch_size=32, epochs=500,
    #         validation_split=0.2, shuffle=True,
    #         callbacks=[
    #             # keras.callbacks.EarlyStopping(
    #             #     monitor='loss', patience=10, mode='auto', restore_best_weights=True),
    #             # GradExtendedTensorBoard(
    #             #     val_data=(X, Y),
    #             #     log_dir=os.path.join(LOG_DIR, 'logs/vae/' + datetime.now().strftime("%m%d-%H%M")),
    #             #     histogram_freq=1,
    #             #     update_freq=1,
    #             #     write_grads=True),
    #             VaeVisCallback(X, 5, total_count=3),
    #             AnnealingCallback(weight=kl_weight, kl_start=5),
    #         ]
    #         )
