import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from glob import glob

import numpy

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples

from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
import os

# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
from Model.callbacks import GradExtendedTensorBoard

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
        Y_ = self.model(X)[0].numpy()

        count = 0
        for y_pred, y_true in zip(Y_, X):
            if np.max(y_true) > 0:
                frame = np.concatenate([y_pred, y_true], axis=1)  # concat along ys axis for view
                fig = plt.figure(f'Epoch:{epoch}')
                error = np.mean((Y_ - X) ** 2)
                fig.suptitle(f'MSE: {error}')
                plt.imshow(frame)
                plt.show()
                plt.savefig(os.path.join(LOG_DIR, 'logs/pic', f'epoch{epoch}_{count+1}.png'))

                plt.close(fig)
                count += 1
            if count >= self.total_count:
                break


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
mu = keras.layers.Dense(7, name='z_mu')(x)
log_sigma = keras.layers.Dense(7, name='z_log_sigma')(x)
sigma = keras.layers.Lambda(lambda x: 0.1 + 0.9 * tf.nn.softplus(x), name='z_sigma')(log_sigma)
z = Sampling(name='sampling')((mu, sigma))

outputs = [mu, sigma, z]
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
outputs = keras.layers.Conv2DTranspose(1, (2, 2), 1, 'same', activation='sigmoid', name='reconstruction')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')

inputs = encoder.inputs
mu, sigma, z = encoder(inputs)
recon = decoder(z)
dist = keras.layers.Concatenate(name='dist', axis=-1)([mu, sigma])
outputs = [recon, dist]

vae = keras.Model(inputs, outputs, name='vae')


def kl_loss(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    loss = -0.5 * tf.reduce_mean(tf.math.log(tf.square(sigma))
                                 - tf.square(sigma)
                                 - tf.square(mu)
                                 + 1)

    return loss


def recon_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) * 64 * 64


if __name__ == '__main__':
    encoder.summary()
    decoder.summary()
    vae.summary()

    kl_weight = tf.Variable(0.)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss=[recon_loss, kl_loss],
                loss_weights=[1, kl_weight]
                # experimental_run_tf_function=False,
                # run_eagerly=True,
                )

    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:3]

    data = get_all_vae_samples(engines, train_decoder=False)['obs']
    X = numpy.expand_dims(data[:, :, :, 0] - data[:, :, :, 1], axis=-1)
    # X = X[:1000]
    Y = numpy.copy(X)

    vae.fit(X, (Y, Y), batch_size=32, epochs=500,
            validation_split=0.2, shuffle=True, validation_batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, mode='auto', restore_best_weights=True),
                GradExtendedTensorBoard(
                    val_data=(X, Y),
                    log_dir=os.path.join(LOG_DIR, 'logs/vae/save' + datetime.now().strftime("%m%d-%H%M")),
                    histogram_freq=1,
                    update_freq=1,
                    write_grads=True),
                keras.callbacks.ModelCheckpoint(filepath='./checkpoint/VAE/',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                save_freq=5,
                                                monitor='loss'),
                VisCallback(X, 5, total_count=3),
                AnnealingCallback(weight=kl_weight, kl_start=10, anneal_time=20),
            ]
            )

    encoder.save(filepath=f'./{encoder.name}')
    decoder.save(filepath=f'./{decoder.name}')
