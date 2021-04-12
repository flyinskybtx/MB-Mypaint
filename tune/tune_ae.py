import tensorflow as tf
from datetime import datetime
from glob import glob

import numpy

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples

from tensorflow import keras

import os

from Model import make_layer, LayerConfig
from Model.callbacks import VaeVisCallback

LOG_DIR = os.path.abspath(os.path.dirname(__file__))


class Encoder(keras.Model):
    def __init__(self, obs_size, latent_size, encoder_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inputs = keras.layers.Input(name='Image_input', shape=(obs_size, obs_size, 1))
        x = inputs
        for i, layer_config in enumerate(encoder_layers, start=1):
            layer_config.number = i
            layer = make_layer(layer_config)
            x = layer(x)

        latent = keras.layers.Dense(latent_size, activation='linear', name='AE_latent')(x)
        outputs = latent

        self.model = keras.models.Model(inputs, outputs, name='Encoder')

    def call(self, inputs, **kwargs):
        return self.model(inputs)


class Decoder(keras.Model):
    def __init__(self, latent_size, decoder_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inputs = keras.layers.Input(name='Latent_Input', shape=(latent_size,))
        x = inputs
        for i, layer_config in enumerate(decoder_layers, start=1):
            layer_config.number = i
            layer = make_layer(layer_config)
            x = layer(x)
        outputs = x
        self.model = keras.models.Model(inputs, outputs, name='Decoder')

    def call(self, inputs, training=False, **kwargs):
        return self.model(inputs)


class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x, training=False, **kwargs):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            latent = self.encoder(X)
            y = self.decoder(latent)
            bce_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(Y, y)
            )
            mse_loss = tf.reduce_mean(keras.losses.mse(Y, y))

            total_loss = bce_loss + mse_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            'mse_loss': mse_loss,
        }

    def test_step(self, data, **kwargs):
        X, Y = data
        latent = self.encoder(X)
        y = self.decoder(latent)
        bce_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(Y, y)
        )
        mse_loss = tf.reduce_mean(keras.losses.mse(Y, y))

        total_loss = bce_loss + mse_loss
        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            'mse_loss': mse_loss,
        }


if __name__ == '__main__':
    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:2]

    data = get_all_vae_samples(engines, train_decoder=False)['obs']
    X = numpy.expand_dims(data[:, :, :, 0] - data[:, :, :, 1], axis=-1)
    Y = numpy.copy(X)

    encoder = Encoder(64, 7, [
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1, activation='tanh'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='tanh'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='tanh'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='flatten'),
        LayerConfig(type='dense', units=256, activation='tanh'),
    ])
    decoder = Decoder(7, [
        LayerConfig(type='dense', units=256, activation='tanh'),
        LayerConfig(type='dense', units=1024, activation='tanh'),
        LayerConfig(type='reshape', target_shape=(8, 8, 16)),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='deconv', filters=16, kernel_size=(2, 2), strides=1, activation='tanh'),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='deconv', filters=32, kernel_size=(2, 2), strides=1, activation='tanh'),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=1, kernel_size=(2, 2), strides=1, activation='sigmoid'),
    ])
    model = AE(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  run_eagerly=False,)

    model.fit(X, Y, batch_size=32, epochs=100,
              validation_split=0.2, shuffle=True,
              callbacks=[
                  keras.callbacks.EarlyStopping(
                      monitor='loss', patience=10, mode='auto', restore_best_weights=True),
                  keras.callbacks.TensorBoard(
                      log_dir=os.path.join(LOG_DIR, 'logs/ae/' + datetime.now().strftime("%m%d-%H%M")),
                      histogram_freq=1,
                      update_freq=1),
                  # VaeVisCallback(X, 2, total_count=5),
              ])
