import os
from datetime import datetime
from glob import glob

import numpy

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples

from tensorflow import keras
import tensorflow as tf

import os

from Model.callbacks import VaeVisCallback

LOG_DIR = os.path.abspath(os.path.dirname(__file__))


class Encoder(keras.layers.Layer):
    def __init__(self, name='encoder', **kwargs):
        super().__init__(name, **kwargs)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv2D(32, (2, 2), 1, padding='same', activation='tanh')
        self.pool1 = keras.layers.MaxPool2D((2, 2), 2, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(16, (2, 2), 1, padding='same', activation='tanh')
        self.pool2 = keras.layers.MaxPool2D((2, 2), 2, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(16, (2, 2), 1, padding='same', activation='tanh')
        self.pool3 = keras.layers.MaxPool2D((2, 2), 2, padding='same')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256, activation='tanh')
        self.latent = keras.layers.Dense(7, activation='linear')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        z = self.latent(x)

        return z


class Decoder(keras.layers.Layer):
    def __init__(self, name='decoder', **kwargs):
        super().__init__(name, **kwargs)
        self.dense2 = keras.layers.Dense(256, activation='relu')
        self.dense3 = keras.layers.Dense(1024, activation='relu')
        self.reshape = keras.layers.Reshape((8, 8, 16))
        self.upsample1 = keras.layers.UpSampling2D((2, 2))
        self.bn4 = keras.layers.BatchNormalization()
        self.deconv1 = keras.layers.Conv2DTranspose(16, (2, 2), 1, 'same', activation='tanh')
        self.upsample2 = keras.layers.UpSampling2D((2, 2))
        self.bn5 = keras.layers.BatchNormalization()
        self.deconv2 = keras.layers.Conv2DTranspose(32, (2, 2), 1, 'same', activation='tanh')
        self.upsample3 = keras.layers.UpSampling2D((2, 2))
        self.deconv3 = keras.layers.Conv2DTranspose(1, (2, 2), 1, 'same', activation='sigmoid', name='output_image')

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        x = self.upsample1(x)
        x = self.bn4(x)
        x = self.deconv1(x)
        x = self.upsample2(x)
        x = self.bn5(x)
        x = self.deconv2(x)
        x = self.upsample3(x)
        x = self.deconv3(x)
        return x


class AutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, name="autoencoder", **kwargs
                 ):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        return reconstructed


if __name__ == '__main__':
    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:2]

    data = get_all_vae_samples(engines, train_decoder=False)['obs']
    X = numpy.expand_dims(data[:, :, :, 0] - data[:, :, :, 1], axis=-1)
    Y = numpy.copy(X)

    model = AutoEncoder()
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss=[keras.losses.binary_crossentropy, keras.losses.mse], )
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
