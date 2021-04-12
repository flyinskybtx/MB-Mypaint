import tensorflow as tf
from tensorflow import keras


def recon_loss(y_true, y_pred):
    recon_loss = keras.losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    recon_loss *= 64 * 64
    return recon_loss


def mse_loss(y_true, y_pred):
    z_loss = tf.reduce_mean(tf.square(y_true[:, :, :, 1] - y_pred[:, :, :, 1]))
    z_loss *= 64 * 64
    return z_loss


class AE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon
