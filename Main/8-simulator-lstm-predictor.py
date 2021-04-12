import tensorflow as tf
from tensorflow import keras


def kl_distance(y_true, y_pred):
    """ distribution M(mu, sigma)'s KL distance to prior N(0, 1)"""
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return -0.5 * tf.reduce_mean(tf.math.log(tf.square(sigma)) - tf.square(sigma) - tf.square(mu) + 1)


def make_encoder():
    encoder = keras.Sequential(name='encoder')  # 参考DCGAN
    encoder.add(keras.layers.Input((64, 64, 2)))
    encoder.add(keras.layers.Conv2D(32, 2, 1, "same"))
    encoder.add(keras.layers.LeakyReLU(alpha=0.2))
    encoder.add(keras.layers.MaxPool2D(2, 2, 'same'))
    encoder.add(keras.layers.Conv2D(16, 2, 2, "same"))
    encoder.add(keras.layers.LeakyReLU(alpha=0.2))
    encoder.add(keras.layers.MaxPool2D(2, 2, 'same'))
    encoder.add(keras.layers.Conv2D(16, 2, 2, "same"))
    encoder.add(keras.layers.LeakyReLU(alpha=0.2))
    encoder.add(keras.layers.MaxPool2D(2, 2, 'same'))
    encoder.add(keras.layers.Flatten())
    encoder.summary()
    return encoder


def make_decoder():
    decoder = keras.Sequential(name='decoder')  # 参考DCGAN
    decoder.add(keras.layers.Input((64,)))
    decoder.add(keras.layers.Reshape((2, 2, 16)))
    decoder.add(keras.layers.UpSampling2D(2))
    decoder.add(keras.layers.Conv2DTranspose(16, 2, 2))
    decoder.add(keras.layers.LeakyReLU(alpha=0.2))
    decoder.add(keras.layers.UpSampling2D(2))
    decoder.add(keras.layers.Conv2DTranspose(32, 2, 2))
    decoder.add(keras.layers.LeakyReLU(alpha=0.2))
    decoder.add(keras.layers.UpSampling2D(2))
    decoder.add(keras.layers.Conv2DTranspose(2, 1, 1, padding='valid'))
    decoder.add(keras.layers.Activation('sigmoid'))
    decoder.summary()
    return decoder


class SVG_FP(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(SVG_FP, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, **kwargs):
        z = self.encoder(inputs)
        recon = self.decoder(z)
        return recon
