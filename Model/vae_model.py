import tensorflow as tf
from tensorflow import keras


def kl_loss(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return -0.5 * tf.reduce_mean(tf.math.log(tf.square(sigma)) - tf.square(sigma) - tf.square(mu) + 1)


def recon_loss(y_true, y_pred):
    loss = keras.losses.binary_crossentropy(y_true, y_pred)
    loss *= 64 * 64
    return loss


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.dist = keras.layers.Concatenate(name='dist', axis=-1)

    def call(self, inputs, training=False, **kwargs):
        if training:
            mu, sigma, z = self.encoder(inputs, training=training)
            dist = self.dist([mu, sigma])
            recon = self.decoder(z, training=training)
            return [recon, dist]

        else:
            z = self.encoder(inputs, training=False)
            recon = self.decoder(z)
            return recon

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            y_pred = self.call(X, training=True)
            loss = self.compiled_loss(Y, y_pred, regularization_losses=self.losses,
                                      )

        grads = tape.gradient(loss, self.trainable_weights, )
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data, **kwargs):
        X, Y = data
        y_pred = self.call(X, training=True)
        loss = self.compiled_loss(Y, y_pred)

        return {m.name: m.result() for m in self.metrics}
