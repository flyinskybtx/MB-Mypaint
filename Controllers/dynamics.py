import numpy as np
import tensorflow as tf
tf.executing_eagerly()

class Dynamics:
    def __init__(self, encoder, decoder, embedder, predictor):
        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        self.predictor = predictor

        self.latent = None

    def initialize(self, delta):
        self.latent = self.encoder(np.expand_dims(delta, axis=-1)).numpy()

    def step(self, actions, z):
        embed = self.embedder(actions)
        d_latent = self.predictor((self.latent, embed, z))
        self.latent += d_latent
        return self.decoder(self.latent)
