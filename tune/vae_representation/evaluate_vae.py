import os
from glob import glob

import numpy
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples
from tune.test_annealing import VisCallback

if __name__ == '__main__':
    encoder = keras.models.load_model('./encoder')
    decoder = keras.models.load_model('./decoder')
    encoder.summary()
    decoder.summary()

    inputs = encoder.inputs
    mu, sigma, z = encoder(inputs)
    recon = decoder(z)
    dist = keras.layers.Concatenate(name='dist', axis=-1)([mu, sigma])
    outputs = [recon, dist]
    vae = keras.Model(inputs, outputs, name='vae')

    engines = glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Physics'))
    vis_data = get_all_vae_samples(engines, train_decoder=False)['obs']
    X = numpy.expand_dims(vis_data[:, :, :, 0] - vis_data[:, :, :, 1], axis=-1)
    X_hat = vae(X)[0].numpy()

    count = 0
    total_count = 10
    for y_pred, y_true in zip(X_hat, X):
        if numpy.max(y_true) > 0:
            frame = numpy.concatenate([y_pred, y_true], axis=1)  # concat along ys axis for view
            error = numpy.mean((X_hat - X) ** 2)
            fig = plt.figure()
            fig.suptitle(f'MSE: {error}')
            plt.imshow(frame)
            plt.show()

            plt.close(fig)
            count += 1
        if count >= total_count:
            break
