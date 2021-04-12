import os

import numpy as np
from tensorflow import keras
import tensorflow as tf

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples
from Main import load_config
from Model.vae_model import VAE
from Model.vae_model import recon_loss, kl_loss
from Model.callbacks import AeVisCallback, VaeVisCallback, AnnealingCallback
from Model.obs_decoder import ObsDecoder
from Model.obs_encoder import ObsEncoder

if __name__ == '__main__':
    config = load_config()
    config.repr_model_config.dist_z=True

    obs_encoder = ObsEncoder(config.repr_model_config,
                             name=f'obs_encoder_{config.repr_model_config.latent_size}')
    obs_encoder.build_graph(
        input_shape=(None, config.continuous_env_config.obs_size, config.continuous_env_config.obs_size,
                     config.repr_model_config.num_channels))
    obs_encoder.summary()

    obs_decoder = ObsDecoder(config.repr_model_config, name=f'obs_decoder_{config.repr_model_config.latent_size}')
    obs_decoder.build_graph(input_shape=(None, config.repr_model_config.latent_size))
    obs_decoder.summary()

    ae = VAE(obs_encoder, obs_decoder)
    kl_weight = tf.Variable(0.)
    ae.compile(optimizer=keras.optimizers.Adam(5e-4),
               loss=[recon_loss, kl_loss],
               loss_weights=[1, kl_weight],
               )

    # data
    author = '1003-c.pot'
    data_dirs = [os.path.join(DATA_DIR, 'HWDB/continuous', author)]
    data = get_all_vae_samples(data_dirs)['obs']
    X = data[:, :, :, 0] - data[:, :, :, 1]
    X = np.stack([X, ], axis=-1)
    # X = np.stack([X, data[:, :, :, 3]], axis=-1)

    # train
    ae.fit(X, (X, X), batch_size=32, epochs=500,
           validation_split=0.2, shuffle=True, validation_batch_size=32,
           callbacks=[
               keras.callbacks.EarlyStopping(
                   monitor='val_loss', patience=10, mode='auto', restore_best_weights=True),
               VaeVisCallback(X[:100], 5, total_count=3),
               AnnealingCallback(weight=kl_weight, start=10, anneal_time=20),

           ]
           )
