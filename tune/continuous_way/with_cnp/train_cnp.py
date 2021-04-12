import os
from datetime import datetime
from glob import glob
import numpy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from Data import DATA_DIR
from Data.new_cnp_data import NewCNPData
from Model import MODEL_DIR
from Model.action_embedder import ActionEmbedder
from Model.callbacks import CnpVisCallback, AnnealingCallback
from Model.cnp_model import CNP
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from results import LOG_DIR
from script.main_procs.hparams import define_hparams


def dist_logp(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)
    log_p = dist.log_prob(y_true)
    loss = -tf.reduce_mean(log_p)
    return loss


def dist_mse(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    mse = tf.reduce_mean(tf.square(mu - y_true))
    return mse


def dist_var_mean(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(sigma)


def dist_var_max(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_max(sigma)


if __name__ == '__main__':
    cfg = define_hparams()

    obs_encoder = ObsEncoder(cfg, name=f'obs_encoder_{cfg.latent_size}')
    obs_encoder.build_graph(input_shape=(None, cfg.obs_size, cfg.obs_size, 1))
    obs_encoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{obs_encoder.name}.h5'))
    obs_encoder.trainable = False

    obs_decoder = ObsDecoder(cfg, name=f'obs_decoder_{cfg.latent_size}')
    obs_decoder.build_graph(input_shape=(None, cfg.latent_size))
    obs_decoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{obs_decoder.name}.h5'))
    obs_decoder.trainable = False

    cnp = CNP(cfg)
    cnp.build_graph(input_shape=[(None, None, cfg.latent_size + 3),
                                 (None, None, cfg.latent_size),
                                 (None, cfg.latent_size + 3)])
    cnp.summary()

    # logp_weight = tf.Variable(0.)
    cnp.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=dist_mse,
        # loss=[dist_mse, dist_logp],
        # loss_weights=[1, logp_weight],
        # metrics=[dist_mse, dist_var_mean],
    )
    cnp.load_weights(filepath=os.path.join(MODEL_DIR, 'checkpoints/CNP/'))

    data_dirs = glob(os.path.join(DATA_DIR, 'HWDB/continuous/*'))

    train_data = NewCNPData(data_dirs, batch_size=16, num_context=(20, 30),
                            encoder=obs_encoder, embedder=None, train=True)
    val_data = NewCNPData(data_dirs, batch_size=16, num_context=(20, 30), steps=100,
                          encoder=obs_encoder, embedder=None, train=False)
    vis_data = NewCNPData(data_dirs, batch_size=16, num_context=(20, 30), steps=20,
                          encoder=obs_encoder, embedder=None, train=False)

    cnp.fit(train_data, epochs=100,
            validation_data=val_data, validation_steps=100,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10, mode='auto',
                    restore_best_weights=True),
                CnpVisCallback(vis_data, obs_encoder, obs_decoder, None,
                               num_context=20, frequency=5, total=3),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(LOG_DIR, 'cnp/' + datetime.now().strftime("%m%d-%H%M")),
                    histogram_freq=1,
                    update_freq=1),
                keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'checkpoints/CNP/'),
                                                save_best_only=True,
                                                save_weights_only=True,
                                                save_freq=5,
                                                monitor='loss'),
                # AnnealingCallback(logp_weight, start=0, anneal_time=10)
            ]
            )
    cnp.save_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{cnp.name}.h5'))
