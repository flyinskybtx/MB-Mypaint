import os
from datetime import datetime
from glob import glob

import numpy
import tensorflow as tf
from tensorflow import keras

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples
from Model import MODEL_DIR
from Model.vae_model import VAE
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from Model.callbacks import AnnealingCallback, VaeVisCallback
from results import LOG_DIR, PIC_DIR
from script.main_procs.hparams import define_hparams


def kl_loss(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    return -0.5 * tf.reduce_mean(tf.math.log(tf.square(sigma)) - tf.square(sigma) - tf.square(mu) + 1)


def recon_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) * 64 * 64


if __name__ == '__main__':
    cfg = define_hparams()

    encoder = ObsEncoder(cfg)
    decoder = ObsDecoder(cfg)

    encoder.build(input_shape=(None, cfg.obs_size, cfg.obs_size, 1))
    encoder.summary()
    decoder.build(input_shape=(None, cfg.latent_size))
    decoder.summary()

    if os.path.exists(os.path.join(MODEL_DIR, 'checkpoints', f'{encoder.name}.h5')):
        encoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{encoder.name}.h5'))
    if os.path.exists(os.path.join(MODEL_DIR, 'checkpoints', f'{decoder.name}.h5')):
        decoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{decoder.name}.h5'))

    vae = VAE(encoder, decoder)

    kl_weight = tf.Variable(0.)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss=[recon_loss, kl_loss],
                loss_weights=[1, kl_weight],
                # run_eagerly=True,
                )

    sims = glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*'))[:2]
    phys = glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Phy*'))

    train_data = get_all_vae_samples(sims, train_decoder=False)['obs']
    X = numpy.expand_dims(train_data[:, :, :, 0] - train_data[:, :, :, 1], axis=-1)
    Y = numpy.copy(X)
    vis_data = get_all_vae_samples(phys, train_decoder=False)['obs']
    vis_data = numpy.expand_dims(vis_data[:, :, :, 0] - vis_data[:, :, :, 1], axis=-1)[:100]

    vae.fit(X, (Y, Y), batch_size=32, epochs=500,
            validation_split=0.2, shuffle=True, validation_batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, mode='auto', restore_best_weights=True),
                # GradExtendedTensorBoard(
                #     val_data=(X, Y),
                #     log_dir=os.path.join(LOG_DIR, 'vae/' + datetime.now().strftime("%m%d-%H%M")),
                #     histogram_freq=1,
                #     update_freq=1,
                #     write_grads=True),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(LOG_DIR, 'vae/' + datetime.now().strftime("%m%d-%H%M")),
                    histogram_freq=1,
                    update_freq=1),
                keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_DIR, 'checkpoints/VAE/'),
                                                save_best_only=True,
                                                save_weights_only=True,
                                                save_freq=5,
                                                monitor='loss'),
                VaeVisCallback(vis_data, 5, img_dir=os.path.join(PIC_DIR, 'vae'), total_count=3),
                AnnealingCallback(weight=kl_weight, start=10, anneal_time=20),
            ]
            )

    encoder.save_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{encoder.name}.h5'))
    decoder.save_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{decoder.name}.h5'))
