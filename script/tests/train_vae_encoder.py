import os
import os.path as osp
from datetime import datetime
from glob import glob
import numpy as np
from tensorflow import keras

from Data import DATA_DIR
from Data.vae_data import get_all_vae_samples
from Model import MODEL_DIR
from Model.vae_model import VAE
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:2]
    # train_data = VAEData(data_dirs=sims, batch_size=32, train_decoder=False)
    # print("Total length:", train_data.__len__())
    # train_data.on_epoch_end()
    # X, Y = train_data.__getitem__(10)
    # print(X.shape)
    data = get_all_vae_samples(engines, train_decoder=False)
    Xs = np.copy(data['obs'])
    Ys = np.copy(data['obs'])

    cfg_vae = define_hparams()
    cfg_vae.is_vae = True
    cfg_vae.latent_size = 16 * 16
    encoder = ObsEncoder(cfg_vae)
    decoder = ObsDecoder(cfg_vae)

    encoder.build(input_shape=(None, cfg_vae.obs_size, cfg_vae.obs_size, 4))
    encoder.model.summary()
    decoder.build(input_shape=(None, cfg_vae.latent_size))
    decoder.model.summary()

    # encoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{encoder.name}.h5'))
    # decoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{decoder.name}.h5'))

    ae = VAE(encoder, decoder)
    ae.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=['mse'],
        run_eagerly=True,
    )

    ae.fit(Xs, Ys, batch_size=128, epochs=500, validation_split=0.2, shuffle=True,
           callbacks=[
               keras.callbacks.EarlyStopping(monitor='loss', patience=20, mode='auto', restore_best_weights=True),
               keras.callbacks.TensorBoard(
                   log_dir=osp.join(MODEL_DIR, 'logs/vae/' + datetime.now().strftime("%Y%m%d-%H%M%S")),
                   histogram_freq=1
               ),

           ])
    #
    # records = ae.fit(
    #     train_data, epochs=500, steps_per_epoch=min(100, train_data.__len__()),
    #     validation_data=val_data, validation_steps=200,
    #     callbacks=[
    #         keras.callbacks.EarlyStopping(
    #             monitor='loss', patience=20, mode='auto', restore_best_weights=True),
    #         VisGenCallback(vis_data, frequency=20),
    #         keras.callbacks.TensorBoard(log_dir=osp.join(MODEL_DIR, 'logs' + datetime.now().strftime("%Y%m%d-%H%M%S"))),
    #     ]
    # )

    # ae.summary()
    encoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{encoder.name}.h5'))
    decoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{decoder.name}.h5'))
