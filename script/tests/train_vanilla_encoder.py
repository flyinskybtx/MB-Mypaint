import copy
import os.path as osp
from datetime import datetime

from tensorflow import keras

from Data.vae_data import VAEData
from Model import MODEL_DIR
from Data.Deprecated.repr_model import VisGenCallback
from Model.vae_model import VanillaAE
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    data_dir = 'offline/random'
    batch_size = 32
    train_data = VAEData(savedir=data_dir, batch_size=batch_size, is_encoder=True, max_samples=1e5)
    val_data = copy.deepcopy(train_data)
    vis_data = copy.deepcopy(train_data)

    cfg_vanilla = define_hparams()
    cfg_vanilla.is_vae = False
    encoder = ObsEncoder(cfg_vanilla)
    decoder = ObsDecoder(cfg_vanilla)

    encoder.build(input_shape=(None, cfg_vanilla.obs_size, cfg_vanilla.obs_size, 4))
    encoder.model.summary()
    decoder.build(input_shape=(None, cfg_vanilla.latent_size))
    decoder.model.summary()

    ae = VanillaAE(encoder, decoder)
    ae.compile(
        optimizer=keras.optimizers.Adam(lr=5e-4),
        metrics=['mse'],
        run_eagerly=True,
    )

    records = ae.fit(
        train_data, epochs=500, steps_per_epoch=min(100, train_data.__len__()),
        # validation_data=val_data, validation_steps=200,
        callbacks=[
            # keras.callbacks.EarlyStopping(
            #     monitor='loss', patience=20, mode='auto', restore_best_weights=True),
            VisGenCallback(vis_data, frequency=20),
            keras.callbacks.TensorBoard(log_dir=osp.join(MODEL_DIR, 'logs' + datetime.now().strftime("%Y%m%d-%H%M%S"))),
        ]
    )

    ae.summary()
    encoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{encoder.name}.h5'))
    decoder.save_weights(osp.join(MODEL_DIR, f'checkpoints/{decoder.name}.h5'))
