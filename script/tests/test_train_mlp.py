import os
from glob import glob
import numpy

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from Data import DATA_DIR
from Data.mlp_data import MLPData
from Data.new_cnp_data import NewCNPData
from Model import MODEL_DIR
from Model.action_embedder import ActionEmbedder
from Model.callbacks import CnpVisCallback, MlpVisCallback
from Model.cnp_model import CNP
from Model.mlp_model import MLP
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams

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

    action_embedder = ActionEmbedder(cfg)
    action_embedder.build(input_shape=(None, 3))

    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:1]
    train_data = MLPData(engines, batch_size=64, encoder=obs_encoder, embedder=action_embedder).all
    # physics = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Phy*')))
    vis_data = MLPData(engines, batch_size=64, encoder=obs_encoder, embedder=action_embedder)

    mlp = MLP(cfg)
    mlp.build_graph(input_shape=[(None, cfg.latent_size + 15), (None, 1)])
    mlp.summary()
    #
    # mlp = keras.Sequential(name='mlp')
    # mlp.add(keras.layers.Input(shape=(23, )))
    # mlp.add(keras.layers.Dense(256, activation='relu'))
    # mlp.add(keras.layers.Dense(256, activation='relu'))
    # mlp.add(keras.layers.Dense(256, activation='linear'))
    # mlp.add(keras.layers.Dense(256, activation='relu'))
    # mlp.add(keras.layers.Dense(8, activation='linear'))

    mlp.summary()
    mlp.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss=keras.losses.mse,
        # loss=[dist_mse, dist_logp],
        # loss_weights=[0, 1],
        # metrics=[dist_logp, dist_mse, dist_var_mean, dist_var_max],
    )
    mlp.fit(train_data[0], train_data[1], epochs=1000, batch_size=32,
            validation_split=0.2,
            # validation_data=val_data, validation_steps=10,
            callbacks=[
                # keras.callbacks.EarlyStopping(
                #     monitor='val_loss',
                #     patience=20, mode='auto',
                #     restore_best_weights=True),
                MlpVisCallback(vis_data, obs_encoder, obs_decoder, action_embedder,
                               frequency=10, total=3)
            ]
            )