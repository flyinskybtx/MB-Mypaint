import copy
import os.path as osp
from tensorflow import keras

from Data.cnp_data import CNPData
from Model import MODEL_DIR
from Data.Deprecated.compact_cnp_model import CNP, CNPVisualiztionCallback
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()

    obs_encoder = ObsEncoder(cfg)
    obs_decoder = ObsDecoder(cfg)
    obs_encoder.build(input_shape=(None, 64, 64, 4))
    obs_decoder.build(input_shape=(None, cfg.latent_size))
    obs_encoder.trainable = False
    obs_decoder.trainable = False

    obs_encoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{obs_encoder.name}.h5'))
    obs_decoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{obs_decoder.name}.h5'))
    # obs_encoder.make_predict_function()
    # decoder.make_predict_function()

    cnp = CNP(cfg, obs_encoder, obs_decoder)
    keras.utils.plot_model(cnp.model, show_shapes=True,
                           to_file=f'{MODEL_DIR}/png/{cnp.model.name}.png')
    cnp.compile(optimizer=keras.optimizers.Adam(5e-4),
                # run_eagerly=True,
                )
    cnp.model.summary()
    train_data = CNPData(savedir='offline/random',
                         batch_size=32,
                         num_context=cfg.num_context,
                         train=True)
    val_data = copy.deepcopy(train_data)
    vis_data = copy.deepcopy(train_data)

    #
    cnp.fit(train_data, epochs=100,
            # steps_per_epoch=100,
            validation_data=val_data, validation_steps=20,
            callbacks=[
                # keras.callbacks.EarlyStopping(
                #     monitor='val_loss',
                #     patience=5, mode='auto',
                #     restore_best_weights=True),
                CNPVisualiztionCallback(vis_data, frequency=5),
            ]
            )
    cnp.summary()
    cnp.save(osp.join(MODEL_DIR, f'checkpoints/{cnp.name}'))
