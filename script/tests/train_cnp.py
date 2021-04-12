import os.path as osp

from tensorflow import keras

from Data.cnp_data import CNPData
from Model import MODEL_DIR
from Model.action_embedder import ActionEmbedder
from Data.Deprecated.dynamics_model import CNPModel, dist_logp
from Model.obs_encoder import ObsEncoder
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()

    cfg.is_vae = True
    obs_encoder = ObsEncoder(cfg)
    obs_encoder.build(input_shape=(None, cfg.obs_size, cfg.obs_size, 4))
    obs_encoder.trainable = False
    obs_encoder.load_weights(osp.join(MODEL_DIR, f'checkpoints/{obs_encoder.name}.h5'))

    action_embedder = ActionEmbedder(cfg)
    action_embedder.build(input_shape=(None, 3))

    # obs_encoder.make_predict_function()
    # decoder.make_predict_function()

    train_data = CNPData(savedir='offline/random',
                         batch_size=32,
                         num_context=cfg.num_context,
                         encoder=obs_encoder,
                         embedder=action_embedder,
                         train=True)
    val_data = CNPData(savedir='offline/random',
                       batch_size=32,
                       num_context=cfg.num_context,
                       encoder=obs_encoder,
                       embedder=action_embedder,
                       train=True)
    # vis_data = CNPData(savedir='offline/mini',
    #                    batch_size=32,
    #                    num_context=cfg_vae.num_context,
    #                    encoder=obs_encoder,
    #                    embedder=embedder,
    #                    train=True)

    dynamics_model = CNPModel(cfg)
    dynamics_model.model.summary()
    dynamics_model.model.compile(
        # run_eagerly=True,
        optimizer=keras.optimizers.Adam(5e-5),
        loss={
            'dist_concat': dist_logp,
            'mu': 'mse',
            # 'sigma': 'mse',
        },
        # metrics={'dist_concat': dist_mse, 'mu': stats},
    )
    dynamics_model.model.fit(train_data, epochs=1000,
                             validation_data=val_data, validation_steps=20,
                             callbacks=[
                                 keras.callbacks.EarlyStopping(
                                     monitor='val_dist_concat_loss',
                                     patience=5, mode='auto',
                                     restore_best_weights=True),
                             ]
                             )
