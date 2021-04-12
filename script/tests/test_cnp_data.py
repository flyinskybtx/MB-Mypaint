import os
from glob import glob

from Data import DATA_DIR
from Data.new_cnp_data import NewCNPData
from Model import MODEL_DIR
from Model.action_embedder import ActionEmbedder
from Model.obs_encoder import ObsEncoder
from Model.obs_decoder import ObsDecoder
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    cfg = define_hparams()

    obs_encoder = ObsEncoder(cfg, name=f'obs_encoder_{cfg.latent_size}')
    obs_encoder.build(input_shape=(None, cfg.obs_size, cfg.obs_size, 1))
    obs_encoder.load_weights(os.path.join(MODEL_DIR, 'checkpoints', f'{obs_encoder.name}.h5'))
    obs_encoder.trainable = False

    action_embedder = ActionEmbedder(cfg)
    action_embedder.build(input_shape=(None, 3))

    engines = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:1]
    cnp_data = NewCNPData(engines, batch_size=16, num_context=(10, 20),
                          encoder=obs_encoder, embedder=action_embedder)

    X, Y = cnp_data.__getitem__(0)
    print(X.keys(), Y.keys())
    print(X['context_x'].shape, Y.shape)
