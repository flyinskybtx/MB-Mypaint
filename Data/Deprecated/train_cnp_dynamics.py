from Data.cnp_data import CNPGenerator
from Model.basics import LayerConfig
from Data.Deprecated.dynamics_model import CNPModel
from Model.action_embedder import ActionEmbedder
from Data.Deprecated.repr_model import ReprModel
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    # --- load obs_encoder
    cfg = define_hparams()
    cfg.train_latent_encoder = False
    cfg.train_decoder = False
    cfg.num_context = (10, 20)
    repr_model = ReprModel(cfg)
    action_embed = ActionEmbedder(cfg)

    # --- build mlp
    cfg.dynamics_layers = {
        'obs_encoder': [
            LayerConfig(type='dense', units=256, activation='tanh'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='tanh'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='tanh'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=128, activation='linear'),
        ],
        'decoder': [
            LayerConfig(type='dense', units=256, activation='tanh'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='tanh'),
            LayerConfig(type='dropout', rate=0.2),
        ]
    }

    dynamics_model = CNPModel(cfg)
    dynamics_model.model.summary()

    # --- data_loader loader
    train_data = CNPGenerator(repr_model=repr_model,
                              action_embed=action_embed,
                              savedir='offline/random',
                              batch_size=32,
                              num_context=cfg.num_context,
                              train=True)
    vali_data = CNPGenerator(repr_model=repr_model,  # Must not use copy
                             action_embed=action_embed,
                             savedir='offline/random',
                             batch_size=32,
                             num_context=cfg.num_context,
                             train=False)

    # --- train mlp
    # Load old
    dynamics_model.load_model()
    dynamics_model.train_model(train_data, vali_data)
    dynamics_model.save_model()
