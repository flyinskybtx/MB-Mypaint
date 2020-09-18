from Model import AttrDict
from Model.basics import LayerConfig


def define_hparams():
    config = AttrDict()
    config.num_workers = 0
    config.train_batch_size = 200
    config.log_level = 'INFO'
    config.framework = 'tf'

    # Direct
    config.stride_size = 32
    config.stride_amplify = 3
    config.num_waypoints = 6
    # Windowed
    config.image_size = 64 * 12
    config.action_shape = 5
    config.window_size = 84 * 2
    config.obs_size = 64
    config.xy_grid = 32
    config.z_grid = 0.1
    config.max_step = 100,
    # Env
    config.num_images = 64
    config.brush_name = 'custom/slow_ink'
    config.rewards = ['img_cosine_reward',
                      'img_mse_loss',
                      'scale_loss',
                      'curvature_loss',
                      'iou_reward',
                      'incremental_reward',
                      'incremental_loss']
    config.num_simulators = 20
    # Data
    config.rollout_episodes = 500
    # Repr learning
    config.train_latent_encoder = False
    config.train_decoder = False
    config.evalute_repr = False
    config.latent_size = 7
    config.encoder_layers = [
        LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='flatten'),
        LayerConfig(type='dense', units=256),

    ]
    config.decoder_layers = [
        LayerConfig(type='dense', units=256),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='dense', units=1024),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='reshape', target_shape=(8, 8, 16)),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=16, kernel_size=(2, 2), strides=1),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=32, kernel_size=(2, 2), strides=1),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=1, kernel_size=(2, 2), strides=1),
    ]
    # Dynamics
    config.dynamics_layers = {
        'encoder': [
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
    config.train_dynamics = False
    config.evaluate_dynamics = False
    config.dynamics_batchsize = 32
    config.num_context = (15, 20)

    return config
