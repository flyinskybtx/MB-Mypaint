import os

from Data import DATA_DIR
from Model import AttrDict
from Model import LayerConfig


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
    config.max_step = 100

    # Brush
    config.brush_info_file = os.path.join(DATA_DIR, f'offline/slow_ink/Physics', 'BrushInfo.myb')

    # Env
    config.num_images = 64
    config.brush_name = 'custom/slow_ink'
    config.rewards = [
        'img_cosine_reward',
        # 'img_mse_loss',
        # 'scale_loss',
        # 'curvature_loss',
        # 'iou_reward',
        'incremental_reward',
        'incremental_loss'
    ]
    config.num_simulators = 20
    # Data
    config.rollout_episodes = 500
    # Repr learning
    config.is_bw_output = True
    config.latent_size = 8
    config.encoder_layers = [
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
        LayerConfig(type='pool', pool_size=2, strides=2),
        LayerConfig(type='flatten'),
        LayerConfig(type='dense', units=256, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
    ]
    config.decoder_layers = [
        LayerConfig(type='dense', units=256, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='dense', units=1024, activation='relu'),
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='reshape', target_shape=(8, 8, 16)),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=16, kernel_size=(2, 2), strides=1, activation='relu'),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=32, kernel_size=(2, 2), strides=1, activation='relu'),
        LayerConfig(type='batch_norm'),
        LayerConfig(type='upsampling', size=(2, 2)),
        LayerConfig(type='deconv', filters=1, kernel_size=(2, 2), strides=1, activation='sigmoid'),
    ]
    # Dynamics
    config.dynamics_layers = {
        'obs_encoder': [
            # LayerConfig(type='batch_norm'),
            LayerConfig(type='dense', units=256, activation='relu'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='relu'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='relu'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='relu'),
            # LayerConfig(type='batch_norm'),
        ],
        'decoder': [
            LayerConfig(type='dense', units=256, activation='relu'),
            LayerConfig(type='dropout', rate=0.2),
            LayerConfig(type='dense', units=256, activation='relu'),
            LayerConfig(type='dropout', rate=0.2),
        ]
    }
    config.train_dynamics = False
    config.evaluate_dynamics = False
    config.dynamics_batchsize = 32
    config.num_context = (40, 50)

    return config
