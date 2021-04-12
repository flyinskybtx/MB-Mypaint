from Model import AttrDict
from Model import LayerConfig


def _ray_config():
    config = AttrDict()
    config.num_workers = 10
    config.create_env_on_driver = True
    config.train_batch_size = 200
    config.batch_mode = 'complete_episodes'  # truncate_episodes
    config.ignore_worker_failures = True
    config.log_level = 'INFO'
    config.framework = 'tf'
    # --- 改了这个参数，会导致action的采样出问题，不知道为什么
    # config.explore = True
    # config.exploration_config = {"type": "EpsilonGreedy",  # StochasticSampling
    #                              "initial_epsilon": 1.0,
    #                              "final_epsilon": 0.02,
    #                              "epsilon_timesteps": 10000, }
    # config.evaluation_config = {"explore": False}
    return config


def _brush_config():
    config = AttrDict()
    config.brush_name = 'custom/slow_ink'
    config.agent_name = 'Physics'
    return config


def _direct_env_config():
    config = AttrDict()
    config.num_xy = 24
    config.num_z = 10
    config.num_waypoints = 6
    config.obs_size = 64
    config.num_images = 64
    config.rewards = [
        'img_cosine_reward',
        # 'img_mse_loss',
        # 'scale_loss',
        # 'curvature_loss',
        # 'iou_reward',
        # 'incremental_reward',
        # 'incremental_loss'
    ]
    config.split_view = True
    return config


def _continuous_env_config():
    config = AttrDict()
    config.image_size = 768  # 32 * 24
    config.window_size = 168  # 84 * 2
    config.obs_size = 64
    config.xy_grid = 32
    config.z_grid = 0.3
    config.max_step = 100
    config.num_images = 64
    config.rewards = [
        'img_cosine_reward',
        # 'img_mse_loss',
        # 'scale_loss',
        # 'curvature_loss',
        # 'iou_reward',
        # 'incremental_reward',
        # 'incremental_loss'
    ]
    config.split_view = False
    return config


def _policy_model_config():
    config = AttrDict()
    config.direct_cnn_layers = [
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
        LayerConfig(type='dropout', rate=0.2),
        LayerConfig(type='dense', units=1024, activation='relu'),
    ]
    config.direct_mlp_layers = [
        LayerConfig(type='batch_norm'),
        LayerConfig(type='conv', filters=64, kernel_size=(4, 4), strids=1, activation='relu'),
        LayerConfig(type='flatten'),
        LayerConfig(type='dense', units=800, activation='relu'),
        LayerConfig(type='dense', units=400, activation='relu'),
    ]
    return config


def _repr_model_config():
    config = AttrDict()
    config.latent_size = 128
    config.dist_z = False
    config.num_channels = 2
    config.decoder_binary_output = False

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
        LayerConfig(type='deconv', filters=config.num_channels, kernel_size=(2, 2), strides=1, activation='sigmoid'),
    ]
    return config


def _cnp_model_config():
    config = AttrDict()


def load_config():
    cfg = AttrDict()
    cfg.ray_config = _ray_config()
    cfg.brush_config = _brush_config()
    cfg.direct_env_config = _direct_env_config()
    cfg.continuous_env_config = _continuous_env_config()
    cfg.policy_model_config = _policy_model_config()
    cfg.repr_model_config = _repr_model_config()
    return cfg
