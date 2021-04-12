from collections import namedtuple

import numpy as np

ExperimentalConfig = namedtuple(
    'ExperimentalConfig',
    [
        'image_size',
        'window_size',
        'obs_size',
        'stride_size',
        'stride_amplify',
        'xy_size',
        'z_size',
        'num_waypoints',
        'max_step',
        'brush_name',
        'action_shape',
        'image_nums',
        'bottleneck',
        'state_dim',
        'reward_names',
    ]
)

experimental_config = ExperimentalConfig(
    image_size=64 * 12,
    window_size=84 * 2,
    obs_size=64,
    stride_size=16 * 2,
    stride_amplify=3,
    xy_size=16 * 2,
    z_size=1 / 10,
    num_keypoints=6,
    max_step=30,
    brush_name='custom/slow_ink',
    action_shape=5,
    image_nums=np.arange(48),
    bottleneck=64,
    state_dim=7,
    rewards=['img_cosine_reward',
             'img_mse_loss',
             'scale_loss',
             'curvature_loss',
             'iou_reward',
             'incremental_reward',
             'incremental_loss'],
)
