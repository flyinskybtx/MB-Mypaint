import numpy as np
from collections import namedtuple

ExperimentalConfig = namedtuple(
    'ExperimentalConfig',
    [
        'image_size',
        'window_size',
        'stride_size',
        'stride_amplify',
        'z_size',
        'num_keypoints',
        'max_step',
        'brush_name',
        'action_shape',
        'image_nums',
        'bottleneck'
    ]
)

experimental_config = ExperimentalConfig(
    image_size=64*12,
    window_size=32*2,
    stride_size=16*2,
    stride_amplify=3,
    z_size=1/10,
    num_keypoints=6,
    max_step=30,
    brush_name='custom/slow_ink',
    action_shape=5,
    image_nums=np.arange(48),
    bottleneck=7,
)
