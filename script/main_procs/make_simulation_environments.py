import copy
import json
import os.path as osp

import numpy as np

from Data import DATA_DIR
from script.main_procs.hparams import define_hparams


def sample_env_configs(num_configs, resample=False):
    """ Sample brush configs for each env cfg
    :param num_configs
    :param resample (default=False)
    :return configs
    """
    if resample:
        factors = np.random.uniform(0.8, 1.2, size=(num_configs, 4))
        np.savetxt(osp.join(DATA_DIR, 'configs/factors.txt'), factors)
    else:
        factors = np.loadtxt(osp.join(DATA_DIR, 'configs/factors.txt'), dtype=np.float)

    configs = [define_hparams() for _ in range(num_configs)]
    for i, (factor, config) in enumerate(zip(factors, configs)):
        config.brush_factor = factor
        config.config_num = i
    return configs
