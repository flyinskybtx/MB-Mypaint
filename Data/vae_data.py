import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import random

from Data import DATA_DIR, NewBaseData, _merge_dict


class VAEData(NewBaseData):
    def __init__(self, data_dirs, batch_size, train_decoder=False, **kwargs):
        super().__init__(data_dirs, batch_size, **kwargs)
        self.max_samples = kwargs.setdefault('max_samples', None)
        self.train_decoder = train_decoder

    def __getitem__(self, index):
        batch = []
        keys = list(self.data.keys())
        random.shuffle(keys)
        while len(batch) < self.batch_size:
            for key in keys:
                value = self.data[key]
                batch.append({k: v[index % len(v)] for k, v in value.items()})
            index += 1
        batch = _merge_dict(batch[:self.batch_size])
        if self.train_decoder:
            return np.stack(batch['obs'], axis=0), np.stack(batch['new_obs'], axis=0)
        else:
            return np.stack(batch['obs'], axis=0), np.stack(batch['obs'], axis=0)

    def __len__(self):
        return int(np.sum([len(list(v.values())[0]) for v in self.data.values()])) // self.batch_size


def get_all_vae_samples(data_dirs, train_decoder=False):
    data = defaultdict(list)
    vae_data = VAEData(data_dirs, batch_size=16, train_decoder=train_decoder)
    for value in vae_data.data.values():
        for k, v in value.items():
            data[k].append(v)
    for k, v in data.items():
        data[k] = np.concatenate(v, axis=0)
    return data


def test():
    sims = sorted(glob(os.path.join(DATA_DIR, f'offline/slow_ink', 'Sim*')),
                     key=lambda x: int(x.split('Sim')[-1]))[:2]
    train_data = VAEData(data_dirs=sims, batch_size=32, train_decoder=False)
    data = get_all_vae_samples(sims, train_decoder=False)
