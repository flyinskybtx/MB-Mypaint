import random
from collections import defaultdict

import numpy as np
from tensorflow import keras

from Data import obs_to_delta, NewBaseData


class MLPData(NewBaseData):
    def __init__(self, data_dirs, batch_size, encoder: keras.Model, embedder=None, **kwargs):
        super().__init__(data_dirs, batch_size, **kwargs)
        self.encoder = encoder
        self.embedder = embedder

        self.on_epoch_end()

    def _get_batch(self, index, batch_size):
        sim = random.choice(list(self.data.keys()))
        data = self.data[sim]
        batch_data = {k: v[index * batch_size: (index + 1) * batch_size] for k, v in data.items()}
        return batch_data

    def __getitem__(self, index):
        batch_data = self._get_batch(index, self.batch_size)

        obs = batch_data['obs']
        new_obs = batch_data['new_obs']
        actions = batch_data['actions']

        x0 = self.encoder(obs_to_delta(obs))
        x1 = self.encoder(obs_to_delta(new_obs))
        u = self.embedder(actions)
        dx = x1 - x0
        z = obs[:, 0, 0, 3].reshape(-1, 1)

        return (x0, u, z), dx

    def __len__(self):
        min_sim_length = np.min([len(list(v.values())[0]) for v in self.data.values()])
        return int(min_sim_length // self.batch_size) - 1

    def on_epoch_end(self):
        self.data = self.shuffle(self.data)

    @property
    def all(self):
        data = defaultdict(list)
        for sim in self.data.values():
            for k, v in sim.items():
                data[k].append(v)
        obs = np.concatenate(data['obs'], axis=0)
        new_obs = np.concatenate(data['new_obs'], axis=0)
        actions = np.concatenate(data['actions'], axis=0)

        x0 = self.encoder(obs_to_delta(obs))
        x1 = self.encoder(obs_to_delta(new_obs))
        u = self.embedder(actions)
        dx = x1 - x0

        z = obs[:, 0, 0, 3].reshape(-1, 1)
        return (x0, u, z), dx
