import os
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from Data import NewBaseData, DATA_DIR


class DirectData(NewBaseData):
    def __init__(self, data_dirs, batch_size, **kwargs):
        super().__init__(data_dirs, batch_size, **kwargs)
        self.obs, self.actions = self._all()

    def __getitem__(self, item):
        obs = self.obs[item * self.batch_size: (item + 1) * self.batch_size]
        actions = self.actions[item * self.batch_size: (item + 1) * self.batch_size]
        actions = actions.reshape(self.batch_size, -1)  # Flatten
        values = np.zeros((self.batch_size, 1))

        return obs, (actions, values)

    def __len__(self):
        return int(self.actions.shape[0] // self.batch_size) - 1

    def _all(self):
        data = defaultdict(list)
        for sim in self.data.values():
            for k, v in sim.items():
                data[k].append(v)
        obs = np.concatenate(data['obs'], axis=0)
        actions = np.concatenate(data['actions'], axis=0)

        return obs, actions

    def shuffle(self):
        index = np.arange(self.actions.shape[0])
        np.random.shuffle(index)
        self.actions = self.actions[index]
        self.obs = self.obs[index]

    def on_epoch_end(self):
        self.shuffle()

    def get_all(self):
        _size = self.actions.shape[0]
        return np.copy(self.obs), (self.actions.reshape(_size, -1), np.zeros((_size, 1)))


class WindowedData(NewBaseData):
    def __init__(self, data_dirs, batch_size, **kwargs):
        super().__init__(data_dirs, batch_size, **kwargs)
        self.obs, self.actions = self._all()

    def __getitem__(self, item):
        obs = self.obs[item * self.batch_size: (item + 1) * self.batch_size]
        actions = self.actions[item * self.batch_size: (item + 1) * self.batch_size]
        values = np.zeros((self.batch_size, 1))

        return obs, (actions, values)

    def __len__(self):
        return int(self.actions.shape[0] // self.batch_size) - 1

    def _all(self):
        data = defaultdict(list)
        for sim in self.data.values():
            for k, v in sim.items():
                data[k].append(v)
        obs = np.concatenate(data['obs'], axis=0)
        actions = np.concatenate(data['actions'], axis=0)

        return obs, actions

    def shuffle(self):
        index = np.arange(self.actions.shape[0])
        np.random.shuffle(index)
        self.actions = self.actions[index]
        self.obs = self.obs[index]

    def on_epoch_end(self):
        self.shuffle()

    def get_all(self, num=None):
        if num is None:
            _size = self.actions.shape[0]
            return self.obs, (self.actions, np.zeros((_size, 1)))

        else:
            return self.obs[:num], (self.actions[:num], np.zeros((num, 1)))


if __name__ == '__main__':
    data_dirs = glob(os.path.join(DATA_DIR, 'HWDB/discrete/*'))
    batch_size = 16
    data = WindowedData(data_dirs, batch_size)
    data.on_epoch_end()

    obs, (action, value) = data.__getitem__(0)
    for i in range(batch_size):
        # plt.imshow(obs[i, :, :, :3])
        img = obs[i]
        plt.imshow(np.concatenate([img[:, :, 0], img[:, :, 1], img[:, :, 2]], axis=-1))

        plt.suptitle(action[i])
        plt.show()
