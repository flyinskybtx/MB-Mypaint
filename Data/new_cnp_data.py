import random

import numpy as np
from tensorflow import keras

from Data import obs_to_delta, NewBaseData


class NewCNPData(NewBaseData):
    def __init__(self, data_dirs, batch_size, num_context: tuple, encoder: keras.Model, embedder=None, steps=None,
                 **kwargs):
        super().__init__(data_dirs, batch_size, **kwargs)
        self.min_num_context, self.max_num_context = num_context
        self.num_context = np.random.randint(self.min_num_context, self.max_num_context)

        self.train = kwargs.setdefault('train', False)
        self.encoder = encoder
        self.embedder = embedder

        if steps is not None:
            for sim in self.data.keys():
                self.data[sim] = {k: v[:steps * batch_size] for k, v in self.data[sim].items()}

        self.on_epoch_end()

    def _get_batch(self, index, batch_size):
        sim = random.choice(list(self.data.keys()))
        data = self.data[sim]
        batch_data = {k: v[index * batch_size: (index + 1) * batch_size] for k, v in data.items()}
        return batch_data

    def __getitem__(self, index):
        if self.train:
            batch_data = self._get_batch(index % self.__len__(), np.max([self.batch_size, self.num_context]))
        else:
            batch_data = self._get_batch(index % self.__len__(), np.sum([self.batch_size, self.num_context]))

        obs = batch_data['obs']
        new_obs = batch_data['new_obs']
        actions = batch_data['actions']

        x0 = self.encoder(obs_to_delta(obs))
        x1 = self.encoder(obs_to_delta(new_obs))
        if self.embedder is not None:
            u = self.embedder(actions)
        else:
            u = actions
        dx = x1 - x0
        xu = np.concatenate([x0, u], axis=-1)

        if self.train:
            context_x = np.repeat(np.expand_dims(xu[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(dx[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = xu[-self.batch_size:]  # If train, include context points
            target_y = dx[-self.batch_size:]
        else:
            context_x = np.repeat(np.expand_dims(xu[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            context_y = np.repeat(np.expand_dims(dx[:self.num_context], axis=0),
                                  self.batch_size, axis=0)
            query_x = xu[-self.batch_size:]  # If test, select other points as query
            target_y = dx[-self.batch_size:]

        return [context_x, context_y, query_x], target_y

    def __len__(self):
        min_sim_length = np.min([len(list(v.values())[0]) for v in self.data.values()])
        if self.train:
            return int(min_sim_length // np.max([self.batch_size, self.num_context])) - 1
        else:
            return int(min_sim_length // np.sum([self.batch_size, self.num_context])) - 1

    def on_epoch_end(self):
        self.data = self.shuffle(self.data)
