from collections import defaultdict

import numpy as np

from Data import BaseData


class AEGenerator(BaseData):
    def __init__(self, savedir, batch_size, **kwargs):
        super().__init__(savedir, batch_size, **kwargs)
        self.is_encoder = kwargs.get('is_encoder')
        self.max_samples = kwargs.setdefault('max_samples', None)
        self._all_in_one()

    def _all_in_one(self):
        all_data = defaultdict(list)
        for subset in self.data.values():
            for k, v in subset.items():
                all_data[k] += v
        for k, v in all_data.items():
            if self.max_samples is not None:
                all_data[k] = v[:int(self.max_samples)]

        self.data = {'all': all_data}

    def __getitem__(self, index):
        batch = self._get_batch(index, self.batch_size)

        if self.is_encoder:
            delta_cur = np.expand_dims(batch['obs'][:, :, :, 0] - batch['obs'][:, :, :, 1], axis=-1)  # cur - prev
            return delta_cur, delta_cur
        else:
            delta_cur = np.expand_dims(batch['obs'][:, :, :, 0] - batch['obs'][:, :, :, 1], axis=-1)  # cur - prev
            delta_new = np.expand_dims(batch['new_obs'][:, :, :, 0] - batch['new_obs'][:, :, :, 1],
                                       axis=-1)  # new - cur
            return [delta_cur, delta_new], delta_new

    def __len__(self):
        min_len = np.min([len(v['actions']) for v in self.data.values()])
        return min_len // self.batch_size - 1


