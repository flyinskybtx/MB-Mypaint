import glob
import os.path as osp
import random
from collections import defaultdict

import numpy as np
from tensorflow import keras
from ray.rllib.offline.json_reader import _from_json
from tqdm import tqdm

from Data import DATA_DIR

MAXMIMUM_SAMPLES = 1e5


class BaseData(keras.utils.Sequence):
    selected_keys = ['obs', 'new_obs', 'actions', 'infos']

    def __init__(self, savedir, batch_size, **kwargs):
        """
        :param savedir:
        :param batch_size:
        """
        self.savedir = osp.join(DATA_DIR, savedir)
        self.batch_size = batch_size
        self.data = {}
        self._load_all()
        self.index = 0
        self.shuffle()

    def _load_all(self):
        self.data = {}
        files = glob.glob(self.savedir + '/*.json')
        all_data = []
        for filename in tqdm(files):
            all_data.append(self.load_file(filename))
        all_data = self._merge_dict(all_data)

        for key, values in all_data.items():
            data = self._merge_dict(values)
            del data['infos']
            for k, v in data.items():
                new_v = [np.vsplit(arr, arr.shape[0]) for arr in v]
                new_v = [item for items in new_v for item in items]
                data[k] = new_v
            self.data[key] = data

    def load_file(self, filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            samples = [_from_json(ll) for ll in lines]
            samples = [self._select_keys(ss) for ss in samples]
        data = defaultdict(list)
        for sample in samples:
            data[sample['infos']['config_num']].append({k: v for k, v in sample.items()})
        return data

    def shuffle(self):
        for _, v in self.data.items():
            # idx = np.arange(len(v['actions']))
            # np.random.shuffle(idx)
            for key, value in v.items():
                random.shuffle(value)
                v[key] = value

    def on_epoch_end(self):
        self.shuffle()

    def _get_batch(self, i, batch_size):
        key = random.choice(list(self.data.keys()))
        batch = {k: np.concatenate(v[i * batch_size: (i + 1) * batch_size], axis=0) for k, v in self.data[key].items()}
        return batch

    def _select_keys(self, sample: dict):
        selected = {k: v for k, v in sample.items() if k in self.selected_keys}
        selected['infos'] = selected['infos'][0]
        return selected

    @staticmethod
    def _merge_dict(data_list):
        res = defaultdict(list)
        for data in data_list:
            for key, value in data.items():
                if isinstance(value, list):
                    res[key] += value
                else:
                    res[key].append(value)
        return res

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def next(self):
        batch = self.__getitem__(self.index)
        self.index += 1
        if self.index >= self.__len__():
            self.index = 0
        return batch
