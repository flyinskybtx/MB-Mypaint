import json
import os
from collections import defaultdict
from glob import glob
from os import path as osp

import numpy as np
from ray.rllib.offline.json_reader import _from_json
from tensorflow import keras
from tqdm import tqdm

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = osp.join(DATA_DIR, 'png')
RAY_RESULTS = '/home/flyinsky/ray_results'


class NewBaseData(keras.utils.Sequence):
    def __init__(self, data_dirs, batch_size, **kwargs):
        """
        :param savedir:
        :param batch_size:
        """
        self.data_dirs = data_dirs
        self.batch_size = batch_size

        self.data = self.load_data(self.data_dirs)

    def load_data(self, dirs):
        data = {}
        for data_dir in tqdm(dirs, desc="Load data directories"):
            name = data_dir.split('/')[-1]
            try:
                data[name] = self._load_dir(data_dir)
            except json.decoder.JSONDecodeError:
                print(f'Error: {name} has a Json Decode Problem')
        lengths = [len(list(v.values())[0]) for v in data.values()]
        print(
            f"Loaded data has {len(list(data.keys()))} simulator data, with length between {np.min(lengths)} - {np.max(lengths)}")
        return data

    @staticmethod
    def _load_dir(data_dir):
        """

        :param data_dir:
        :return: list
        """
        data = defaultdict(list)
        for filename in glob(os.path.join(data_dir, '*.json')):
            with open(filename, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    sample = _from_json(line)
                    for k, v in sample.items():
                        data[k].append(v)

        return {k: np.concatenate(v, axis=0) for k, v in data.items()}

    @staticmethod
    def shuffle(data):
        for key, value in data.items():
            idx = np.arange(len(value[list(value.keys())[0]]))
            np.random.shuffle(idx)
            value = {k: v[idx] for k, v in value.items()}
            data[key] = value
        return data

    def on_epoch_end(self):
        self.data = self.shuffle(self.data)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


def obs_to_delta(obs):
    return np.expand_dims(obs[:, :, :, 0] - obs[:, :, :, 1], axis=-1)


def _merge_dict(data_list):
    res = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            if isinstance(value, list):
                res[key] += value
            else:
                res[key].append(value)
    return res
