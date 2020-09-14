import os

MODEL_DIR = os.path.abspath(os.path.dirname(__file__))


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
