import numpy as np


class ActionEmbedder:
    def __init__(self, config):
        self.action_shape = config.action_shape

    def _embedding(self, action):
        embedding = np.zeros(3 * self.action_shape)
        for i, act in enumerate(action):
            embedding[i * self.action_shape + int(act)] = 1
        return embedding

    def transform(self, actions):
        return np.stack(self._embedding(action) for action in actions)
