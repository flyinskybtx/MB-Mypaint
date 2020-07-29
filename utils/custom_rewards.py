# Rewards:
import numpy as np
from sklearn import metrics


def cos_sim_reward(tar, obs):
    value = metrics.pairwise.cosine_similarity(tar.reshape(1, -1), obs.reshape(1, -1))[0, 0]
    if value == np.nan:
        raise ValueError
        # return -1
    return value
