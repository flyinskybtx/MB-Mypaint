import random
import string

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.offline import JsonReader

from Model.cnn_model import CnnModel, LayerConfig
from Model.supervised_cnn_model import MaskedMultiCategorical


class WindowedCnnModel(CnnModel):
    def custom_loss(self, policy_loss, loss_inputs):
        reader = JsonReader(self.model_config['custom_model_config']['offline_dataset'])
        input_ops = reader.tf_input_ops()

        obs = restore_original_dimensions(tf.cast(input_ops['obs'], tf.float32), self.obs_space)
        logits, _ = self.forward({'obs': obs}, [], None)
        action_dist = MaskedMultiCategorical(logits, self.model_config, input_lens=self.action_space.nvec)
        # actions = tf.cast(action_dist.deterministic_sample(), tf.float32)
        #  masked_mse = tf.reduce_mean(tf.boolean_mask(
        #  tf.square(actions - tf.cast(input_ops['actions'], tf.float32)), mask))
        mask = np.array([True, True, False])
        logp = -action_dist.masked_logp(input_ops['actions'], mask)

        self.imitation_loss = tf.reduce_mean(logp)
        self.policy_loss = policy_loss

        return policy_loss + self.imitation_loss

    def custom_stats(self):
        return {
            'policy_loss': self.policy_loss,
            'imitation_loss': self.imitation_loss,
        }


if __name__ == '__main__':
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, WindowedCnnModel)

    model_config = {
        'custom_model': model_name,
        "custom_model_config": {
            'blocks': [
                LayerConfig(conv=[16, [5, 5], 3], batch_norm=True, activation='relu', pool=[2, 2, 'same'], dropout=0.3),
                LayerConfig(conv=[32, [3, 3], 1], activation='relu', pool=[2, 2, 'same']),
                LayerConfig(flatten=True),
                LayerConfig(fc=64, activation='linear', dropout=0.2),
            ]
        },
        'offline_dataset': '../Data/offline/windowed',
    }

    observation_space = gym.spaces.Box(low=0,
                                       high=1,
                                       shape=(192, 192, 1),
                                       dtype=np.float)
    action_space = gym.spaces.MultiDiscrete([28, 28, 10] * 6)
    num_outputs = 18
    name = 'test_windowed_cnn'

    model = WindowedCnnModel(observation_space, action_space, num_outputs, model_config, name)
