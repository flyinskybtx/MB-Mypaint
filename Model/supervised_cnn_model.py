import random
import string

import gym
import numpy as np
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import MultiCategorical

from Model.cnn_model import CnnModel, LayerConfig


class SupervisedCnnModel(CnnModel):
    def custom_loss(self, policy_loss, loss_inputs):
        supervised_action = self.model_config['custom_model_config']['supervised_action'].reshape(1, -1)
        mask = np.reshape(supervised_action != 0, (-1,))
        logits, _ = self.forward({"obs": loss_inputs['obs']}, [], None)

        supervised_action = tf.repeat(supervised_action, repeats=tf.shape(logits)[0], axis=0)

        action_dist = MultiCategorical(logits, self.model_config, input_lens=self.action_space.nvec)
        logp = -action_dist.logp(supervised_action)
        masked_logp = tf.boolean_mask(logp, mask)

        self.imitation_loss = tf.reduce_mean(masked_logp)
        self.policy_loss = policy_loss

        return policy_loss + self.imitation_loss

    def custom_stats(self):
        return {
            'policy_loss': self.policy_loss,
            'imitation_loss': self.imitation_loss,
        }


if __name__ == '__main__':
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, SupervisedCnnModel)

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
        'supervised_action': np.zeros((18,))
    }

    observation_space = gym.spaces.Box(low=0,
                                       high=1,
                                       shape=(192, 192, 1),
                                       dtype=np.float)
    action_space = gym.spaces.MultiDiscrete([28, 28, 10] * 6)
    num_outputs = 18
    name = 'test_custom_cnn'

    model = CnnModel(observation_space, action_space, num_outputs, model_config, name)
