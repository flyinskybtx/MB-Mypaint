import numpy as np
import tensorflow as tf
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.offline import JsonReader

from Model.cnn_model import CnnModel


class MaskedMultiCategorical(MultiCategorical):
    def masked_logp(self, actions, mask):
        if isinstance(actions, tf.Tensor):
            actions = tf.unstack(tf.cast(actions, tf.int32), axis=1)
        logps = tf.stack(
            [cat.logp(act) for cat, act in zip(self.cats, actions)])
        masked_logps = tf.boolean_mask(logps, mask)
        return tf.reduce_sum(masked_logps, axis=0)


class SupervisedCnnModel(CnnModel):
    def custom_loss(self, policy_loss, loss_inputs):
        reader = JsonReader(self.model_config['custom_model_config']['offline_dataset'])
        input_ops = reader.tf_input_ops()

        supervised_obs = restore_original_dimensions(tf.cast(input_ops['obs'], tf.float32), self.obs_space)
        logits, _ = self.forward({'obs': supervised_obs}, [], None)
        action_dist = MaskedMultiCategorical(logits, self.model_config, input_lens=self.action_space.nvec)
        mask = np.array([True, True, False] * int(len(self.action_space.nvec) / 3))
        logp = -action_dist.masked_logp(input_ops['actions'], mask)

        self.imitation_loss = tf.reduce_mean(logp)
        self.policy_loss = policy_loss

        return policy_loss + self.imitation_loss

    def custom_stats(self):
        return {
            'policy_loss': self.policy_loss,
            'imitation_loss': self.imitation_loss,
        }
