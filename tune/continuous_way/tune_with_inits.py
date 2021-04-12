import os
import random
import string
from glob import glob
import matplotlib.pyplot  as plt
import numpy as np
import ray
import tqdm
from ray.rllib import rollout
from ray.rllib.agents import pg, ddpg, a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.tune import register_env
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian, DiagGaussian, Deterministic, TFActionDistribution

_, tf, _ = try_import_tf()

from Data import DATA_DIR
from Env.continuous_env import ContinuousEnv
from Model import LayerConfig, MODEL_DIR
from Model.policy_model import CustomPolicyModel
from script.main_procs.hparams import define_hparams


class CustomSquashedGaussian(DiagGaussian):
    def __init__(self, inputs, model, low=0, high=1):
        self.low = low
        self.high = high
        super().__init__(inputs, model)

        mean, std = tf.split(inputs, 2, axis=1)
        self.mean = mean
        self.std = std
        self.sample_op = self._build_sample_op()
        self.sampled_action_logp_op = self.logp(self.sample_op)

    def _squash(self, raw_values):
        return tf.clip_by_value(raw_values, self.low, self.high)

    @override(TFActionDistribution)
    def _build_sample_op(self):
        return self._squash(self.mean + self.std * tf.random.normal(tf.shape(self.mean)))


if __name__ == '__main__':
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=2,
             # temp_dir='/home/flyinsky/ray_tmp'
             )
    print('Start Ray')

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, CustomPolicyModel)
    ModelCatalog.register_custom_action_dist("squashed_gaussian", CustomSquashedGaussian)
    ModelCatalog.register_custom_action_dist("deterministic", Deterministic)
    ModelCatalog.register_custom_action_dist("diag_gaussian", DiagGaussian)

    register_env('ContinuousEnv-v1', lambda _config: ContinuousEnv(_config))

    config = {
        'num_workers': 1,
        "train_batch_size": 200,
        'log_level': 'INFO',
        'framework': 'tf',
        # ----------------------------------------------obs
        'brush_config': env_config,
        # ----------------------------------------------
        'model': {
            'custom_model': model_name,
            "custom_model_config": {
                'layers': [
                    LayerConfig(type='batch_norm'),
                    LayerConfig(type='conv', filters=32, kernel_size=(2, 2), strids=1, activation='relu'),
                    LayerConfig(type='pool', pool_size=2, strides=2),
                    LayerConfig(type='batch_norm'),
                    LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
                    LayerConfig(type='pool', pool_size=2, strides=2),
                    LayerConfig(type='batch_norm'),
                    LayerConfig(type='conv', filters=16, kernel_size=(2, 2), strids=2, activation='relu'),
                    LayerConfig(type='pool', pool_size=2, strides=2),
                    LayerConfig(type='flatten'),
                    LayerConfig(type='dropout', rate=0.2),
                    LayerConfig(type='dense', units=1024, activation='relu'),
                ]
            },
            "custom_action_dist": "diag_gaussian"
        },
        # ----------------------------------------------

    }

    trainer = pg.PGTrainer(config=config, env='ContinuousEnv-v1')
    print('Created A2C Trainer for ContinuousEnv-v1')

    weights_file = os.path.join(MODEL_DIR, 'checkpoints/continuous_policy.h5')
    if os.path.exists(weights_file):
        trainer.import_model(weights_file)
        print('Imported supervised model as start point')

    for i in tqdm.trange(1000):
        if i % 50 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 50 == 0:
            rollout.rollout(trainer, env_name='ContinuousEnv-v1',
                            num_steps=env_config.max_step, num_episodes=3,
                            no_render=False)
        result = trainer.train()
        print(f"\t Episode Reward: "
              f"{result['episode_reward_max']:.4f}  |  "
              f"{result['episode_reward_mean']:.4f}  |  "
              f"{result['episode_reward_min']:.4f}")
