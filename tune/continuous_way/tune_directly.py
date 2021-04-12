import os
import random
import string
from glob import glob

import ray
import tqdm
from ray.rllib import rollout
from ray.rllib.agents import pg, a3c, ddpg
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.models.tf.tf_action_dist import SquashedGaussian, DiagGaussian, Deterministic
from Data import DATA_DIR
from Env.continuous_env import ContinuousEnv
from Model import LayerConfig
from Model.policy_model import CustomPolicyModel
from script.main_procs.hparams import define_hparams

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
    ModelCatalog.register_custom_action_dist("squashed_gaussian", SquashedGaussian)
    ModelCatalog.register_custom_action_dist("deterministic", Deterministic)
    ModelCatalog.register_custom_action_dist("diag_gaussian", DiagGaussian)

    register_env('ContinuousEnv-v1', lambda _config: ContinuousEnv(_config))

    config = {
        'num_workers': 10,
        "train_batch_size": 200,
        'log_level': 'INFO',
        'framework': 'tf',
        # ----------------------------------------------
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

    trainer = ddpg.DDPGTrainer(config=config, env='ContinuousEnv-v1')
    print('Created Trainer for ContinuousEnv-v1')

    for i in tqdm.trange(1000):
        if i % 50 == 0 and i > 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 50 == 0:
            rollout.rollout(trainer, env_name='ContinuousEnv-v1', num_steps=env_config.max_step, num_episodes=3,
                            no_render=False)
        result = trainer.train()
        print(f"\t Episode Reward: "
              f"{result['episode_reward_max']:.4f}  |  "
              f"{result['episode_reward_mean']:.4f}  |  "
              f"{result['episode_reward_min']:.4f}")
