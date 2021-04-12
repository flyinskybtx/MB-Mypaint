import os
import random
import string
from glob import glob

import ray
import tqdm
from ray.rllib import rollout
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from Data import DATA_DIR
from Env.discrete_env import DiscreteEnv
from Model import LayerConfig, MODEL_DIR
from Model.policy_model import CustomPolicyModel
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=1)
    print('Start Ray')

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, CustomPolicyModel)
    register_env('DiscreteEnv-v1', lambda _config: DiscreteEnv(_config))

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
        },
        # ----------------------------------------------

    }

    trainer = a3c.A2CTrainer(config=config, env='DiscreteEnv-v1')
    print('Created A2C Trainer for DiscreteEnv-v1')

    weights_file = os.path.join(MODEL_DIR, 'checkpoints/discrete_policy.h5')
    if os.path.exists(weights_file):
        trainer.import_model(weights_file)
        print('Imported supervised model as start point')

    for i in tqdm.trange(1000):
        result = trainer.train()
        print(f"\t Episode Reward: "
              f"{result['episode_reward_max']:.4f}  |  "
              f"{result['episode_reward_mean']:.4f}  |  "
              f"{result['episode_reward_min']:.4f}")

        if i % 50 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 50 == 0:
            rollout.rollout(trainer, env_name='DiscreteEnv-v1',
                            num_steps=env_config.max_step, num_episodes=3,
                            no_render=False)
