import random
import string

import ray
from ray.rllib import rollout
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Data.Deprecated.core_config import *
# Settings
from Data.Deprecated.windowed_env import WindowedCnnEnv
from Data.Deprecated.cnn_model import LayerConfig
from Data.Deprecated.windowd_cnn_model import WindowedCnnModel

if __name__ == '__main__':
    random.seed(0)
    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=1, temp_dir='/home/flyinsky/ray_tmp')
    print('Start Ray')

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, WindowedCnnModel)
    register_env('WindowedCnnEnv-v0', lambda env_config: WindowedCnnEnv(env_config))

    config = {
        "train_batch_size": 200,
        'num_workers': 0,
        'log_level': 'INFO',
        'framework': 'tf',
        #  -------------------------------------
        'brush_config': {
            'image_size': experimental_config.image_size,
            'window_size': experimental_config.window_size,
            'z_size': experimental_config.z_size,
            'brush_name': experimental_config.brush_name,
            'image_nums': experimental_config.image_nums,
            'action_shape': experimental_config.action_shape,
        },
        #  -------------------------------------
        'model': {
            'custom_model': model_name,
            "custom_model_config": {
                'blocks': [
                    LayerConfig(conv=[64, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                                pool=[2, 2, 'same'], dropout=0.5),
                    LayerConfig(conv=[32, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                                pool=[2, 2, 'same'], dropout=0.5),
                    LayerConfig(conv=[16, (2, 2), 1], padding='same', activation='relu', dropout=0.5),
                    LayerConfig(flatten=True),
                    LayerConfig(fc=4096, activation='relu', dropout=0.5),
                    LayerConfig(fc=2048, activation='relu', dropout=0.5),
                    LayerConfig(fc=1024, activation='relu', dropout=0.5),
                ],
                'offline_dataset': '../Data/offline/windowed'
            },
        },
        # ----------------------------------------------

    }

    a2c_trainer = a3c.A2CTrainer(config=config, env='WindowedCnnEnv-v0')
    a2c_trainer.import_model('../Model/checkpoints/supervised_windowed_model.h5')

    for i in tqdm(range(1000)):
        result = a2c_trainer.train()
        print(f"\t Episode Reward: "
              f"{result['episode_reward_max']:.4f}  |  "
              f"{result['episode_reward_mean']:.4f}  |  "
              f"{result['episode_reward_min']:.4f}")

        if i % 10 == 0:
            checkpoint = a2c_trainer.save()
            print("checkpoint saved at", checkpoint)
        if i % 50 == 0:
            rollout.rollout(a2c_trainer, env_name='WindowedCnnEnv-v0', num_steps=50, num_episodes=1, no_render=False)
