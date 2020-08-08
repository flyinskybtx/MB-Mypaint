import random
import string

import ray
from ray.rllib import rollout
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Env.core_config import experimental_config
# Settings
from Env.direct_env import DirectCnnEnv
from Model.cnn_model import LayerConfig
from Model.supervised_cnn_model import SupervisedCnnModel

if __name__ == '__main__':
    import_supervised_model = 'supervised_direct_model'

    random.seed(0)
    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=1, temp_dir='/home/flyinsky/ray_tmp')
    print('Start Ray')

    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, SupervisedCnnModel)
    register_env('DirectCnnEnv-v0', lambda env_config: DirectCnnEnv(env_config))

    config = {
        'num_workers': 0,
        "train_batch_size": 200,
        'log_level': 'INFO',
        'framework': 'tf',
        # ----------------------------------------------
        'env_config': {
            'image_size': experimental_config.image_size,
            'stride_size': experimental_config.stride_size,
            'stride_amplify': experimental_config.stride_amplify,
            'z_size': experimental_config.z_size,
            'brush_name': experimental_config.brush_name,
            'num_keypoints': experimental_config.num_keypoints,
            'image_nums': experimental_config.image_nums,
        },
        # ----------------------------------------------
        'model': {
            'custom_model': model_name,
            "custom_model_config": {
                'blocks': [
                    LayerConfig(conv=[128, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                                pool=[2, 2, 'same'], dropout=0.5),
                    LayerConfig(conv=[64, (2, 2), 1], padding='same', batch_norm=False, activation='relu',
                                pool=[2, 2, 'same'], dropout=0.5),
                    LayerConfig(conv=[32, (2, 2), 1], padding='same', activation='relu', dropout=0.5),
                    LayerConfig(flatten=True),
                    LayerConfig(fc=4096, activation='relu', dropout=0.5),
                    LayerConfig(fc=2048, activation='relu', dropout=0.5),
                    LayerConfig(fc=1024, activation='relu', dropout=0.5),
                ],
                'offline_dataset': '../Data/offline/direct'
            },
        },
        # ----------------------------------------------

    }
    a2c_trainer = a3c.A2CTrainer(config=config, env='DirectCnnEnv-v0')
    print('Created A2C Trainer for DirectCnnEnv-v0')

    if import_supervised_model:
        a2c_trainer.import_model(f'../Model/checkpoints/{import_supervised_model}.h5')
        print('Imported supervised model as start point')

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
            rollout.rollout(a2c_trainer, env_name='DirectCnnEnv-v0', num_steps=1, no_render=False)
