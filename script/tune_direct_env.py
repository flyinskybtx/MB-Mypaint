import random
import string

import ray
from ray.rllib import rollout
from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from tqdm import tqdm

from Data.data_process import load_stroke_png, preprocess_stroke_png, extract_skeleton_trace
# Settings
from Env.direct_env import DirectCnnEnv
from Model.cnn_model import CnnModel, LayerConfig

image_size = 192 * 4
roi_grid_size = 16 * 2  # Each roi contains 3*3 roi grids
pixels_per_grid = 3
z_grid_size = 1 / 10
num_keypoints = 6

ori_img = load_stroke_png(11)
print(f'Shape of origin image is {ori_img.shape}')

preprocessed_img = preprocess_stroke_png(ori_img, image_size=image_size)
print(f'Shape of preprocessed image is {preprocessed_img.shape}')

reference_path = extract_skeleton_trace(preprocessed_img, roi_grid_size)

env_config = {
    'image_size': image_size,
    'roi_grid_size': roi_grid_size,
    'pixels_per_grid': pixels_per_grid,
    'z_grid_size': z_grid_size,
    'brush_name': 'custom/slow_ink',
    'num_keypoints': num_keypoints,
    'target_image': preprocessed_img,
}

model_config = {
    'custom_model': 'model_name',
    "custom_model_config": {
        'blocks': [
            LayerConfig(conv=[128, (2, 2), 1], padding='same', batch_norm=True, activation='relu',
                        pool=[2, 2, 'same'], dropout=0.5),
            LayerConfig(conv=[64, (2, 2), 1], padding='same', batch_norm=True, activation='relu',
                        pool=[2, 2, 'same'], dropout=0.5),
            LayerConfig(conv=[32, (2, 2), 1], padding='same', activation='relu', dropout=0.5),
            LayerConfig(flatten=True),
            LayerConfig(fc=4096, activation='relu', dropout=0.5),
            LayerConfig(fc=2048, activation='relu', dropout=0.5),
            LayerConfig(fc=1024, activation='relu', dropout=0.5),
        ]
    },
}

if __name__ == '__main__':
    model_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ModelCatalog.register_custom_model(model_name, CnnModel)
    model_config['custom_model'] = model_name

    ray.shutdown(True)
    ray.init(num_gpus=1, temp_dir='/home/baitianxiang/ray_tmp')

    config = {
        'env_config': env_config,
        'num_workers': 2,
        'log_level': 'ERROR',
        'framework': 'tf',
        'model': model_config,
    }

    register_env('DirectCnnEnv-v0', lambda env_config: DirectCnnEnv(env_config))

    a2c_trainer = a3c.A2CTrainer(config=config, env='DirectCnnEnv-v0')

    policy = a2c_trainer.get_policy()
    cur_model = policy.model.base_model
    cur_model.summary()

    for i in tqdm(range(1000)):
        result = a2c_trainer.train()
        print(f"{result['episode_reward_max']:.4f}  |  "
              f"{result['episode_reward_mean']:.4f}  |  "
              f"{result['episode_reward_min']:.4f}")

        if i % 10 == 0:
            checkpoint = a2c_trainer.save()

            print("checkpoint saved at", checkpoint)
            rollout.rollout(a2c_trainer, env_name='DirectCnnEnv-v0', num_steps=1, no_render=False)
