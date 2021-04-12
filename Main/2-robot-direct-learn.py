import os
from glob import glob

import ray
import tqdm
from ray.rllib import rollout
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from Data import RAY_RESULTS
from Env.direct_env import RobotDirectEnv
from Main import load_config
from Model.policy_model import CustomPolicyModel

if __name__ == '__main__':
    cfg = load_config()

    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=1)

    model_name = "CnnModel"
    env_name = "RobotDirectEnv-v1"
    ModelCatalog.register_custom_model(model_name, CustomPolicyModel)
    register_env(env_name, lambda _config: RobotDirectEnv(_config))

    config = cfg.ray_config
    config.num_workers = 0
    config.env_config = cfg.direct_env_config
    config.env_config.brush_config = cfg.brush_config
    config.model = {
        'custom_model': model_name,
        "custom_model_config": {
            'layers': cfg.policy_model_config.direct_cnn_layers,
            # 'layers': cfg.policy_model_config.direct_mlp_layers,
        }
    }

    # trainer = a3c.A2CTrainer(config=config, env=env_name)
    trainer = ppo.PPOTrainer(config=config, env=env_name)
    # trainer = ars.ARSTrainer(config=config, env=env_name)
    print(f'Created A2C Trainer for {env_name}')

    policy_model_name = "direct_robot_policy"
    # policy_model_name = "direct_mlp_policy"

    # weights_file = os.path.join(MODEL_DIR, f'checkpoints/{policy_model_name}.h5')
    # if os.path.exists(weights_file):
    #     trainer.import_model(weights_file)
    #     print('Imported supervised model as start point')

    checkpoint_name = 'PPO_RobotDirectEnv_Baseline'
    _number = sorted(glob(os.path.join(RAY_RESULTS, checkpoint_name, 'checkpoint_*')))[-1].split('_')[-1]
    checkpoint_file = os.path.join(RAY_RESULTS, checkpoint_name, f'checkpoint_{_number}', f'checkpoint-{_number}')
    print(f"Load checkpoint from {checkpoint_file}")
    trainer.restore(checkpoint_file)
    rollout.rollout(trainer, env_name=env_name, num_steps=10, num_episodes=10, no_render=False)

    # for i in tqdm.trange(1000):
    #     if i % 50 == 0:
    #         rollout.rollout(trainer, env_name=env_name, num_steps=10, num_episodes=3, no_render=False)

        # result = trainer.train()
        # print(f"\t Episode Reward: "
        #       f"{result['episode_reward_max']:.4f}  |  "
        #       f"{result['episode_reward_mean']:.4f}  |  "
        #       f"{result['episode_reward_min']:.4f}")
        #
        # if i % 50 == 0 and i > 0:
        #     checkpoint = trainer.save()
        #     print("checkpoint saved at", checkpoint)
