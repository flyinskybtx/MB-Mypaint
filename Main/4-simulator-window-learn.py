import ray
import tqdm
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from Env.continuous_env import SimulatorContinuousEnv
from Main import load_config
from Model.policy_model import CustomPolicyModel

if __name__ == '__main__':
    cfg = load_config()

    ray.shutdown(True)  # Shut down and restart Ray
    ray.init(num_gpus=1)

    model_name = "WindowModel"
    env_name = "ContinuousEnv-v2"
    ModelCatalog.register_custom_model(model_name, CustomPolicyModel)
    register_env(env_name, lambda _config: SimulatorContinuousEnv(_config))

    config = cfg.ray_config
    config.env_config = cfg.continuous_env_config
    config.env_config.brush_config = cfg.brush_config
    config.model = {
        'custom_model': model_name,
        'layers': cfg.policy_model_config.direct_cnn_layers,
    }

    algorithm = 'PPO'

    ray.tune.run(
        algorithm,
        reuse_actors=True,
        checkpoint_at_end=True,
        verbose=1,
        stop={
            'training_iteration': 100,
        },
        config={
            'env': env_name,
            'env_config': config.env_config,
            'model': {
                'custom_model': model_name,
                'custom_model_config': {
                    'layers': cfg.policy_model_config.direct_cnn_layers,
                }
            },
            'num_workers': 1,
            'create_env_on_driver': True,
            "lr": ray.tune.loguniform(1e-5, 1e-2),
            # "reuse_actors": True,
            "log_level":"INFO",
            "ignore_worker_failures": True,
            "batch_mode": 'complete_episodes',
        },
    )
