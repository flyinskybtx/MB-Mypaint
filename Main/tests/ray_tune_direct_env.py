import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from Env.direct_env_v2 import RobotDirectEnvV2
from Main import load_config
from Main.algo_params import algo_trainers
from Model.policy_model import CustomPolicyModel

if __name__ == '__main__':
    ray.init(num_gpus=1)
    cfg = load_config()

    model_name = "CnnModel"
    env_name = "RobotDirectEnv-v2"
    algorithm = "PPO"
    ModelCatalog.register_custom_model(model_name, CustomPolicyModel)
    register_env(env_name, lambda _config: RobotDirectEnvV2(_config))

    config = cfg.ray_config
    config.env = env_name
    config.env_config = cfg.direct_env_config
    config.env_config.brush_config = cfg.brush_config
    config.model = {
        'custom_model': model_name,
        "custom_model_config": {
            'layers': cfg.policy_model_config.direct_cnn_layers,
        }
    }

    resources = algo_trainers[algorithm].default_resource_request(config).to_json()
    # config.update(algo_params[algorithm])

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
            # "lr": 0.001,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            # "reuse_actors": True,
            "log_level":"INFO",
            "ignore_worker_failures": True,
            "batch_mode": 'complete_episodes',
        },
    )
