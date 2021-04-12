from ray import tune
from ray.rllib.agents import ppo

algo_trainers = {
    "PPO": ppo.PPOTrainer
}

algo_params = {
    'PPO': {
        'use_critic': True,
        'use_gae': True,
        'lambda': tune.uniform(0.5, 1.5),  # 1.0
        'kl_coeff': tune.uniform(0.1, 0.3),  # 0.2
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "shuffle_sequences": True,
        "lr": tune.loguniform(1e-1, 1e-5),  # 5e-5
        "lr_schedule": None,
        "vf_share_layers": True,
        "vf_loss_coeff": tune.uniform(0.5, 1.2),  # 1.0
        "entropy_coeff": 0.0,
        "entropy_coeff_schedule": None,
        "clip_param": 0.3,
        "vf_clip_param": 10.0,
        "grad_clip": None,
        "kl_target": 0.01,
    }
}
