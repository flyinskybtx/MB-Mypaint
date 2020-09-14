import glob
import os
import os.path as osp

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from tqdm import trange

from Data import DATA_DIR
from Data.data_process_lib import refpath_to_actions
from Env.main_env import MainEnv
from script.main_procs.hparams import define_hparams
from script.main_procs.make_simulation_environments import sample_env_configs


def remove_data(savedir, pattern='*.json'):
    """ Remove old data according to pattern in 'savedir'

    :param pattern:
    :return:
    """
    savedir = osp.join(DATA_DIR, savedir)
    old_files = glob.glob(f"{savedir}/{pattern}")
    for f in old_files:
        os.remove(f)
        print(f'Removed file: {f}')


def rollout_and_collect_data(env_config, num_episodes, savedir):
    savedir = osp.join(DATA_DIR, savedir)
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(savedir)

    rewards = []

    env = MainEnv(env_config)
    for eps_id in trange(num_episodes):
        eps_rew = 0
        obs = env.reset()
        done = False
        t = 0
        ref_actions = refpath_to_actions(env.cur_ref_path,
                                         env_config.xy_grid,
                                         env_config.action_shape)
        for action in ref_actions:
            assert np.min(action) >= 0 and np.max(action) < env_config.action_shape
            if done:
                break
            new_obs, rew, done, info = env.step(action)
            if np.max(new_obs[:, :, 0]) > 0 or np.max(obs[:, :, 0]) > 0:  # If not empty image
                batch_builder.add_values(
                    t=t,
                    eps_id=eps_id,
                    obs=obs,
                    new_obs=new_obs,
                    actions=action,
                    dones=done,
                    infos={'config_num': env_config.config_num},
                )
            obs = new_obs.copy()
            # delta = obs[:, :, 0] - obs[:, :, 1]
            # plt.imshow(delta)
            # plt.show()
            t += 1
            eps_rew += rew
        rewards.append(eps_rew)
        if batch_builder.buffers:  # Buffer is not empty
            writer.write(batch_builder.build_and_reset())

    return [np.min(rewards), np.mean(rewards), np.max(rewards)]


def collect_sim_data(configs, num_episodes, remove_old=False):
    if remove_old:
        remove_data('offline/random')
    for i, config in enumerate(configs, start=1):
        print(f"Proceesing {i}/{len(configs)}")
        rewards = rollout_and_collect_data(config, num_episodes=num_episodes, savedir='offline/random')


if __name__ == '__main__':
    cfg = define_hparams()
    sim_configs = sample_env_configs(cfg.num_simulators, resample=False)
    collect_sim_data(sim_configs, cfg.rollout_episodes, remove_old=True)
