from skimage import transform

from Data.HWDB.load_HWDB import get_skeleton_paths_from_pot, HWDB_DIR
from Data.data_process_lib import cut_roi, skeleton_path_to_wps
from Env.continuous_env import ContinuousEnv
from script.main_procs.hparams import define_hparams
import numpy as np
import os


def random_actions_from_skeleton(skeleton_path, xy_grid, image_size):
    start_point = skeleton_path[0][0]
    _, action_xy = skeleton_path_to_wps(skeleton_path, xy_grid, image_size, discrete=False)
    action_z = np.random.random(size=(action_xy.shape[0], 1))
    actions = np.concatenate([action_xy, action_z], axis=-1)
    return actions, start_point


def collect_data_from_actions(env, actions, start_point):
    obs = env.reset()
    env.target_image = np.zeros_like(env.target_image)
    env.position = (start_point[0], start_point[1], 0)

    imgs = [obs]
    points = []

    for action in actions:
        point = env.position
        points.append(point)
        if env.done:
            break
        obs, _, _, _ = env.step(action)
        imgs.append(obs)

    target_image = env.cur_img.astype(float)
    real_observations = []
    for obs, point in zip(imgs, points):
        tar = cut_roi(target_image, point, env.window_size)
        tar = transform.resize(tar, (env.obs_size, env.obs_size, 1))
        tar = np.round(np.clip(tar, 0, 1), decimals=0).squeeze(axis=-1)
        obs[:, :, 2] = tar  # 替换target

        real_observations.append(obs)

    return {'obs': real_observations[:-1],
            'new_obs': real_observations[1:],
            'actions': actions[:len(real_observations) - 1],
            'target_img': target_image,
            'points': points,
            }


if __name__ == '__main__':
    env_config = define_hparams()
    env = ContinuousEnv(env_config)

    margin = 0
    author = '1001-c.pot'
    skeleton_paths = get_skeleton_paths_from_pot(os.path.join(HWDB_DIR, author),
                                                 margin, env_config.image_size, total_strokes=10)
    for i, sp in enumerate(skeleton_paths):
        start_point = sp[0][0]
        _, action_xy = skeleton_path_to_wps(sp, env_config.xy_grid, env_config.image_size, discrete=False)
        action_z = np.random.random(size=(action_xy.shape[0], 1))
        actions = np.concatenate([action_xy, action_z], axis=-1)

        data = collect_data_from_actions(env, actions, start_point)
        import matplotlib.pyplot as plt

        for obs, action in zip(data['obs'], data['actions']):
            plt.imshow(obs[:, :, :3])
            plt.suptitle(action)
            plt.show()
