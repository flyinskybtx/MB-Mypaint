import os
from glob import glob

import numpy as np
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter
from skimage import transform
from tqdm import tqdm

from Data import DATA_DIR
from Data.HWDB.load_HWDB import HWDB_DIR, get_skeleton_paths_from_pot
from Data.data_process_lib import skeleton_path_to_wps, cut_roi
from Env.continuous_env import ContinuousEnv
from script.main_procs.hparams import define_hparams

if __name__ == '__main__':
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = ContinuousEnv(env_config)

    margin = 0
    author = '1001-c.pot'
    skeleton_paths = get_skeleton_paths_from_pot(os.path.join(HWDB_DIR, author),
                                                 margin, env_config.image_size, total_strokes=5000)

    env.reset()
    save_dir = os.path.join(DATA_DIR, 'HWDB/continuous/', author.strip('.pot'))
    os.makedirs(save_dir, exist_ok=True)
    writer = JsonWriter(save_dir)
    batch_builder = SampleBatchBuilder()
    #
    for i, sp in enumerate(tqdm(skeleton_paths)):
        start_point = sp[0][0]
        wps, action_xy = skeleton_path_to_wps(sp, env_config.xy_grid, env_config.image_size, discrete=False)
        action_z = np.random.random(size=(action_xy.shape[0], 1))
        actions = np.concatenate([action_xy, action_z], axis=-1)
        obs = env.reset()
        env.position = [start_point[0], start_point[1], 0]  # 重置笔刷到skeleton 出发点
        obs_history = [obs]

        for action in actions:
            obs, _, _, _, = env.step(action)
            obs_history.append(np.copy(obs))
        target = env.cur_img.astype(float)

        if np.max(target) == 0:  # 目标图像是空图像，有问题
            continue
        for obs, new_obs, point, action in zip(obs_history, obs_history[1:], wps, actions):
            tar = cut_roi(target, point, env_config.window_size)
            tar = transform.resize(tar, (env_config.obs_size, env_config.obs_size, 1))
            tar = np.round(np.clip(tar, 0, 1), decimals=0).squeeze(axis=-1)
            obs[:, :, 2] = tar  # 替换target

            # --- Save
            batch_builder.add_values(
                obs=obs,
                actions=action.tolist(),
                new_obs=new_obs
            )
        writer.write(batch_builder.build_and_reset())

    # for stroke in tqdm(strokes):
    #     zs = np.random.randint(0, 10, size=(brush_config.num_waypoints, 1))
    #     actions = np.concatenate([stroke, zs], axis=-1).tolist()
    #
    #     env.mypaint_painter.reset()
    #     obs, _, _, _ = env.step(actions)
    #
    #     # --- Save
    #     batch_builder.add_values(
    #         obs=obs,
    #
    #         actions=actions,
    #     )
    #     writer.write(batch_builder.build_and_reset())
