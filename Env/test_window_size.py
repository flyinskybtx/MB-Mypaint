import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Data.data_process_lib import refpath_to_actions
from Env.core_config import experimental_config
from Env.windowed_env import WindowedCnnEnv

if __name__ == '__main__':
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'obs_size': experimental_config.obs_size,
        'xy_size': experimental_config.xy_size,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
    }
    env = WindowedCnnEnv(env_config)
    # --------------------------------
    #
    # for i in tqdm(range(1, 10)):
    #     obs = env.reset()  # obs = [cur, prev, tar, z]
    #     cur_cnp, prev_cnp, tar, z = map(lambda x: np.squeeze(x, axis=-1), np.split(obs, 4, axis=-1))
    #     reference_path = env.cur_ref_path
    #     actions = refpath_to_actions(reference_path,
    #                                  step_size=experimental_config.xy_size,
    #                                  action_shape=experimental_config.action_shape)
    #     actions[:10, -1] = 4
    #     actions[10:, -1] = 2
    #
    #     done = False
    #     t = 0
    #
    #     for action in actions:
    #         if done:
    #             break
    #         obs, rew, done, info = env.step(action)
    #         cur, prev, tar, z = map(lambda x: np.squeeze(x, axis=-1), np.split(obs, 4, axis=-1))  # update tar & z
    #         t += 1
    #
    #         plt.imshow(np.concatenate([cur, prev, tar], axis=-1))
    #         plt.title(f'{i}-({t})')
    #         plt.show()

    # ------------ view in whole ----------------- #
    for i in tqdm(range(1, 10)):

        obs = env.reset()  # obs = [cur, prev, tar, z]
        cur_cnp, prev_cnp, tar, z = map(lambda x: np.squeeze(x, axis=-1), np.split(obs, 4, axis=-1))
        reference_path = env.cur_ref_path
        actions = refpath_to_actions(reference_path,
                                     step_size=experimental_config.xy_size,
                                     action_shape=experimental_config.action_shape)
        actions[:10, -1] = 3
        done = False
        t = 0

        prev_img = np.zeros((env.image_size, env.image_size))

        for action in actions:
            if done:
                break
            obs, rew, done, info = env.step(action)
            xx, yy, _ = info['history'][-1]
            half_window = experimental_config.window_size / 2
            bbox = np.zeros((experimental_config.image_size, experimental_config.image_size))
            bbox[int(xx - half_window):int(xx + half_window),
            int(yy - half_window):int(yy + half_window)] = 1
            bbox[int(xx - half_window + 2):int(xx + half_window - 2),
            int(yy - half_window + 2):int(yy + half_window - 2)] = 0
            cur_img = env.agent.get_img((env.image_size, env.image_size))
            tar_img = env.target_image
            t += 1

            plt.imshow(np.stack([bbox, cur_img - prev_img, prev_img], axis=-1))
            plt.title(f'{i}-({t})')
            plt.show()

            prev_img = cur_img
