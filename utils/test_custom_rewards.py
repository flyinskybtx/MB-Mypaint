# Rewards:

from Data.data_process_lib import refpath_to_actions
from Data.Deprecated.core_config import experimental_config
from Env.windowed_env import WindowedCnnEnv
from utils.custom_rewards import *


def test_ref_rewards():
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'xy_grid': experimental_config.xy_grid,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
        'obs_size': experimental_config.obs_size,
    }

    env = WindowedCnnEnv(env_config)
    env.reset()

    cur_img = env.agent.get_img((experimental_config.image_size, experimental_config.image_size))
    tar_img = env.target_image
    done = False

    reference_path = env.cur_ref_path
    actions = refpath_to_actions(reference_path,
                                 xy_size=experimental_config.obs_size,
                                 action_shape=experimental_config.action_shape)

    for action in actions:
        prev_img = np.copy(cur_img)
        if done:
            break
        env.step(action)
        cur_img = env.agent.get_img((experimental_config.image_size, experimental_config.image_size))

        # ------------------
        # delta = cur_img - prev_img
        # print(f'incremental_reward: {incremental_reward(tar_img, delta)}')
        # print(f'incremental_loss: {incremental_loss(tar_img, delta)}')

    history = env.history
    cur_path = np.array([[x, y] for (x, y, z) in history])

    print(f'img_cosine_reward {img_cosine_reward(tar_img, cur_img)}')
    print(f'img_mse_loss {img_mse_loss(tar_img, cur_img)}')
    print(f'scale_loss {scale_loss(tar_img, cur_img)}')
    print(f'iou_reward {iou_reward(tar_img, cur_img)}')
    print(f'curvature_loss {curvature_loss(reference_path, cur_path)}')


def test_random_rewards():
    env_config = {
        'image_size': experimental_config.image_size,
        'window_size': experimental_config.window_size,
        'xy_grid': experimental_config.xy_grid,
        'z_size': experimental_config.z_size,
        'brush_name': experimental_config.brush_name,
        'image_nums': experimental_config.image_nums,
        'action_shape': experimental_config.action_shape,
        'obs_size': experimental_config.obs_size,
    }

    env = WindowedCnnEnv(env_config)
    env.reset()

    cur_img = env.agent.get_img((experimental_config.image_size, experimental_config.image_size))
    tar_img = env.target_image
    done = False

    reference_path = env.cur_ref_path
    actions = np.array([env.action_space.sample() for _ in range(len(reference_path))])
    actions[:5, -1] = 3

    for action in actions:
        prev_img = np.copy(cur_img)
        if done:
            break
        env.step(action)
        cur_img = env.agent.get_img((experimental_config.image_size, experimental_config.image_size))
        delta = cur_img - prev_img

        # ------------------
        print(f'incremental_reward: {incremental_reward(tar_img, delta)}')
        print(f'incremental_loss: {incremental_loss(tar_img, delta)}')

    history = env.history
    cur_path = np.array([[x, y] for (x, y, z) in history])

    print(f'img_cosine_reward {img_cosine_reward(tar_img, cur_img)}')
    print(f'img_mse_loss {img_mse_loss(tar_img, cur_img)}')
    print(f'scale_loss {scale_loss(tar_img, cur_img)}')
    print(f'iou_reward {iou_reward(tar_img, cur_img)}')
    print(f'curvature_loss {curvature_loss(reference_path, cur_path)}')


if __name__ == '__main__':
    test_ref_rewards()
    print('**********************************')
    test_random_rewards()
