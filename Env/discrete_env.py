import os
from glob import glob

import gym
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import transform

from Data import DATA_DIR
from Data.data_process_lib import cut_roi, load_all_stroke_pngs, \
    skeleton_path_to_wps
from Data.png.extract_reference_waypoints import load_all_png_skeleton_waypoints
from Model import AttrDict
from script.main_procs.hparams import define_hparams
# 先不考虑路径的reward，只用最终的reward
from utils.custom_rewards import img_cosine_reward
from utils.mypaint_agent import MypaintPainter


class DiscreteEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        # --- Spaces

        self._x = None
        self._y = None
        self._z = None
        self._step = None

        self.image_size = env_config['image_size']
        self.obs_size = env_config['obs_size']
        self.window_size = env_config['window_size']
        self.z_grid = env_config['z_grid']
        self.xy_grid = env_config['xy_grid']
        self.max_step = env_config['max_step']

        # 环境信息
        self.num_images = env_config['num_images']
        self.images = load_all_stroke_pngs(self.image_size)
        self.skeleton_waypoints = load_all_png_skeleton_waypoints(self.image_size)
        self.mypaint_painter = MypaintPainter(env_config['brush_config'])

        self.action_space = gym.spaces.MultiDiscrete([3, 3, int(1 / self.z_grid) + 1])  # xy delta, z absolute
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 4), dtype=np.float)

    def reset(self, **kwargs):
        """
        obs: cur, prev, tar, zs
        :return:
        """
        image_num = kwargs.setdefault('image_num', None)
        if image_num is None:
            image_num = np.random.randint(low=0, high=self.num_images - 1)

        # --- 设置图像和起始点
        self.target_image = self.images[image_num]
        self.cur_ref_path = self.skeleton_waypoints[image_num]
        self.position = np.concatenate([self.cur_ref_path[0][0], [0]], axis=-1)

        # --- 重置笔刷
        self.mypaint_painter.reset()
        self.mypaint_painter.paint(*self.painter_position)
        self._prev_img = np.zeros((self.image_size, self.image_size))

        # --- 设置信息
        self.done = False
        self._step = 0
        self.history = []
        self.wps_img = np.zeros((self.image_size, self.image_size))

        # --- 输出
        obs = self.obs
        return obs

    def step(self, action: list):
        """

        Args:
            action:

        Returns:

        """
        # --- 更新笔刷位置 ---
        self.update_position(action)

        # --- 执行动作，获取动作前/后图像
        self._prev_img = np.copy(self.cur_img)
        self.mypaint_painter.paint(*self.painter_position)

        # --- 判断结束
        if self._z == 0 or self._step >= self.max_step:
            self.done = True
            self.reward = self.calculate_reward()
        else:
            self.done = False
            self.reward = 0

        # --- 输出
        return self.obs, self.reward, self.done, {}

    def render(self, **kwargs):
        SPLIT_VIEW=True
        if self.done:
            wps_img = scipy.ndimage.convolve(self.wps_img, np.ones((9, 9)), mode='constant', cval=0.0)
            wps_img = np.clip(wps_img, 0, 1)

            fig = plt.figure()
            if self.done:
                fig.suptitle(f"Reward: {self.reward}")
            else:
                fig.suptitle(f"Step {self._step}")
            if not SPLIT_VIEW:
                plt.imshow(np.stack([wps_img, self.cur_img, self.target_image], axis=-1))
                plt.show()
            else:
                plt.imshow(np.concatenate([wps_img, self.cur_img, self.target_image], axis=1))
                plt.show()

    def calculate_reward(self):
        reward = img_cosine_reward(self.target_image, self.cur_img)
        return reward

    def update_position(self, action):
        x = self._x + (action[0] - 1) * self.xy_grid
        y = self._y + (action[1] - 1) * self.xy_grid
        z = action[2] * self.z_grid
        self.position = (x, y, z)

        # 更新路径点图像
        self.wps_img *= 0.9  # 衰减
        self.wps_img[int(self._x), int(self._y)] = 1

    @property
    def position(self):
        return np.array([int(self._x), int(self._y), self._z])

    @position.setter
    def position(self, value):
        self._x = float(np.clip(value[0], 0., self.image_size - 1))
        self._y = float(np.clip(value[1], 0., self.image_size - 1))
        self._z = np.clip(value[2], 0., 1.)

    @property
    def painter_position(self):
        return [self._x / self.image_size, self._y / self.image_size, self._z]

    @property
    def cur_img(self):
        return self.mypaint_painter.get_img(shape=(self.image_size, self.image_size))

    @property
    def obs(self):
        obs = transform.resize(self.window, (self.obs_size, self.obs_size, 3))  # Resize to a smaller obs size
        obs = np.clip(obs, 0, 1)
        obs = np.round(obs, decimals=0)  # BW-wize the obs
        z = self._z * np.ones((self.obs_size, self.obs_size, 1))

        obs = np.concatenate([obs, z], axis=-1)
        return obs

    @property
    def window(self):
        cur = cut_roi(self.cur_img, self.position, self.window_size)
        prev = cut_roi(self._prev_img, self.position, self.window_size)
        tar = cut_roi(self.target_image, self.position, self.window_size)
        window = np.stack([cur, prev, tar], axis=-1)
        return window


# def get_ref_actions(ref_path, xy_grid):
#     """ Reference 2-D path to actions with random Z """
#     delta = refpath[1:] - refpath[:-1]
#     actions = np.round(delta / (2. / (action_shape - 1) * xy_size))
#     actions += action_shape // 2
#     z = np.random.randint(low=0, high=5, size=(actions.shape[0], 1))
#     z[:int(0.5 * len(z))] = np.clip(z[:int(0.5 * len(z))] + 1, 0,
#                                     action_shape - 1)  # let the first half go down and
#     # the last half
#     # go up
#     z[int(0.5 * len(z)):] = np.clip(z[int(0.5 * len(z)):] - 1, 0, action_shape - 1)
#     actions = np.concatenate([actions, z], axis=-1)
#
#     return actions

def test():
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = DiscreteEnv(env_config)

    for i in range(5):
        env.reset()
        wps, action_xy = skeleton_path_to_wps(env.cur_ref_path, env_config.xy_grid, env_config.image_size)
        action_z = np.random.randint(0, int(1 / env_config.z_grid) + 1, size=(action_xy.shape[0], 1))
        actions = np.concatenate([action_xy, action_z], axis=-1)

        for point, action in zip(wps[1:], actions):
            obs, reward, done, _ = env.step(action)
            position = env.position
            print(position[:2], point)

            # plt.imshow(np.concatenate([obs[:, :, 0], obs[:, :, 1], obs[:, :, 2], ], axis=-1))
            # plt.show()
            # env.render()
            if done:
                break

        print()


if __name__ == '__main__':
    test()
