import os
from glob import glob

import gym
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import transform

from Data import DATA_DIR
from Data.data_process_lib import cut_roi, load_all_stroke_pngs, get_startpoint_from_img, skeleton_path_to_wps
from Data.png.extract_reference_waypoints import load_all_png_skeleton_waypoints
from Model import AttrDict
from script.main_procs.hparams import define_hparams
# 先不考虑路径的reward，只用最终的reward
from utils.custom_rewards import rewards_dict
from utils.mypaint_agent import MypaintPainter


class SimulatorContinuousEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        # --- Spaces
        self.image_size = env_config['image_size']
        self.obs_size = env_config['obs_size']
        self.window_size = env_config['window_size']
        self.xy_grid = env_config['xy_grid']  # xy方向最大步长
        self.z_grid = env_config['z_grid']  # z方向最大步长
        self.max_step = env_config['max_step']  # 最大步数停止条件
        self.reward_types = env_config['rewards']

        # 环境信息
        self.num_images = env_config['num_images']
        self.images = load_all_stroke_pngs(self.image_size)  # 读取目标图像
        self.mypaint_painter = MypaintPainter(env_config['brush_config'])  # 配置笔刷

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
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
        self.target_image[np.where(self.target_image > 0)] = 1.0
        start_x, start_y = get_startpoint_from_img(self.target_image)
        self.position = (start_x, start_y, 0)

        # --- 重置笔刷
        self.mypaint_painter.reset()
        self.mypaint_painter.paint(*self.painter_position)
        self._prev_img = np.zeros((self.image_size, self.image_size))

        # --- 设置信息
        self.done = False
        self._step = 0
        self.history = []
        self.wps_img = np.zeros((self.image_size, self.image_size))
        self.wps_img[int(self._x), int(self._y)] = 1
        self.prev_reward = 0

        # --- 输出
        return self.obs

    def step(self, action: list):
        """

        Args:
            action:

        Returns:

        """
        # --- 更新笔刷位置 ---
        self.update_position(action)
        self._step += 1

        # --- 执行动作，获取动作前/后图像
        self._prev_img = np.copy(self.cur_img)
        self.mypaint_painter.paint(*self.painter_position)

        # --- 判断结束
        if self._z == 0 or self._step >= self.max_step:
            self.done = True
        else:
            self.done = False

        # --- Rewards
        rewards = self.calculate_reward()
        cur_reward = 0 + np.sum(list(rewards.values()))
        if np.sum(self.cur_img) - np.sum(self._prev_img) == 0:
            cur_reward -= 0.01  # Punish empty steps

        reward = cur_reward - self.prev_reward
        self.prev_reward = cur_reward

        # --- 输出
        return self.obs, reward, self.done, {'rewards': rewards}

    def render(self, **kwargs):
        if self.done:
            wps_img = scipy.ndimage.convolve(self.wps_img, np.ones((9, 9)), mode='constant', cval=0.0)
            wps_img = np.clip(wps_img, 0, 1)

            fig = plt.figure()
            if self.done:
                reward = 0 + np.sum(list(self.calculate_reward().values()))
                fig.suptitle(f"Reward: {reward}")
            else:
                fig.suptitle(f"Step {self._step}")
            plt.imshow(np.stack([wps_img, self.cur_img, self.target_image], axis=-1))
            plt.show()
            plt.close(fig)

    def calculate_reward(self):
        rewards = {}
        for _type in self.reward_types:
            if 'incremental' in _type:
                pass
            elif 'curvature' in self.reward_types:
                raise NotImplementedError(f'{_type} is not supported currently')
            rewards[_type] = rewards_dict[_type](self.target_image, self.cur_img)
        return rewards

    def update_position(self, action):
        assert -1 <= action[0] <= 1
        assert -1 <= action[1] <= 1
        assert -1 <= action[0] <= 1

        x = self._x + action[0] * self.xy_grid  # 0 为移动量中点，
        y = self._y + action[1] * self.xy_grid
        z = self._z + action[2] * self.z_grid
        self.position = (x, y, z)

        # 更新路径点图像
        self.wps_img *= 0.99  # 衰减
        self.wps_img[int(self._x), int(self._y)] = 1

    @property
    def position(self):
        return np.array([int(self._x), int(self._y), self._z])

    @position.setter
    def position(self, value):
        self._x = float(np.clip(value[0], 0., self.image_size - 1))
        self._y = float(np.clip(value[1], 0., self.image_size - 1))
        self._z = float(np.clip(value[2], 0., 1.))

    @property
    def painter_position(self):
        return [self._x / self.image_size, self._y / self.image_size, self._z]

    @property
    def cur_img(self):
        img = self.mypaint_painter.get_img(shape=(self.image_size, self.image_size))
        img[np.where(img > 0)] = 1.0
        return img

    @property
    def obs(self):
        obs = transform.resize(self.window, (self.obs_size, self.obs_size, 3))  # Resize to a smaller obs size
        obs[np.where(obs > 0) ] = 1.0
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


class ContinuousEnv(gym.Env):
    def __init__(self, env_config: AttrDict):
        # --- Spaces

        self._x = None
        self._y = None
        self._z = None
        self._step = None

        self.image_size = env_config['image_size']
        self.obs_size = env_config['obs_size']
        self.window_size = env_config['window_size']
        self.xy_grid = env_config['xy_grid']  # xy方向最大步长
        self.max_step = env_config['max_step']  # 最大步数停止条件
        self.reward_types = env_config['rewards']

        # 环境信息
        self.num_images = env_config['num_images']
        self.images = load_all_stroke_pngs(self.image_size)  # 读取目标图像
        self.skeleton_waypoints = load_all_png_skeleton_waypoints(self.image_size)  # 读取间隔为1的骨架轨迹
        self.mypaint_painter = MypaintPainter(env_config['brush_config'])  # 配置笔刷

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
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
        startpoint = self.cur_ref_path[0][0]
        self.position = (startpoint[0], startpoint[1], 0)

        # --- 重置笔刷
        self.mypaint_painter.reset()
        self.mypaint_painter.paint(*self.painter_position)
        self._prev_img = np.zeros((self.image_size, self.image_size))

        # --- 设置信息
        self.done = False
        self._step = 0
        self.history = []
        self.wps_img = np.zeros((self.image_size, self.image_size))
        self.wps_img[int(self._x), int(self._y)] = 1
        self.skel_length = np.sum([len(p) for p in self.cur_ref_path])

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
        self._step += 1

        # --- 执行动作，获取动作前/后图像
        self._prev_img = np.copy(self.cur_img)
        self.mypaint_painter.paint(*self.painter_position)

        # --- 判断结束
        if self._z <= 0.1 or self._step >= self.max_step:
            self.done = True
        else:
            self.done = False

        # --- Rewards
        rewards = self.calculate_reward()
        self.reward = 0 + np.sum(list(rewards.values()))

        # --- 输出
        return self.obs, self.reward, self.done, {'rewards': rewards}

    def render(self, **kwargs):
        if self.done:
            wps_img = scipy.ndimage.convolve(self.wps_img, np.ones((9, 9)), mode='constant', cval=0.0)
            wps_img = np.clip(wps_img, 0, 1)

            fig = plt.figure()
            if self.done:
                fig.suptitle(f"Reward: {self.reward}")
            else:
                fig.suptitle(f"Step {self._step}")
            plt.imshow(np.stack([wps_img, self.cur_img, self.target_image], axis=-1))
            plt.show()
            plt.close(fig)

    def calculate_reward(self):
        if not self.done:
            rewards = {}
            for _type in self.reward_types:
                if 'incremental' in _type:
                    rewards[_type] = rewards_dict[_type](self.target_image, self.cur_img - self._prev_img)
        else:
            rewards = {}
            for _type in self.reward_types:
                if 'incremental' in _type:
                    pass
                elif 'curvature' in self.reward_types:
                    raise NotImplementedError(f'{_type} is not supported currently')
                rewards[_type] = rewards_dict[_type](self.target_image, self.cur_img)

        return rewards

    def update_position(self, action):

        x = self._x + 2 * (action[0] - 0.5) * self.xy_grid  # 0.5为移动量中点，
        y = self._y + 2 * (action[1] - 0.5) * self.xy_grid
        z = action[2]
        self.position = (x, y, z)

        # 更新路径点图像
        self.wps_img *= 0.99  # 衰减
        self.wps_img[int(self._x), int(self._y)] = 1

    @property
    def position(self):
        return np.array([int(self._x), int(self._y), self._z])

    @position.setter
    def position(self, value):
        self._x = float(np.clip(value[0], 0., self.image_size - 1))
        self._y = float(np.clip(value[1], 0., self.image_size - 1))
        self._z = float(np.clip(value[2], 0., 1.))

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


def test():
    # Settings
    env_config = define_hparams()
    physics = glob(os.path.join(DATA_DIR, f'offline/slow_ink/Physics'))[0]
    env_config.brush_info_file = os.path.join(physics, 'BrushInfo.myb')

    env = ContinuousEnv(env_config)

    for i in range(5):
        env.reset()
        action_xy = skeleton_path_to_wps(env.cur_ref_path, env_config.xy_grid)
        action_z = np.random.random(size=(action_xy.shape[0], 1))
        actions = np.concatenate([action_xy, action_z], axis=-1)

        for action in actions:
            obs, reward, done, _ = env.step(action)
            position = env.position

            # plt.imshow(np.concatenate([obs[:, :, 0], obs[:, :, 1], obs[:, :, 2], ], axis=-1))
            # plt.show()
            env.render()
            if done:
                break

        print()


def test_robot_continuous_env():
    config = AttrDict()

    config.brush_config = AttrDict()
    config.brush_config.brush_name = 'custom/slow_ink'
    config.brush_config.agent_name = 'Physics'

    continuous_env_config = AttrDict()
    continuous_env_config.image_size = 768  # 32 * 24
    continuous_env_config.window_size = 168  # 84 * 2
    continuous_env_config.obs_size = 64
    continuous_env_config.xy_grid = 32
    continuous_env_config.z_grid = 0.1
    continuous_env_config.max_step = 100
    continuous_env_config.num_images = 64
    continuous_env_config.rewards = [
        'img_cosine_reward',
    ]
    continuous_env_config.split_view = False

    config.continuous_env_config = continuous_env_config
    config.continuous_env_config.brush_config = config.brush_config

    env = SimulatorContinuousEnv(config.continuous_env_config)
    for i in range(5):
        env.reset()
        while not env.done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            print(reward)
            # plt.imshow(np.concatenate([obs[:, :, 0], obs[:, :, 1], obs[:, :, 2], ], axis=-1))
            # plt.show()
            env.render()
            if done:
                break

        print()


if __name__ == '__main__':
    test_robot_continuous_env()
